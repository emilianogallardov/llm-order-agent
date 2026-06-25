"""Eval suite: the cases the system is designed to get right, plus the adversarial
cases an external review (codex) proved it must.

These assert on the DETERMINISTIC output, so they are golden-set regression tests,
not vibes. The LLM step is replaced by a MockClient returning a fixed extraction,
which makes the suite fast, free, and reproducible. The order_text passed to
process() is realistic, because the validator now grounds model-extracted facts
against it. The same prompts and pipeline run live (see run.py --live, stress_test.py).

Run standalone:  python -m evals.test_cases
Run with pytest: pytest -q
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from order_agent.agent import OrderAgent          # noqa: E402
from order_agent.catalog import Catalog            # noqa: E402
from order_agent.llm import MockClient             # noqa: E402
from order_agent.schema import OrderStatus         # noqa: E402

_BASE = json.loads((Path(__file__).resolve().parent.parent / "catalog.json").read_text())


def _catalog(data: dict | None = None) -> Catalog:
    return Catalog(copy.deepcopy(data or _BASE))


def _mock(extraction: dict) -> MockClient:
    return MockClient(lambda system, user: extraction)


def _line(**kw) -> dict:
    base = {
        "raw_text": "", "product_family": None, "stated_attributes": {},
        "product_id": None, "vendor_query": None, "quantity": None, "uom": None,
    }
    base.update(kw)
    return base


# --- Test Case 1: happy path (alias + grounded attributes + contract pricing) -
def test_happy_path_resolves_to_865():
    text = ("50 lbs sharp white cheddar from the main dairy co, "
            "20 cases cleaned romaine from the green produce distributor")
    extraction = {"lines": [
        _line(raw_text="50 lbs sharp white cheddar", product_family="cheddar",
              stated_attributes={"flavor": "sharp", "form": "block"},
              product_id="PRD-CHED-SHARP-WHITE", vendor_query="the main dairy co",
              quantity=50, uom="lb"),
        _line(raw_text="20 cases cleaned romaine", product_family="romaine",
              stated_attributes={"form": "cleaned"}, product_id="PRD-ROM-CLEANED",
              vendor_query="the green produce distributor", quantity=20, uom="case"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)

    assert result.status == OrderStatus.READY_FOR_STAGING
    assert str(result.order_total) == "865.00"
    assert result.lines[0].product_id == "PRD-CHED-SHARP-WHITE"
    assert str(result.lines[0].line_total) == "225.00"
    assert result.lines[1].product_id == "PRD-ROM-CLEANED"
    assert str(result.lines[1].line_total) == "640.00"


# --- Test Case 2: ambiguous variant + vendor must BLOCK finalization ----------
def test_ambiguous_order_requires_clarification():
    text = "50 lbs cheddar from the dairy supplier, 20 cases cleaned romaine from Green"
    extraction = {"lines": [
        _line(raw_text="50 lbs cheddar", product_family="cheddar", stated_attributes={},
              vendor_query="the dairy supplier", quantity=50, uom="lb"),
        _line(raw_text="20 cases cleaned romaine", product_family="romaine",
              stated_attributes={"form": "cleaned"}, vendor_query="Green", quantity=20, uom="case"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)

    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert result.lines == []
    assert "cheddar variant" in result.blocked_fields
    assert "supplier entity" in result.blocked_fields


# --- The validator owns product resolution ------------------------------------
def test_confidently_wrong_model_pick_is_caught():
    text = "40 lbs cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar", stated_attributes={},
              product_id="PRD-CHED-SHARP-WHITE", vendor_query="the main dairy co",
              quantity=40, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)

    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert any("ambiguous_variant" in r for r in result.reasons)


def test_attributes_override_a_wrong_model_pick():
    text = "50 lbs mild shredded cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "mild", "form": "shred"},
              product_id="PRD-CHED-SHARP-WHITE", vendor_query="the main dairy co",
              quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)

    assert result.status == OrderStatus.READY_FOR_STAGING
    assert result.lines[0].product_id == "PRD-CHED-MILD-SHRED"
    assert str(result.order_total) == "190.00"


# --- Grounding: the model can't override what the buyer actually said ----------
def test_quantity_inflation_is_blocked():
    """Text says 50 lbs; a bad model emits 5000. Ungrounded quantity blocks."""
    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=5000, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "quantity_uom_not_grounded" in result.reasons


def test_vendor_swap_is_blocked():
    """Text names an unapproved vendor; the model swaps in an approved one not in
    the text. The vendor reference isn't grounded, so it blocks."""
    text = "60 lbs ground beef 80/20 from backalley wholesale meats"
    extraction = {"lines": [
        _line(raw_text=text, product_family="beef",
              stated_attributes={"cut": "ground", "blend": "80/20"},
              vendor_query="premier", quantity=60, uom="lb"),  # "premier" not in text
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "vendor_not_grounded" in result.reasons


def test_attribute_swap_forces_clarification():
    """Text says 'sharp white'; the model emits 'mild'/'shred'. Those aren't in
    the text, so they're dropped, the variant is ambiguous, and it clarifies."""
    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "mild", "form": "shred"},
              vendor_query="the main dairy co", quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert any("ambiguous_variant" in r for r in result.reasons)


def test_unsupported_constraint_is_blocked():
    """Buyer asked for 'organic', which the catalog can't express. Don't silently
    stage a non-organic SKU; block for clarification."""
    text = "50 lbs organic sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp", "form": "block", "organic": "true"},
              vendor_query="the main dairy co", quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert any("unsupported_constraint" in r for r in result.reasons)


# --- Fail closed, never crash, on malformed model output ----------------------
def test_empty_order_is_blocked():
    result = OrderAgent(catalog=_catalog(), client=_mock({"lines": []})).process("anything")
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "empty_order" in result.reasons


def test_malformed_shapes_fail_closed():
    text = "1 lb sharp white cheddar from the main dairy co"
    cases = [
        {"lines": "not a list"},                                  # top-level shape
        {"lines": [_line(raw_text=text, product_family="cheddar",
                         stated_attributes={"flavor": "sharp"},
                         vendor_query="the main dairy co", quantity=1, uom=123)]},  # uom not str
        {"lines": [_line(raw_text=text, product_family="cheddar",
                         stated_attributes={"flavor": "sharp"},
                         vendor_query="the main dairy co", quantity=float("inf"), uom="lb")]},
    ]
    for extraction in cases:
        result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
        assert result.status == OrderStatus.VALIDATION_BLOCKED, extraction
        assert result.lines == []


# --- Pricing integrity --------------------------------------------------------
def test_future_dated_contract_is_ignored():
    """A future-effective (or duplicate) contract inserted first must not win.
    Only contracts effective on/before today are eligible; latest wins."""
    data = copy.deepcopy(_BASE)
    data["contracts"].insert(0, {
        "contract_id": "CT-FUTURE", "product_id": "PRD-CHED-SHARP-WHITE",
        "parent_vendor_id": "SUP-DAIRY-PARENT-001", "uom": "lb",
        "unit_price": "999.99", "effective": "2099-01-01",
    })
    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(data), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.READY_FOR_STAGING
    assert str(result.lines[0].contract_unit_price) == "4.50"
    assert str(result.order_total) == "225.00"


def test_duplicate_active_contract_is_blocked():
    """Two active contracts with the same effective date is a real ambiguity.
    Refuse to price rather than pick by list order."""
    data = copy.deepcopy(_BASE)
    for cid, price in (("CT-DUPE-A", "1.00"), ("CT-DUPE-B", "9.00")):
        data["contracts"].insert(0, {
            "contract_id": cid, "product_id": "PRD-CHED-SHARP-WHITE",
            "parent_vendor_id": "SUP-DAIRY-PARENT-001", "uom": "lb",
            "unit_price": price, "effective": "2026-02-01",
        })
    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(data), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "ambiguous_contract" in result.reasons


# --- Structural change / non-determinism (REQUIRED) ---------------------------
def test_supplier_rename_survives_via_canonical_ids():
    drifted = copy.deepcopy(_BASE)
    for v in drifted["vendors"]:
        if v["id"] == "SUP-DAIRYCORP":
            v["name"] = "DairyCorp (a division of DairyCorp Holdings LLC)"

    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=50, uom="lb"),
    ]}
    payloads = []
    for _ in range(5):
        result = OrderAgent(catalog=_catalog(drifted), client=_mock(extraction)).process(text)
        payloads.append(json.dumps(result.to_dict(), sort_keys=True))

    assert len(set(payloads)) == 1
    assert str(result.lines[0].contract_unit_price) == "4.50"
    assert result.lines[0].parent_vendor_id == "SUP-DAIRY-PARENT-001"


def test_broken_supplier_hierarchy_fails_closed():
    broken = copy.deepcopy(_BASE)
    for v in broken["vendors"]:
        if v["id"] == "SUP-DAIRYCORP":
            v["parent_id"] = "SUP-DAIRYCORP-HOLDINGS-LLC"  # no contract bound here
    text = "50 lbs sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=50, uom="lb"),
    ]}
    result = OrderAgent(catalog=_catalog(broken), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "supplier_parent_entity_unresolved" in result.reasons


# --- UOM canonicalization vs mismatch -----------------------------------------
def test_uom_synonyms_are_canonicalized():
    text = "50 pounds sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=50, uom="lbs"),
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.READY_FOR_STAGING
    assert result.lines[0].uom == "lb"
    assert str(result.order_total) == "225.00"


def test_uom_mismatch_is_blocked():
    text = "20 cases sharp white cheddar from the main dairy co"
    extraction = {"lines": [
        _line(raw_text=text, product_family="cheddar",
              stated_attributes={"flavor": "sharp"}, vendor_query="the main dairy co",
              quantity=20, uom="case"),  # contract is per lb
    ]}
    result = OrderAgent(catalog=_catalog(), client=_mock(extraction)).process(text)
    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "uom_mismatch" in result.reasons


# --- Tiny runner so the suite works without pytest installed ------------------
def _main() -> int:
    tests = [
        test_happy_path_resolves_to_865,
        test_ambiguous_order_requires_clarification,
        test_confidently_wrong_model_pick_is_caught,
        test_attributes_override_a_wrong_model_pick,
        test_quantity_inflation_is_blocked,
        test_vendor_swap_is_blocked,
        test_attribute_swap_forces_clarification,
        test_unsupported_constraint_is_blocked,
        test_empty_order_is_blocked,
        test_malformed_shapes_fail_closed,
        test_future_dated_contract_is_ignored,
        test_duplicate_active_contract_is_blocked,
        test_supplier_rename_survives_via_canonical_ids,
        test_broken_supplier_hierarchy_fails_closed,
        test_uom_synonyms_are_canonicalized,
        test_uom_mismatch_is_blocked,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Eval suite: the cases the system is designed to get right.

These assert on the DETERMINISTIC output, so they are golden-set regression tests,
not vibes. The LLM step is replaced by a MockClient returning a fixed extraction,
which makes the suite fast, free, and reproducible. The exact same prompts and
pipeline run live through the real client (see run.py --live and stress_test.py).

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


# --- Test Case 1: happy path (alias + attributes + contract pricing) ----------
def test_happy_path_resolves_to_865():
    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs sharp cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "sharp", "form": "block"},
                "product_id": "PRD-CHED-SHARP-WHITE",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
            },
            {
                "raw_text": "20 cases cleaned romaine from the green produce distributor",
                "product_family": "romaine",
                "stated_attributes": {"form": "cleaned"},
                "product_id": "PRD-ROM-CLEANED",
                "vendor_query": "the green produce distributor",
                "quantity": 20,
                "uom": "case",
            },
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.READY_FOR_STAGING
    assert str(result.order_total) == "865.00"
    assert len(result.lines) == 2

    cheddar = result.lines[0]
    assert cheddar.product_id == "PRD-CHED-SHARP-WHITE"
    assert cheddar.vendor_id == "SUP-DAIRYCORP"
    assert str(cheddar.contract_unit_price) == "4.50"
    assert str(cheddar.line_total) == "225.00"

    romaine = result.lines[1]
    assert romaine.product_id == "PRD-ROM-CLEANED"
    assert str(romaine.line_total) == "640.00"


# --- Test Case 2: ambiguous variant + vendor must BLOCK finalization ----------
def test_ambiguous_order_requires_clarification():
    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs cheddar from the dairy supplier",
                "product_family": "cheddar",
                "stated_attributes": {},  # bare "cheddar": no distinguishing attrs
                "product_id": None,
                "vendor_query": "the dairy supplier",  # not in alias table
                "quantity": 50,
                "uom": "lb",
            },
            {
                "raw_text": "20 cases cleaned romaine from Green",
                "product_family": "romaine",
                "stated_attributes": {"form": "cleaned"},
                "product_id": "PRD-ROM-CLEANED",
                "vendor_query": "Green",  # too vague to resolve
                "quantity": 20,
                "uom": "case",
            },
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(ambiguous order text)")

    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert result.lines == []            # fail closed: nothing staged
    assert result.order_total is None
    assert "cheddar variant" in result.blocked_fields
    assert "supplier entity" in result.blocked_fields
    assert result.clarification          # a question was drafted


# --- The improvement: deterministic layer owns product resolution -------------
def test_confidently_wrong_model_pick_is_caught():
    """The model commits to a specific SKU, but the buyer only said 'cheddar'.
    Because the stated attributes don't single out one product, the validator
    must clarify rather than trust the model's (over)confident product_id."""
    extraction = {
        "lines": [
            {
                "raw_text": "40 lbs cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {},  # nothing distinguishing was actually said
                "product_id": "PRD-CHED-SHARP-WHITE",  # model guessed anyway
                "vendor_query": "the main dairy co",
                "quantity": 40,
                "uom": "lb",
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.CLARIFICATION_REQUIRED
    assert result.lines == []
    assert any("ambiguous_variant" in r for r in result.reasons)


def test_attributes_override_a_wrong_model_pick():
    """The model named the wrong variant (sharp) but the text said 'mild shredded'.
    The validator resolves by attributes, so it lands on the mild shred SKU and
    its $3.80 contract, not the model's pick."""
    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs mild shredded cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "mild", "form": "shred"},
                "product_id": "PRD-CHED-SHARP-WHITE",  # wrong; attributes win
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.READY_FOR_STAGING
    assert result.lines[0].product_id == "PRD-CHED-MILD-SHRED"
    assert str(result.lines[0].contract_unit_price) == "3.80"
    assert str(result.order_total) == "190.00"


# --- Test Case 3: structural change / non-determinism (REQUIRED) ---------------
def test_supplier_rename_survives_via_canonical_ids():
    """DairyCorp is reorganized (display name changes, a holding company appears)
    but the contract stays bound to the same parent id. Id-based resolution must
    still find $4.50 and produce a byte-stable payload across repeated runs."""
    drifted = copy.deepcopy(_BASE)
    for v in drifted["vendors"]:
        if v["id"] == "SUP-DAIRYCORP":
            v["name"] = "DairyCorp (a division of DairyCorp Holdings LLC)"
            # parent_id stays SUP-DAIRY-PARENT-001; contract unchanged.

    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs sharp cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "sharp", "form": "block"},
                "product_id": "PRD-CHED-SHARP-WHITE",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
            }
        ]
    }

    payloads = []
    for _ in range(5):  # re-run to prove stability under repeated calls
        agent = OrderAgent(catalog=_catalog(drifted), client=_mock(extraction))
        result = agent.process("(order text)")
        payloads.append(json.dumps(result.to_dict(), sort_keys=True))

    assert len(set(payloads)) == 1       # identical every run
    result = agent.process("(order text)")
    assert result.status == OrderStatus.READY_FOR_STAGING
    assert str(result.lines[0].contract_unit_price) == "4.50"
    assert str(result.order_total) == "225.00"
    assert result.lines[0].parent_vendor_id == "SUP-DAIRY-PARENT-001"


def test_broken_supplier_hierarchy_fails_closed():
    """Same product, but DairyCorp is reparented under an entity with no contract
    binding. The hierarchy can't be resolved, so the system blocks rather than
    fabricating a price."""
    broken = copy.deepcopy(_BASE)
    for v in broken["vendors"]:
        if v["id"] == "SUP-DAIRYCORP":
            v["parent_id"] = "SUP-DAIRYCORP-HOLDINGS-LLC"  # no contract bound here

    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs sharp cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "sharp", "form": "block"},
                "product_id": "PRD-CHED-SHARP-WHITE",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(broken), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert result.lines == []
    assert "supplier_parent_entity_unresolved" in result.reasons


def test_uom_synonyms_are_canonicalized():
    """Humans type 'lbs'/'cases'; the catalog stores 'lb'/'case'. Synonyms must
    canonicalize so a correct order isn't blocked over plurals."""
    extraction = {
        "lines": [
            {
                "raw_text": "50 pounds sharp cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "sharp", "form": "block"},
                "product_id": "PRD-CHED-SHARP-WHITE",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lbs",  # synonym of the contract's "lb"
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.READY_FOR_STAGING
    assert result.lines[0].uom == "lb"            # canonicalized
    assert str(result.order_total) == "225.00"


def test_uom_mismatch_is_blocked():
    """A weight price must never be multiplied by a case count. Wrong UOM blocks."""
    extraction = {
        "lines": [
            {
                "raw_text": "20 cases sharp cheddar from the main dairy co",
                "product_family": "cheddar",
                "stated_attributes": {"flavor": "sharp", "form": "block"},
                "product_id": "PRD-CHED-SHARP-WHITE",
                "vendor_query": "the main dairy co",
                "quantity": 20,
                "uom": "case",  # contract is priced per lb
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert "uom_mismatch" in result.reasons


# --- Tiny runner so the suite works without pytest installed ------------------
def _main() -> int:
    tests = [
        test_happy_path_resolves_to_865,
        test_ambiguous_order_requires_clarification,
        test_confidently_wrong_model_pick_is_caught,
        test_attributes_override_a_wrong_model_pick,
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

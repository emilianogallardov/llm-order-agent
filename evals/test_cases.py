"""Eval suite: the three cases the system is designed to get right.

These assert on the DETERMINISTIC output, so they are golden-set regression tests,
not vibes. The LLM step is replaced by a MockClient returning a fixed extraction,
which makes the suite fast, free, and reproducible. The exact same prompts and
pipeline run live through the real client (see run.py --live).

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


# --- Test Case 1: happy path (alias + contract pricing) -----------------------
def test_happy_path_resolves_to_865():
    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs sharp cheddar from the main dairy co",
                "product_id": "PRD-CHED-SHARP-WHITE",
                "product_family": "cheddar",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
                "missing_attributes": [],
            },
            {
                "raw_text": "20 cases romaine from the green produce distributor",
                "product_id": "PRD-ROM-CLEANED",
                "product_family": "romaine",
                "vendor_query": "the green produce distributor",
                "quantity": 20,
                "uom": "case",
                "missing_attributes": [],
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
    # A faithful model flags what it can't pin down instead of guessing.
    extraction = {
        "lines": [
            {
                "raw_text": "50 lbs cheddar from the dairy supplier",
                "product_id": None,
                "product_family": "cheddar",
                "vendor_query": "the dairy supplier",
                "quantity": 50,
                "uom": "lb",
                "missing_attributes": ["flavor", "form"],
            },
            {
                "raw_text": "20 cases romaine from Green",
                "product_id": "PRD-ROM-CLEANED",
                "product_family": "romaine",
                "vendor_query": "Green",
                "quantity": 20,
                "uom": "case",
                "missing_attributes": [],
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
                "product_id": "PRD-CHED-SHARP-WHITE",
                "product_family": "cheddar",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
                "missing_attributes": [],
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
                "product_id": "PRD-CHED-SHARP-WHITE",
                "product_family": "cheddar",
                "vendor_query": "the main dairy co",
                "quantity": 50,
                "uom": "lb",
                "missing_attributes": [],
            }
        ]
    }
    agent = OrderAgent(catalog=_catalog(broken), client=_mock(extraction))
    result = agent.process("(order text)")

    assert result.status == OrderStatus.VALIDATION_BLOCKED
    assert result.lines == []
    assert "supplier_parent_entity_unresolved" in result.reasons


def test_uom_mismatch_is_blocked():
    """A weight price must never be multiplied by a case count. Wrong UOM blocks."""
    extraction = {
        "lines": [
            {
                "raw_text": "20 cases sharp cheddar from the main dairy co",
                "product_id": "PRD-CHED-SHARP-WHITE",
                "product_family": "cheddar",
                "vendor_query": "the main dairy co",
                "quantity": 20,
                "uom": "case",  # contract is priced per lb
                "missing_attributes": [],
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
        test_supplier_rename_survives_via_canonical_ids,
        test_broken_supplier_hierarchy_fails_closed,
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

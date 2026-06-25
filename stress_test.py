"""Live stress test: jumbled, real-world-messy orders through the REAL model.

Sends sloppy human messages through the live extraction model + deterministic
validators and checks that the system selects the right product variant and the
right vendor among many lookalikes, and blocks/clarifies when it should.

    ANTHROPIC_API_KEY=... MODEL_ORDER_EXTRACTION=claude-sonnet-4-6 \
        ./.venv/bin/python stress_test.py
"""

from __future__ import annotations

import json
import sys

from order_agent.agent import OrderAgent
from order_agent.llm import LLMClient
from order_agent.selector import get_execution_config

# (message, expected_status, {product_id: line_total} or None, expected_order_total or None)
CASES = [
    (
        "yo can u throw together 50 lbs of that sharp white cheddar from the main dairy co, "
        "plus 20 cases cleaned romaine from the green produce distributor — keep the cheese at our contract rate \U0001f64f",
        "ready_for_staging",
        {"PRD-CHED-SHARP-WHITE": "225.00", "PRD-ROM-CLEANED": "640.00"},
        "865.00",
    ),
    (
        "need like 30 lbs ground beef the 80/20 from premier and a case of evoo from restaurant depot asap",
        "ready_for_staging",
        {"PRD-BEEF-GRND-8020": "126.00", "PRD-OIL-EVOO": "72.00"},
        "198.00",
    ),
    (
        "can i get some cheddar, 40 lbs, from the dairy guys",
        "clarification_required",
        None,
        None,
    ),
    (
        "25 cases romaine hearts from valley harvest pls",
        "ready_for_staging",
        {"PRD-ROM-HEARTS": "725.00"},
        "725.00",
    ),
    (
        "100 lbs boneless skinless chicken breast from premier meats, and 10 bags of all purpose flour from bulk pantry",
        "ready_for_staging",
        {"PRD-CHKN-BRST-BNLS": "290.00", "PRD-FLOUR-AP": "185.00"},
        "475.00",
    ),
    (
        "lemme grab 20 cases of the sharp white cheddar from main dairy co",  # per-lb item ordered in cases
        "validation_blocked",
        None,
        None,
    ),
    (
        "15 lbs mozzarella shred from golden state creamery",
        "ready_for_staging",
        {"PRD-MOZZ-WM-SHRED": "59.25"},
        "59.25",
    ),
    (
        "send over 12 cases roma tomatoes from sunfresh and 8 cases iceberg from freshgreen",
        "ready_for_staging",
        {"PRD-TOM-ROMA": "297.00", "PRD-ICEBERG": "212.00"},
        "509.00",
    ),
    (
        "i need 60 lbs ground beef 80/20 from backalley wholesale meats",  # off-roster vendor
        "clarification_required",
        None,
        None,
    ),
]


def check(result_dict: dict, exp_status: str, exp_lines, exp_total) -> tuple[bool, str]:
    if result_dict["status"] != exp_status:
        return False, f"status={result_dict['status']} (wanted {exp_status})"
    if exp_total is not None and result_dict["order_total"] != exp_total:
        return False, f"total={result_dict['order_total']} (wanted {exp_total})"
    if exp_lines is not None:
        got = {ln["product_id"]: ln["line_total"] for ln in result_dict["lines"]}
        if got != exp_lines:
            return False, f"lines={got} (wanted {exp_lines})"
    return True, "ok"


def main() -> int:
    cfg = get_execution_config("order_extraction")
    print(f"# live model = {cfg.model} ({cfg.model_config.provider})\n")

    agent = OrderAgent(client=LLMClient())  # real client from env key
    passed = 0
    for i, (msg, exp_status, exp_lines, exp_total) in enumerate(CASES, 1):
        result = agent.process(msg)
        d = result.to_dict()
        ok, why = check(d, exp_status, exp_lines, exp_total)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] case {i}: {why}")
        print(f"       msg: {msg[:88]}")
        if d["status"] == "ready_for_staging":
            picks = ", ".join(f"{ln['product_id']}@{ln['contract_unit_price']}x{ln['quantity']}{ln['uom']}={ln['line_total']}"
                              for ln in d["lines"])
            print(f"       -> {d['status']} total={d['order_total']} | {picks}")
        else:
            print(f"       -> {d['status']} | blocked={d['blocked_fields']} reasons={d['reasons']}")
        print()

    print(f"{passed}/{len(CASES)} live cases passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())

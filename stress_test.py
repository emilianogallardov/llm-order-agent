"""Live stress test: jumbled, real-world-messy orders through the REAL model.

Sends sloppy human messages through the live extraction model + deterministic
validators and checks the system selects the right product variant and vendor
among many lookalikes, and blocks/clarifies when it should. Runs the whole set
multiple times to surface flakiness (a case that passes one run and fails another
is a reliability bug, not a pass).

    ANTHROPIC_API_KEY=... MODEL_ORDER_EXTRACTION=claude-sonnet-4-6 \
        ./.venv/bin/python stress_test.py [runs]
"""

from __future__ import annotations

import sys

from order_agent.agent import OrderAgent
from order_agent.llm import LLMClient
from order_agent.selector import get_execution_config

READY = "ready_for_staging"
CLARIFY = "clarification_required"
BLOCKED = "validation_blocked"
NOT_READY = "not_ready"  # accept either clarify or block

# (message, expected_status, {product_id: line_total} or None, expected_order_total or None)
CASES = [
    ("yo can u throw together 50 lbs of that sharp white cheddar from the main dairy co, "
     "plus 20 cases cleaned romaine from the green produce distributor — keep the cheese at contract rate \U0001f64f",
     READY, {"PRD-CHED-SHARP-WHITE": "225.00", "PRD-ROM-CLEANED": "640.00"}, "865.00"),

    ("need like 30 lbs ground beef the 80/20 from premier and a case of evoo from restaurant depot asap",
     READY, {"PRD-BEEF-GRND-8020": "126.00", "PRD-OIL-EVOO": "72.00"}, "198.00"),

    ("can i get some cheddar, 40 lbs, from the dairy guys",
     CLARIFY, None, None),

    ("25 cases romaine hearts from valley harvest pls",
     READY, {"PRD-ROM-HEARTS": "725.00"}, "725.00"),

    ("100 lbs boneless skinless chicken breast from premier meats, and 10 bags of all purpose flour from bulk pantry",
     READY, {"PRD-CHKN-BRST-BNLS": "290.00", "PRD-FLOUR-AP": "185.00"}, "475.00"),

    ("lemme grab 20 cases of the sharp white cheddar from main dairy co",  # per-lb item ordered in cases
     BLOCKED, None, None),

    ("15 lbs mozzarella shred from golden state creamery",
     READY, {"PRD-MOZZ-WM-SHRED": "59.25"}, "59.25"),

    ("send over 12 cases roma tomatoes from sunfresh and 8 cases iceberg from freshgreen",
     READY, {"PRD-TOM-ROMA": "297.00", "PRD-ICEBERG": "212.00"}, "509.00"),

    ("i need 60 lbs ground beef 80/20 from backalley wholesale meats",  # off-roster vendor
     CLARIFY, None, None),

    # --- harder probes ---------------------------------------------------------
    ("hey hope you're well — when you get a sec, grab 50 lbs of sharp white cheddar at our contract rate",
     CLARIFY, None, None),  # no vendor named at all

    ("20 lbs swiss cheese from golden state creamery",  # product not in catalog
     NOT_READY, None, None),

    ("30 LBS of SHARP WHITE CHEDDER FROM THE MAIN DAIRY CO",  # typo + uppercase
     READY, {"PRD-CHED-SHARP-WHITE": "135.00"}, "135.00"),

    ("12.5 lbs ground beef 80/20 from premier",  # decimal quantity
     READY, {"PRD-BEEF-GRND-8020": "52.50"}, "52.50"),

    ("0 cases romaine hearts from valley harvest",  # non-positive quantity
     NOT_READY, None, None),

    ("10 lbs shredded cheddar from the main dairy co",  # 'shredded' -> mild shred SKU @3.80
     READY, {"PRD-CHED-MILD-SHRED": "38.00"}, "38.00"),

    ("big one: 50 lbs sharp white cheddar from the main dairy co, 20 cases cleaned romaine from the green produce "
     "distributor, 30 lbs ground beef 80/20 from premier, and a case of evoo from restaurant depot",
     READY,
     {"PRD-CHED-SHARP-WHITE": "225.00", "PRD-ROM-CLEANED": "640.00",
      "PRD-BEEF-GRND-8020": "126.00", "PRD-OIL-EVOO": "72.00"}, "1063.00"),
]


def check(d: dict, exp_status: str, exp_lines, exp_total) -> tuple[bool, str]:
    if exp_status == NOT_READY:
        if d["status"] == READY:
            return False, f"status={d['status']} (wanted not-ready)"
        return True, "ok"
    if d["status"] != exp_status:
        return False, f"status={d['status']} (wanted {exp_status}) reasons={d.get('reasons')}"
    if exp_total is not None and d["order_total"] != exp_total:
        return False, f"total={d['order_total']} (wanted {exp_total})"
    if exp_lines is not None:
        got = {ln["product_id"]: ln["line_total"] for ln in d["lines"]}
        if got != exp_lines:
            return False, f"lines={got} (wanted {exp_lines})"
    return True, "ok"


def main() -> int:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cfg = get_execution_config("order_extraction")
    print(f"# live model = {cfg.model} ({cfg.model_config.provider}) | {runs} runs x {len(CASES)} cases\n")

    agent = OrderAgent(client=LLMClient())
    results: dict[int, list[bool]] = {i: [] for i in range(len(CASES))}
    fail_detail: dict[int, str] = {}

    for r in range(1, runs + 1):
        round_pass = 0
        for i, (msg, exp_status, exp_lines, exp_total) in enumerate(CASES):
            d = agent.process(msg).to_dict()
            ok, why = check(d, exp_status, exp_lines, exp_total)
            results[i].append(ok)
            round_pass += ok
            if not ok:
                fail_detail[i] = f"{why} | msg={msg[:70]!r}"
        print(f"run {r}: {round_pass}/{len(CASES)} passed")

    print("\n--- per-case stability across runs ---")
    flaky = 0
    perfect = 0
    for i in range(len(CASES)):
        n_ok = sum(results[i])
        if n_ok == runs:
            perfect += 1
        else:
            tag = "FLAKY" if 0 < n_ok < runs else "FAIL "
            flaky += 1
            print(f"  [{tag}] case {i+1}: {n_ok}/{runs} ok  -> {fail_detail.get(i, '')}")
    print(f"\n{perfect}/{len(CASES)} cases passed every run; {flaky} need attention")
    return 0 if flaky == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

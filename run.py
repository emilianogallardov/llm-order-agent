"""CLI entry point.

    # offline demo (no API key needed) — uses a built-in mock extraction
    python run.py --demo

    # live — calls the model selected for the order_extraction task
    OPENAI_API_KEY=...  python run.py --live "50 lbs sharp cheddar from the main dairy co, 20 cases romaine from the green produce distributor"
"""

from __future__ import annotations

import argparse
import json
import sys

from order_agent.agent import OrderAgent
from order_agent.catalog import Catalog
from order_agent.llm import MockClient
from order_agent.selector import get_execution_config

_DEMO_EXTRACTION = {
    "lines": [
        {
            "raw_text": "50 lbs sharp cheddar from the main dairy co",
            "product_family": "cheddar",
            "stated_attributes": {"flavor": "sharp"},
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


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM order agent")
    parser.add_argument("order", nargs="?", help="free-text order")
    parser.add_argument("--demo", action="store_true", help="run offline with a mock extraction")
    parser.add_argument("--live", action="store_true", help="call the real model")
    args = parser.parse_args()

    cfg = get_execution_config("order_extraction")
    print(f"# task=order_extraction  model={cfg.model}  provider={cfg.model_config.provider}\n",
          file=sys.stderr)

    if args.demo or not args.live:
        agent = OrderAgent(catalog=Catalog.load(), client=MockClient(lambda s, u: _DEMO_EXTRACTION))
        order_text = args.order or (
            "50 lbs sharp cheddar from the main dairy co, "
            "20 cases cleaned romaine from the green produce distributor"
        )
    else:
        if not args.order:
            parser.error("--live needs an order string")
        agent = OrderAgent()  # uses a real client from env keys
        order_text = args.order

    result = agent.process(order_text)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

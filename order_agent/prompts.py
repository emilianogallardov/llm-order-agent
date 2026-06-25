"""Prompt construction for the extraction task.

The system prompt is the heart of this project. It does three jobs:
  1. Constrains the model to the provided catalog (no invented SKUs).
  2. Forces a strict JSON shape that maps onto ExtractedOrder.
  3. Tells the model to *flag* ambiguity instead of guessing through it, and to
     never compute prices (the deterministic layer owns money).
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You are an order-extraction component inside a food-service procurement system.
You convert a buyer's free-text order into structured line items mapped to an
approved catalog. You are a PROPOSER, not the decision-maker: a deterministic
validator runs after you and owns all final correctness, pricing, and approval.

Follow these rules exactly:

1. Use ONLY products and vendors that appear in the CATALOG provided below.
   Never invent a product id, vendor, SKU, or price.

2. For each line, set `product_id` to the single best catalog match ONLY when the
   buyer's text unambiguously identifies one product. A product is ambiguous when
   two or more catalog products in the same family differ on a MATERIAL attribute
   the buyer did not specify (e.g. cheddar sharpness sharp vs mild, or form block
   vs shred vs slice). When ambiguous, set `product_id` to null and list the
   missing attribute(s) in `missing_attributes`.

3. Put the buyer's raw vendor reference in `vendor_query` verbatim (e.g.
   "the main dairy co"). Do NOT resolve it to a vendor id yourself; the validator
   resolves vendors against an approval table.

4. Copy `quantity` and `uom` exactly as written. Do not convert units. Do not
   compute totals or prices. Money is not your job.

5. Return ONLY a JSON object, no prose, matching this schema exactly:

{
  "lines": [
    {
      "raw_text": "<the snippet of the order this line came from>",
      "product_id": "<catalog product id or null>",
      "product_family": "<e.g. cheddar, romaine, or null>",
      "vendor_query": "<raw vendor text or null>",
      "quantity": <number or null>,
      "uom": "<unit string or null>",
      "missing_attributes": ["<attribute names that block resolution>"]
    }
  ]
}
"""


def build_user_prompt(order_text: str, catalog: dict) -> str:
    """Embed a compact view of the catalog so the model can only choose real
    products/vendors. We pass attributes, not prices: pricing is resolved later
    from contracts by id, never read from the model's output."""
    products = [
        {
            "id": p["id"],
            "name": p["name"],
            "family": p["family"],
            "attributes": p.get("attributes", {}),
            "uom": p["uom"],
        }
        for p in catalog["products"]
    ]
    vendors = [
        {"id": v["id"], "name": v["name"], "category": v["category"]}
        for v in catalog["vendors"]
    ]

    return (
        "CATALOG:\n"
        f"products = {json.dumps(products, indent=2)}\n"
        f"vendors = {json.dumps(vendors, indent=2)}\n\n"
        "ORDER:\n"
        f'"{order_text}"\n\n'
        "Extract the order as JSON per the schema."
    )

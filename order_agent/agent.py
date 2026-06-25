"""Orchestration: free-text order -> structured payload.

Flow:
  1. selector picks the model for the `order_extraction` task
  2. the LLM proposes an ExtractedOrder (constrained to the catalog)
  3. deterministic validators resolve it into a ResolvedOrder (owns correctness)

The LLM client is injected, so the same pipeline runs live (real provider) or
offline/in tests (MockClient). The agent never trusts the model for money or
final approval; it only trusts it to read messy text and propose candidates.
"""

from __future__ import annotations

from typing import Optional

from .catalog import Catalog
from .llm import LLMClient, default_client
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import ExtractedLine, ExtractedOrder, ResolvedOrder
from .selector import get_execution_config
from .validators import resolve_order


class OrderAgent:
    def __init__(self, catalog: Optional[Catalog] = None, client: Optional[LLMClient] = None):
        self.catalog = catalog or Catalog.load()
        self.client = client or default_client()
        if self.client is None:
            raise RuntimeError(
                "No LLM client available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
                "or pass a MockClient for offline runs."
            )

    def process(self, order_text: str) -> ResolvedOrder:
        cfg = get_execution_config("order_extraction")
        user_prompt = build_user_prompt(order_text, self.catalog._data)  # noqa: SLF001

        raw = self.client.complete_json(cfg, SYSTEM_PROMPT, user_prompt)
        extracted = _parse_extracted(raw)

        return resolve_order(extracted, self.catalog)


def _parse_extracted(raw: dict) -> ExtractedOrder:
    """Defensive parse of the model's JSON into typed lines. Missing keys become
    None/empty so the deterministic layer can decide, rather than crashing."""
    lines = []
    for item in raw.get("lines", []):
        lines.append(
            ExtractedLine(
                raw_text=item.get("raw_text", ""),
                product_id=item.get("product_id"),
                product_family=item.get("product_family"),
                stated_attributes=item.get("stated_attributes") or {},
                vendor_query=item.get("vendor_query"),
                quantity=item.get("quantity"),
                uom=item.get("uom"),
            )
        )
    return ExtractedOrder(lines=lines)

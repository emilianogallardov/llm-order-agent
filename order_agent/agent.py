"""Orchestration: free-text order -> structured payload.

Flow:
  1. selector picks the model for the `order_extraction` task
  2. the LLM proposes an ExtractedOrder (constrained to the catalog)
  3. deterministic validators resolve it into a ResolvedOrder (owns correctness)

The LLM client is injected, so the same pipeline runs live (real provider) or
offline/in tests (MockClient). The agent never trusts the model for money or
final approval; it only trusts it to read messy text and propose candidates.

If the primary model's provider has no key, or the call/JSON fails, the agent
falls back to the task's fallback model. Malformed output never crashes: it
becomes a blocked payload.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .catalog import Catalog
from .llm import LLMClient, MockClient, default_client, provider_key_available
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import ExtractedLine, ExtractedOrder, OrderStatus, ResolvedOrder
from .selector import execution_config_for, get_execution_config
from .validators import resolve_order

_TASK = "order_extraction"


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
        user_prompt = build_user_prompt(order_text, self.catalog._data)  # noqa: SLF001

        raw, err = self._extract(user_prompt)
        if raw is None:
            return _blocked("model", f"extraction_failed:{err}")

        extracted, malformed = _parse_extracted(raw)
        if malformed:
            return _blocked("model", "malformed_model_output")

        return resolve_order(extracted, self.catalog, order_text)

    def _extract(self, user_prompt: str) -> Tuple[Optional[dict], Optional[str]]:
        """Call the model, falling back to the task's fallback model on a missing
        provider key or a failed call. Returns (json, None) or (None, reason)."""
        cfg = get_execution_config(_TASK)

        # MockClient (tests/offline) has no provider; call it once, directly.
        if isinstance(self.client, MockClient):
            try:
                return self.client.complete_json(cfg, SYSTEM_PROMPT, user_prompt), None
            except Exception as e:  # noqa: BLE001
                return None, type(e).__name__

        candidates = [cfg.model] + ([cfg.fallback] if cfg.fallback else [])
        last_err = "no_usable_model"
        for model_id in candidates:
            mcfg = execution_config_for(_TASK, model_id)
            if not provider_key_available(mcfg.model_config.provider):
                last_err = f"no_key_for_{mcfg.model_config.provider}"
                continue
            try:
                return self.client.complete_json(mcfg, SYSTEM_PROMPT, user_prompt), None
            except Exception as e:  # noqa: BLE001
                last_err = f"{model_id}:{type(e).__name__}"
        return None, last_err


def _parse_extracted(raw: object) -> Tuple[ExtractedOrder, bool]:
    """Defensive parse of the model's JSON. Returns (order, malformed). Malformed
    top-level shape (not a dict, or `lines` not a list) is flagged so the caller
    fails closed. A non-dict line item becomes an empty line that blocks itself."""
    if not isinstance(raw, dict):
        return ExtractedOrder(lines=[]), True
    items = raw.get("lines")
    if not isinstance(items, list):
        return ExtractedOrder(lines=[]), True

    lines = []
    for item in items:
        if not isinstance(item, dict):
            lines.append(ExtractedLine("", None, None, None, None, None, {}))
            continue
        sa = item.get("stated_attributes")
        if not isinstance(sa, dict):
            sa = {}
        lines.append(
            ExtractedLine(
                raw_text=item.get("raw_text", ""),
                product_id=item.get("product_id"),
                product_family=item.get("product_family"),
                vendor_query=item.get("vendor_query"),
                quantity=item.get("quantity"),
                uom=item.get("uom"),
                stated_attributes=sa,
            )
        )
    return ExtractedOrder(lines=lines), False


def _blocked(field: str, reason: str) -> ResolvedOrder:
    return ResolvedOrder(
        status=OrderStatus.VALIDATION_BLOCKED,
        lines=[],
        order_total=None,
        blocked_fields=[field],
        reasons=[reason],
    )

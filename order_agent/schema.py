"""Structured types that cross the LLM boundary and the final payload schema.

The LLM only ever produces `ExtractedOrder` (a *proposal*). The deterministic
layer turns that into a `ResolvedOrder` (the *decision*). Keeping these two types
separate is the whole point: the model proposes, validators own correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class OrderStatus(str, Enum):
    READY_FOR_STAGING = "ready_for_staging"
    CLARIFICATION_REQUIRED = "clarification_required"
    VALIDATION_BLOCKED = "validation_blocked"


# --- What the LLM proposes -----------------------------------------------------
@dataclass
class ExtractedLine:
    """One line the model pulled from the free-text order. These are *guesses*:
    a proposed product id, a proposed vendor reference, and the verbatim qty/uom.
    The model also flags when a material attribute (e.g. sharpness) is missing."""

    raw_text: str
    product_id: Optional[str]          # proposed catalog id, or None if unsure
    product_family: Optional[str]      # e.g. "cheddar", "romaine"
    vendor_query: Optional[str]        # raw vendor reference, e.g. "the main dairy co"
    quantity: Optional[float]
    uom: Optional[str]                 # verbatim unit: "lb", "case", ...
    missing_attributes: list[str] = field(default_factory=list)


@dataclass
class ExtractedOrder:
    lines: list[ExtractedLine]


# --- What the deterministic layer decides --------------------------------------
@dataclass
class ResolvedLine:
    raw_text: str
    product_id: str
    product_name: str
    vendor_id: str
    vendor_name: str
    parent_vendor_id: str
    quantity: Decimal
    uom: str
    contract_unit_price: Decimal
    contract_id: str
    line_total: Decimal


@dataclass
class ResolvedOrder:
    status: OrderStatus
    lines: list[ResolvedLine] = field(default_factory=list)
    order_total: Optional[Decimal] = None
    blocked_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    clarification: Optional[str] = None

    def to_dict(self) -> dict:
        """Stable serialization. The shape never changes across statuses, so a
        consumer (or a regression test) can rely on every key being present."""
        return {
            "status": self.status.value,
            "order_total": str(self.order_total) if self.order_total is not None else None,
            "lines": [
                {
                    "raw_text": ln.raw_text,
                    "product_id": ln.product_id,
                    "product_name": ln.product_name,
                    "vendor_id": ln.vendor_id,
                    "vendor_name": ln.vendor_name,
                    "parent_vendor_id": ln.parent_vendor_id,
                    "quantity": str(ln.quantity),
                    "uom": ln.uom,
                    "contract_unit_price": str(ln.contract_unit_price),
                    "contract_id": ln.contract_id,
                    "line_total": str(ln.line_total),
                }
                for ln in self.lines
            ],
            "blocked_fields": self.blocked_fields,
            "reasons": self.reasons,
            "clarification": self.clarification,
        }

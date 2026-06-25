"""Deterministic validation: the layer that owns correctness.

The model proposed an ExtractedOrder. Here we resolve it against the catalog with
plain, testable rules and produce a ResolvedOrder. Money is computed with Decimal
(never float). Anything we can't resolve with certainty fails closed: we block or
ask for clarification, we never fabricate a SKU, vendor, price, or total.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .catalog import Catalog
from .schema import (
    ExtractedLine,
    ExtractedOrder,
    OrderStatus,
    ResolvedLine,
    ResolvedOrder,
)

_CENTS = Decimal("0.01")


class _LineBlock(Exception):
    """Raised when a line can't be staged. `kind` is 'clarify' or 'block'."""

    def __init__(self, kind: str, field: str, reason: str):
        self.kind = kind
        self.field = field
        self.reason = reason


def resolve_order(extracted: ExtractedOrder, catalog: Catalog) -> ResolvedOrder:
    resolved_lines: list[ResolvedLine] = []
    blocked_fields: list[str] = []
    reasons: list[str] = []
    worst = OrderStatus.READY_FOR_STAGING

    for line in extracted.lines:
        try:
            resolved_lines.append(_resolve_line(line, catalog))
        except _LineBlock as b:
            blocked_fields.append(b.field)
            reasons.append(b.reason)
            if b.kind == "block":
                worst = OrderStatus.VALIDATION_BLOCKED
            elif worst != OrderStatus.VALIDATION_BLOCKED:
                worst = OrderStatus.CLARIFICATION_REQUIRED

    # Fail closed: if any line couldn't be staged, stage nothing.
    if worst != OrderStatus.READY_FOR_STAGING:
        return ResolvedOrder(
            status=worst,
            lines=[],
            order_total=None,
            blocked_fields=blocked_fields,
            reasons=reasons,
            clarification=_draft_clarification(worst, blocked_fields)
            if worst == OrderStatus.CLARIFICATION_REQUIRED
            else None,
        )

    order_total = sum((ln.line_total for ln in resolved_lines), Decimal("0")).quantize(
        _CENTS, rounding=ROUND_HALF_UP
    )
    return ResolvedOrder(
        status=OrderStatus.READY_FOR_STAGING,
        lines=resolved_lines,
        order_total=order_total,
    )


def _resolve_line(line: ExtractedLine, catalog: Catalog) -> ResolvedLine:
    # 1. Quantity must be present, numeric, strictly positive.
    if line.quantity is None:
        raise _LineBlock("block", "quantity", "missing_quantity")
    try:
        quantity = Decimal(str(line.quantity))
    except Exception:
        raise _LineBlock("block", "quantity", "non_numeric_quantity")
    if quantity <= 0:
        raise _LineBlock("block", "quantity", "non_positive_quantity")

    # 2. UOM must be present.
    if not line.uom:
        raise _LineBlock("block", "uom", "missing_uom")

    # 3. Product must resolve to exactly one catalog SKU. The model flags
    #    ambiguity; we also refuse to proceed if it left product_id null or named
    #    a family with multiple distinct products and missing attributes.
    if line.missing_attributes:
        raise _LineBlock(
            "clarify",
            f"{line.product_family or 'product'} variant",
            f"ambiguous_variant: missing {', '.join(line.missing_attributes)}",
        )
    if not line.product_id:
        family_size = len(catalog.products_in_family(line.product_family or ""))
        if family_size > 1:
            raise _LineBlock("clarify", f"{line.product_family} variant", "ambiguous_variant")
        raise _LineBlock("block", "product", "unresolved_product")

    product = catalog.product(line.product_id)
    if product is None:
        raise _LineBlock("block", "product", f"unknown_product_id:{line.product_id}")

    # 4. Vendor: resolve the raw reference through the approved-alias table by id.
    matches = catalog.resolve_vendor_alias(line.vendor_query or "")
    if len(matches) == 0:
        raise _LineBlock("clarify", "supplier entity", "unresolved_vendor_reference")
    if len(matches) > 1:
        raise _LineBlock("clarify", "supplier entity", "ambiguous_vendor_reference")
    vendor = catalog.vendor(matches[0]["vendor_id"])
    if vendor is None or not vendor.get("approved", False):
        raise _LineBlock("block", "supplier entity", "unapproved_vendor")

    # Cross-check: the resolved vendor must actually supply this product.
    if product.get("vendor_id") != vendor["id"]:
        raise _LineBlock("block", "supplier entity", "vendor_product_mismatch")

    # 5. Parent entity: contracts bind to the parent, which must resolve even if
    #    the child vendor was reorganized. A broken chain fails closed.
    parent_id = catalog.parent_vendor_id(vendor["id"])
    if parent_id is None:
        raise _LineBlock("block", "supplier entity", "supplier_parent_entity_unresolved")

    # 6. Contract pricing by canonical ids + exact UOM. No implicit conversion.
    contract = catalog.contract(product["id"], parent_id, line.uom)
    if contract is None:
        # Distinguish a UOM mismatch from a missing contract for a precise reason.
        any_uom = catalog.contract(product["id"], parent_id, _other_uom(catalog, product["id"], parent_id))
        if any_uom is not None:
            raise _LineBlock("block", "uom", "uom_mismatch")
        raise _LineBlock("block", "contract", "no_active_contract")

    # 7. Deterministic money. Decimal in, Decimal out, rounded once.
    unit_price: Decimal = contract["unit_price"]
    line_total = (quantity * unit_price).quantize(_CENTS, rounding=ROUND_HALF_UP)

    return ResolvedLine(
        raw_text=line.raw_text,
        product_id=product["id"],
        product_name=product["name"],
        vendor_id=vendor["id"],
        vendor_name=vendor["name"],
        parent_vendor_id=parent_id,
        quantity=quantity,
        uom=line.uom,
        contract_unit_price=unit_price,
        contract_id=contract["contract_id"],
        line_total=line_total,
    )


def _other_uom(catalog: Catalog, product_id: str, parent_id: str) -> str:
    """Return some UOM a contract exists under for this product+parent, so we can
    tell 'wrong unit' apart from 'no contract at all'."""
    for c in catalog._contracts:  # noqa: SLF001 - internal helper
        if c["product_id"] == product_id and c["parent_vendor_id"] == parent_id:
            return c["uom"]
    return ""


def _draft_clarification(status: OrderStatus, fields: list[str]) -> str:
    if not fields:
        return "Some line items need clarification before this order can be staged."
    unique = ", ".join(dict.fromkeys(fields))
    return f"Please clarify the following before I can stage this order: {unique}."

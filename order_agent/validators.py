"""Deterministic validation: the layer that owns correctness.

The model proposed an ExtractedOrder. Here we resolve it against the catalog with
plain, testable rules and produce a ResolvedOrder. Money is computed with Decimal
(never float). Two things we never trust:

  1. Model-supplied ids/prices: we re-derive the SKU, vendor, parent, and price
     from the catalog by canonical id.
  2. Model-supplied facts (quantity, vendor reference, attributes): we ground them
     against the original order text. A fact the text doesn't support is dropped
     or blocked, never used to stage an order.

Anything we can't resolve with certainty fails closed: we block or ask for
clarification, and we never fabricate a SKU, vendor, price, or total. Unexpected
errors are caught and turned into a block, so malformed model output can't crash
the pipeline or leak a half-built payload.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from .catalog import Catalog
from .grounding import in_text, line_span, quantity_uom_grounded
from .uom import canonicalize
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


def resolve_order(
    extracted: ExtractedOrder,
    catalog: Catalog,
    order_text: str,
    as_of: Optional[str] = None,
) -> ResolvedOrder:
    # Fail closed on an empty extraction: no lines is not a $0 order, it's a
    # failed parse. Never stage it.
    if not extracted.lines:
        return ResolvedOrder(
            status=OrderStatus.VALIDATION_BLOCKED,
            lines=[],
            order_total=None,
            blocked_fields=["order"],
            reasons=["empty_order"],
        )

    as_of = as_of or date.today().isoformat()

    resolved_lines: list[ResolvedLine] = []
    blocked_fields: list[str] = []
    reasons: list[str] = []
    worst = OrderStatus.READY_FOR_STAGING

    for line in extracted.lines:
        try:
            resolved_lines.append(_resolve_line(line, catalog, order_text, as_of))
        except _LineBlock as b:
            blocked_fields.append(b.field)
            reasons.append(b.reason)
            if b.kind == "block":
                worst = OrderStatus.VALIDATION_BLOCKED
            elif worst != OrderStatus.VALIDATION_BLOCKED:
                worst = OrderStatus.CLARIFICATION_REQUIRED
        except Exception:  # noqa: BLE001 - fail closed on any unexpected shape
            blocked_fields.append("line")
            reasons.append("validation_error")
            worst = OrderStatus.VALIDATION_BLOCKED

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


def _resolve_line(
    line: ExtractedLine, catalog: Catalog, order_text: str, as_of: str
) -> ResolvedLine:
    # Ground model-extracted facts against the order text. Quantity+unit are
    # grounded against the tight per-line span (the model's raw_text, if it's a
    # real substring) so one line can't borrow a number from another. Vendor and
    # attributes are grounded against the whole order, since a tight product span
    # often won't contain the vendor clause.
    span = line_span(line.raw_text, order_text)
    text_lower = order_text.lower()
    grounded = bool(order_text)

    # 1. Quantity: present, finite, positive.
    if line.quantity is None:
        raise _LineBlock("block", "quantity", "missing_quantity")
    try:
        quantity = Decimal(str(line.quantity))
    except Exception:
        raise _LineBlock("block", "quantity", "non_numeric_quantity")
    if not quantity.is_finite():
        raise _LineBlock("block", "quantity", "non_finite_quantity")
    if quantity <= 0:
        raise _LineBlock("block", "quantity", "non_positive_quantity")

    # 2. UOM present and a real unit, then canonicalized (lbs -> lb). This is
    #    canonicalization, not conversion: distinct physical units never merge.
    uom = canonicalize(line.uom)
    if not uom:
        raise _LineBlock("block", "uom", "missing_uom")

    # 3. Quantity and unit must be PAIRED in the text ("50 lbs", "a case").
    #    Grounding the pair catches inflation (5 can't match inside 50) and unit
    #    swaps (model says "lb" when the buyer said "cases").
    if grounded and not quantity_uom_grounded(quantity, uom, span):
        raise _LineBlock("block", "quantity", "quantity_uom_not_grounded")

    # 4. Resolve the product by ATTRIBUTES, grounded in the span. The model's
    #    product_id is only a hint; the catalog owns the decision.
    family = line.product_family
    if not family and line.product_id:
        hint = catalog.product(line.product_id)
        family = hint["family"] if hint else None
    if not family or not catalog.products_in_family(family):
        raise _LineBlock("block", "product", "unresolved_product")

    stated = line.stated_attributes if isinstance(line.stated_attributes, dict) else {}
    schema_keys = catalog.family_attribute_keys(family)
    if grounded:
        # A stated constraint the catalog can't express (e.g. "organic") that the
        # buyer really asked for must block, not be silently dropped.
        for key in stated:
            if key not in schema_keys and in_text(key, text_lower):
                raise _LineBlock("clarify", f"{family} attribute", f"unsupported_constraint:{key}")
        # Keep only catalog attributes whose value the text supports. A swapped
        # attribute (text "sharp", model "mild") is dropped, making the variant
        # ambiguous and forcing clarification instead of a wrong SKU.
        effective = {k: v for k, v in stated.items() if k in schema_keys and in_text(v, text_lower)}
    else:
        effective = {k: v for k, v in stated.items() if k in schema_keys}

    matches = catalog.match_products_by_attributes(family, effective)
    if len(matches) == 0:
        raise _LineBlock("clarify", f"{family} variant", "no_matching_variant")
    if len(matches) > 1:
        needed = catalog.distinguishing_attributes(matches)
        reason = f"ambiguous_variant: specify {', '.join(needed)}" if needed else "ambiguous_variant"
        raise _LineBlock("clarify", f"{family} variant", reason)
    product = matches[0]

    # 5. Vendor: the reference must be grounded in the span (catches a model that
    #    swaps an unapproved vendor for an approved one), then resolved by id.
    vendor_query = line.vendor_query or ""
    if grounded and vendor_query and not in_text(vendor_query, text_lower):
        raise _LineBlock("block", "supplier entity", "vendor_not_grounded")

    alias_matches = catalog.resolve_vendor_alias(vendor_query)
    if len(alias_matches) == 0:
        raise _LineBlock("clarify", "supplier entity", "unresolved_vendor_reference")
    if len(alias_matches) > 1:
        raise _LineBlock("clarify", "supplier entity", "ambiguous_vendor_reference")
    vendor = catalog.vendor(alias_matches[0]["vendor_id"])
    if vendor is None or not vendor.get("approved", False):
        raise _LineBlock("block", "supplier entity", "unapproved_vendor")

    # Cross-check: the resolved vendor must actually supply this product.
    if product.get("vendor_id") != vendor["id"]:
        raise _LineBlock("block", "supplier entity", "vendor_product_mismatch")

    # 6. Parent entity: contracts bind to the parent, which must resolve even if
    #    the child vendor was reorganized. A broken chain fails closed.
    parent_id = catalog.parent_vendor_id(vendor["id"])
    if parent_id is None:
        raise _LineBlock("block", "supplier entity", "supplier_parent_entity_unresolved")

    # 7. Contract pricing by canonical ids + exact UOM, effective on/before today.
    #    Duplicate active contracts at the same effective date are a genuine
    #    ambiguity: refuse to price rather than pick by list order.
    winners = catalog.active_contracts(product["id"], parent_id, uom, as_of=as_of)
    if len(winners) == 0:
        other = _other_uom(catalog, product["id"], parent_id)
        if other and catalog.active_contracts(product["id"], parent_id, other, as_of=as_of):
            raise _LineBlock("block", "uom", "uom_mismatch")
        raise _LineBlock("block", "contract", "no_active_contract")
    if len(winners) > 1:
        raise _LineBlock("block", "contract", "ambiguous_contract")
    contract = winners[0]

    # 8. Deterministic money. Decimal in, Decimal out, rounded once.
    unit_price = Decimal(contract["unit_price"])
    line_total = (quantity * unit_price).quantize(_CENTS, rounding=ROUND_HALF_UP)

    return ResolvedLine(
        raw_text=str(line.raw_text or ""),
        product_id=product["id"],
        product_name=product["name"],
        vendor_id=vendor["id"],
        vendor_name=vendor["name"],
        parent_vendor_id=parent_id,
        quantity=quantity,
        uom=uom,
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

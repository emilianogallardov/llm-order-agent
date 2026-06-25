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

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from .catalog import Catalog
from .grounding import attr_in_text, in_text, line_span, quantity_uom_grounded, quantity_uom_mentions
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

    # Coverage: every quantity+unit the buyer wrote must produce a line. If the
    # model extracted fewer lines than the order has quantity+unit mentions, it
    # silently dropped an item. Fail closed rather than stage a partial order.
    total_mentions = quantity_uom_mentions(order_text)
    if total_mentions and len(extracted.lines) < total_mentions:
        return ResolvedOrder(
            status=OrderStatus.VALIDATION_BLOCKED,
            lines=[],
            order_total=None,
            blocked_fields=["order"],
            reasons=["incomplete_extraction"],
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
    # Ground model-extracted facts against the per-line span: the model's raw_text
    # must be a verbatim slice of the order, and quantity, unit, attributes, and
    # product identity are all checked against THAT slice, so one line can't borrow
    # a fact from another. An unverifiable span fails closed. The vendor reference
    # is grounded against the whole order (the product span may omit the vendor
    # clause), with a negation guard.
    span = line_span(line.raw_text, order_text)
    if span is None:
        raise _LineBlock("block", "line", "line_span_unverifiable")
    # An honest line span mentions exactly one quantity+unit. A span covering two
    # line items lets one line's facts bleed into another, so reject it.
    if quantity_uom_mentions(span) > 1:
        raise _LineBlock("block", "line", "span_covers_multiple_items")

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

    # 3. Quantity and unit must be PAIRED in the span ("50 lbs", "a case").
    #    Grounding the pair catches inflation (5 can't match inside 50) and unit
    #    swaps (model says "lb" when the buyer said "cases").
    if not quantity_uom_grounded(quantity, uom, span):
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
    # A constraint the catalog can't express (e.g. "organic") that the buyer
    # really asked for must block, not be silently dropped. Catch it whether the
    # model put the word in the key or the value.
    for key, val in stated.items():
        if key not in schema_keys and (in_text(key, span) or attr_in_text(val, span)):
            raise _LineBlock("clarify", f"{family} attribute", f"unsupported_constraint:{key}")
    # Keep only catalog attributes whose value the span supports. A swapped
    # attribute (span "sharp", model "mild") is dropped, making the variant
    # ambiguous and forcing clarification instead of a wrong SKU.
    effective = {k: v for k, v in stated.items() if k in schema_keys and attr_in_text(v, span)}

    matches = catalog.match_products_by_attributes(family, effective)
    if len(matches) == 0:
        raise _LineBlock("clarify", f"{family} variant", "no_matching_variant")
    if len(matches) > 1:
        needed = catalog.distinguishing_attributes(matches)
        reason = f"ambiguous_variant: specify {', '.join(needed)}" if needed else "ambiguous_variant"
        raise _LineBlock("clarify", f"{family} variant", reason)
    product = matches[0]

    # 5. Product identity must be grounded by a NOUN, not by attribute overlap:
    #    the family name (inflected) or an explicit catalog keyword has to appear
    #    in the span. A matched adjective ("all-purpose", "whole milk", "ground")
    #    is not enough, so "all-purpose cleaner" can't pass as flour.
    if not _product_grounded(family, product, span):
        raise _LineBlock("clarify", f"{family} variant", "product_not_grounded")

    # 6. Vendor: the reference must appear in THIS line's span (catches a model
    #    that swaps in a vendor named elsewhere in the order, e.g. an account
    #    note) and must not be negated ("not premier"), then resolved by id.
    vendor_query = line.vendor_query or ""
    if vendor_query and not in_text(vendor_query, span):
        raise _LineBlock("block", "supplier entity", "vendor_not_grounded")
    if vendor_query and _is_negated(vendor_query, span):
        raise _LineBlock("block", "supplier entity", "vendor_negated")

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

    # 7. Parent entity: contracts bind to the parent, which must resolve even if
    #    the child vendor was reorganized. A broken chain fails closed.
    parent_id = catalog.parent_vendor_id(vendor["id"])
    if parent_id is None:
        raise _LineBlock("block", "supplier entity", "supplier_parent_entity_unresolved")

    # 8. Contract pricing by canonical ids + exact UOM, effective on/before today.
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

    # 9. Deterministic money. Decimal in, Decimal out, rounded once.
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


def _product_grounded(family: str, product: dict, span: str) -> bool:
    """The product must be named by a NOUN in the span: the family name (allowing
    inflection "tomato"->"tomatoes" and a single typo "chedder"->"cheddar") or an
    explicit catalog keyword ("evoo"). Matched attributes/adjectives do NOT count,
    so "all-purpose cleaner" or "whole milk" can't masquerade as flour/mozzarella
    (no word in those spans is within one edit of the family noun)."""
    if attr_in_text(family, span):
        return True
    fam = family.lower()
    if len(fam) >= 5:
        for word in re.findall(r"[a-z]+", span):
            if len(word) >= 5 and _within_one_edit(word, fam):
                return True
    return any(in_text(kw, span) for kw in product.get("keywords", []))


def _within_one_edit(a: str, b: str) -> bool:
    """True if a and b differ by at most one insertion, deletion, or substitution."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(c1 != c2 for c1, c2 in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            diff += 1
            if diff > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def _is_negated(vendor_query: str, text_lower: str) -> bool:
    """True if the vendor reference is negated in the text ('not premier', 'not
    from premier', 'do not use premier', 'avoid premier'). Allows a few filler
    words between the negation and the vendor. Substring grounding alone misses
    this; full natural-language negation is out of scope (see README)."""
    v = re.escape(vendor_query.strip().lower())
    neg = r"(?:do\s+not|don'?t|not|no|avoid|except|never|skip)"
    return bool(re.search(rf"\b{neg}\b[a-z\s,]{{0,15}}(?<![a-z]){v}(?![a-z])", text_lower))


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

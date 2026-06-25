"""Catalog access: products, vendors (with parent hierarchy), aliases, contracts.

Everything here is keyed by canonical id, never by display name. Display-name
matching is exactly the failure the validators are built to avoid: a vendor can
be renamed or reparented and the ids must still resolve.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "catalog.json"


class Catalog:
    def __init__(self, data: dict):
        self._data = data
        self._products = {p["id"]: p for p in data["products"]}
        self._vendors = {v["id"]: v for v in data["vendors"]}
        self._aliases = data.get("vendor_aliases", [])
        self._contracts = data.get("contracts", [])

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Catalog":
        path = path or _DEFAULT_PATH
        return cls(json.loads(Path(path).read_text()))

    # --- products -------------------------------------------------------------
    def product(self, product_id: str) -> Optional[dict]:
        return self._products.get(product_id)

    def products_in_family(self, family: str) -> list[dict]:
        return [p for p in self._products.values() if p["family"] == family]

    def family_attribute_keys(self, family: str) -> set[str]:
        """Every attribute key any product in the family carries. A stated key
        outside this set is a constraint the catalog can't express (e.g. organic)."""
        keys: set[str] = set()
        for p in self.products_in_family(family):
            keys.update(p.get("attributes", {}).keys())
        return keys

    def match_products_by_attributes(self, family: str, stated: dict | None) -> list[dict]:
        """Products in the family whose attributes are consistent with the ones
        stated in the order text. A stated attribute must equal the product's
        value for that key; keys the product doesn't carry are ignored. This is
        how the validator re-derives the SKU instead of trusting the model."""
        out = []
        for p in self.products_in_family(family):
            attrs = p.get("attributes", {})
            if all(
                str(attrs[k]).lower() == str(v).lower()
                for k, v in (stated or {}).items()
                if k in attrs
            ):
                out.append(p)
        return out

    @staticmethod
    def distinguishing_attributes(products: list[dict]) -> list[str]:
        """Attribute keys on which the candidate products disagree, i.e. the ones
        the buyer would need to specify to narrow it to one."""
        keys: set[str] = set()
        for p in products:
            keys.update(p.get("attributes", {}).keys())
        out = []
        for k in keys:
            vals = {
                str(p["attributes"][k]).lower()
                for p in products
                if k in p.get("attributes", {})
            }
            if len(vals) > 1:
                out.append(k)
        return sorted(out)

    # --- vendors --------------------------------------------------------------
    def vendor(self, vendor_id: str) -> Optional[dict]:
        return self._vendors.get(vendor_id)

    def resolve_vendor_alias(self, text: str) -> list[dict]:
        """Return the alias rows whose alias string matches the raw text. More
        than one match means the reference is ambiguous and must be clarified."""
        if not text:
            return []
        needle = text.strip().lower()
        return [a for a in self._aliases if a["alias"].strip().lower() == needle]

    def parent_vendor_id(self, vendor_id: str) -> Optional[str]:
        """Walk to the contract-bearing parent entity. Returns None if the chain
        is broken (a reparented vendor whose new parent isn't registered)."""
        v = self._vendors.get(vendor_id)
        if not v:
            return None
        parent_id = v.get("parent_id", vendor_id)
        # The parent must itself be a known entity, otherwise the hierarchy is
        # unresolved and we must fail closed rather than fabricate a binding.
        if parent_id != vendor_id and parent_id not in self._vendors and not _is_parent_token(parent_id, self._contracts):
            return None
        return parent_id

    # --- contracts ------------------------------------------------------------
    def contract(
        self,
        product_id: str,
        parent_vendor_id: str,
        uom: str,
        as_of: Optional[str] = None,
    ) -> Optional[dict]:
        """Contract pricing is bound to (product, PARENT vendor, uom). Binding to
        the parent is what survives a child vendor being reorganized.

        When `as_of` (an ISO date) is given, only contracts effective on or before
        that date are eligible, and the one with the latest effective date wins.
        That stops a future-dated or duplicate contract from silently pricing the
        order."""
        matches = [
            c
            for c in self._contracts
            if c["product_id"] == product_id
            and c["parent_vendor_id"] == parent_vendor_id
            and c["uom"] == uom
        ]
        if as_of is not None:
            matches = [c for c in matches if c.get("effective", "") <= as_of]
        if not matches:
            return None
        best = max(matches, key=lambda c: c.get("effective", ""))
        return {**best, "unit_price": Decimal(best["unit_price"])}


def _is_parent_token(parent_id: str, contracts: list[dict]) -> bool:
    """A parent id is valid if some contract is bound to it, even when no vendor
    row carries that id directly (the parent is a contracting entity)."""
    return any(c["parent_vendor_id"] == parent_id for c in contracts)

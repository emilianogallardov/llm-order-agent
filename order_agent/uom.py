"""Unit-of-measure canonicalization.

Humans write "lbs", "pounds", "cases"; the catalog stores one canonical token per
unit ("lb", "case"). This maps a free-text unit onto its canonical form via a
governed synonym table.

Important distinction: this is *canonicalization* (lbs -> lb), never *conversion*
(lb -> case). Synonyms of the same physical unit collapse to one token; genuinely
different units do not. So ordering "cases" of a per-lb item still fails to match
its per-lb contract and is correctly blocked. An unknown unit is returned
unchanged, so it won't accidentally match anything.
"""

from __future__ import annotations

# canonical token -> accepted synonyms (lower-cased, punctuation stripped)
_SYNONYMS: dict[str, set[str]] = {
    "lb": {"lb", "lbs", "lb.", "pound", "pounds", "#"},
    "oz": {"oz", "ozs", "ounce", "ounces"},
    "case": {"case", "cases", "cs", "cs.", "cse"},
    "bag": {"bag", "bags"},
    "each": {"each", "ea", "ea.", "unit", "units", "ct", "count", "piece", "pieces"},
    "gal": {"gal", "gals", "gallon", "gallons"},
    "qt": {"qt", "qts", "quart", "quarts"},
    "dozen": {"dozen", "dz", "doz"},
    "box": {"box", "boxes", "bx"},
    "can": {"can", "cans"},
}

_LOOKUP: dict[str, str] = {
    syn: canon for canon, syns in _SYNONYMS.items() for syn in syns
}


def canonicalize(raw: object) -> str | None:
    # Defensive: a non-string unit (model returned a number/list) is not a usable
    # unit. Return None so the validator blocks on a missing/invalid UOM rather
    # than crashing.
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().rstrip(".")
    return _LOOKUP.get(key, raw.strip().lower())

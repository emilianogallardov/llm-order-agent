"""Text grounding: check that the facts the model extracted actually appear in
the buyer's original order text (or, better, in the specific line span the model
claims a line came from).

This is the missing half of "model proposes, validators own correctness." The
validator already refuses model-supplied ids/prices, but the model still chooses
the quantity, unit, vendor reference, and attributes. A hallucinated or swapped
fact (50 -> 5000, "cases" -> "lb", "backalley" -> "premier", "sharp" -> "mild")
is catalog-valid but unfaithful to what the user said. Grounding catches that: a
fact the text doesn't support is dropped or blocked, never trusted.

Known limits (deterministic substring grounding can't close these alone):
  - Negation: "backalley, not premier" still contains the word "premier".
  - Omitted constraint: if the buyer says "organic" and the model drops the key,
    there's nothing in the extraction to check.
Closing those needs confidence-scored extraction with character offsets or a
human-in-the-loop gate; see README.
"""

from __future__ import annotations

import re

from .uom import synonyms_for

# spelled small numbers + articles, so "a case" / "two cases" ground to 1 / 2
_WORD_NUMBERS: dict[int, list[str]] = {
    1: ["a", "an", "one"], 2: ["two"], 3: ["three"], 4: ["four"], 5: ["five"],
    6: ["six"], 7: ["seven"], 8: ["eight"], 9: ["nine"], 10: ["ten"],
    11: ["eleven"], 12: ["twelve", "dozen"],
}


def in_text(value: object, text: str) -> bool:
    """True if the value's text appears (substring, case-insensitive) in text."""
    v = str(value).strip().lower()
    return bool(v) and v in text.lower()


def line_span(raw_text: object, order_text: str) -> str:
    """The slice of the order this line came from. If the model's raw_text is a
    real substring of the order, we ground against just that span (so a quantity
    from one line can't borrow words from another). Otherwise we fall back to the
    whole order, which is no weaker than grounding against everything."""
    order_lower = order_text.lower()
    if isinstance(raw_text, str) and raw_text.strip() and raw_text.lower() in order_lower:
        return raw_text.lower()
    return order_lower


def _quantity_forms(qty) -> list[str]:
    """Digit and (for small whole numbers) spelled forms of a quantity."""
    forms = [str(qty)]
    if qty == qty.to_integral_value():
        iv = int(qty)
        forms.append(str(iv))
        forms.extend(_WORD_NUMBERS.get(iv, []))
    return forms


def quantity_uom_grounded(qty, uom_canonical: str, text: str) -> bool:
    """True only if the order text pairs this quantity WITH this unit, e.g. the
    span contains "50 lbs" / "20 cases" / "a case". Grounding the pair at once
    catches both quantity inflation ("5" can't match inside "50") and unit swaps
    (model says "lb" but the text says "cases")."""
    text = text.lower()
    syns = synonyms_for(uom_canonical)
    for form in _quantity_forms(qty):
        # number part: digits guarded so "5" won't match inside "50"; words by \b
        if form.replace(".", "").isdigit():
            num = rf"(?<!\d){re.escape(form)}(?!\d)"
        else:
            num = rf"\b{re.escape(form)}\b"
        for syn in syns:
            unit = re.escape(syn) + (r"\b" if syn[-1:].isalnum() else "")
            if re.search(rf"{num}\s*{unit}", text):
                return True
    return False

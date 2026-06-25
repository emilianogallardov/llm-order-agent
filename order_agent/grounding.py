"""Text grounding: check that the facts the model extracted actually appear in
the buyer's original order text.

This is the missing half of "model proposes, validators own correctness." The
validator already refuses model-supplied ids/prices, but the model still chooses
the quantity, the vendor reference, and the attributes. A hallucinated or swapped
fact (50 -> 5000, "backalley" -> "premier", "sharp" -> "mild") is catalog-valid
but unfaithful to what the user said. Grounding catches that: a fact the text
does not support is dropped or blocked, never trusted.
"""

from __future__ import annotations

import re
from decimal import Decimal

# spelled small numbers + articles, so "a case" / "two cases" ground to 1 / 2
_WORD_NUMBERS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "dozen": 12,
}


def in_text(value: object, text_lower: str) -> bool:
    """True if the value's text appears in the (lower-cased) order text."""
    v = str(value).strip().lower()
    return bool(v) and v in text_lower


def quantity_grounded(qty: Decimal, text_lower: str) -> bool:
    """True if the quantity is supported by the order text: its digits appear, or
    (for small whole numbers) a spelled form or article does. Catches inflation
    like 50 -> 5000 while still accepting 'a case' and 'two cases'."""
    forms = {str(qty)}
    if qty == qty.to_integral_value():
        forms.add(str(int(qty)))
    if any(f in text_lower for f in forms):
        return True
    if qty == qty.to_integral_value():
        iv = int(qty)
        for word, n in _WORD_NUMBERS.items():
            if n == iv and re.search(rf"\b{re.escape(word)}\b", text_lower):
                return True
    return False

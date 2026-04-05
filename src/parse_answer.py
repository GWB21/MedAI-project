"""
Parse model output to extract A/B/C/D answer.
Multiple patterns are tried in order of specificity.
"""

import re
from typing import Optional


def parse_answer(raw_output: str) -> Optional[str]:
    """
    Extract choice A/B/C/D from model output text.

    Returns:
        "A", "B", "C", or "D", or None if parsing fails.
    """
    raw = raw_output.strip()
    if not raw:
        return None

    # Pattern 1: Single character
    if raw.upper() in ("A", "B", "C", "D"):
        return raw.upper()

    # Pattern 2: "The answer is X" / "The option is X"
    match = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-Da-d])", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Starts with "A." or "A)" or "A "
    match = re.match(r"^([A-Da-d])[.)\s]", raw)
    if match:
        return match.group(1).upper()

    # Pattern 4: First character is A/B/C/D
    if raw[0].upper() in ("A", "B", "C", "D"):
        return raw[0].upper()

    # Pattern 5: First occurrence of A/B/C/D anywhere
    for char in raw.upper():
        if char in ("A", "B", "C", "D"):
            return char

    return None

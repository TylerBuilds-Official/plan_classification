"""Label-aware sheet number extraction"""
import re

from .category_patterns import CATEGORY_PATTERNS


# Labels that precede a sheet number, ordered longest-first
SHEET_LABELS = [
    r'DRAWING\s*NUMBER',
    r'DRAWING\s*NO\.?:?',
    r'SHEET\s*NUMBER',
    r'SHEET\s*NO\.?:?',
    r'SHEET\s*#',
    r'SHT\s*NO\.?:?',
    r'SHT\s*#',
    r'DWG\s*NO\.?:?',
    r'DWG\s*#',
    r'NO\.?:',
]

# Any title block label — if we hit one while scanning, stop
_ANY_LABEL = re.compile(
    r'(?:SHEET|SHT|DWG|DRAWING|SCALE|REV|DATE|DRAWN|CHECKED|APPROVED|TITLE|SIZE|NO\.)',
    re.IGNORECASE,
)

# Compiled sheet number label pattern
_LABEL_PATTERN = re.compile(
    r'(?:' + '|'.join(SHEET_LABELS) + r')',
    re.IGNORECASE,
)


def extract_sheet_number(text: str ) -> tuple[str, str] | None:

    """
    Extract sheet number and discipline using label-aware parsing

    Finds "Sheet No." (or similar label), grabs the value immediately
    adjacent. No raw regex fallback — returns None if label-aware
    fails, letting the pipeline try OCR or AI instead.
    """

    if not text or not text.strip():

        return None

    return _extract_from_label(text)


def _extract_from_label(text: str ) -> tuple[str, str] | None:

    """
    Find a sheet number label, grab the immediately adjacent value

    Strict extraction: only checks same-line text and the first
    non-empty line after the label. If that line is another label,
    bail — the text ordering is garbled and we can't trust it.
    """

    match = _LABEL_PATTERN.search(text)
    if not match:

        return None

    after_label = text[match.end():]

    # Try same-line value first
    same_line = after_label.split('\n')[0]
    cleaned   = re.sub(r'^[\s:.\-#]+', '', same_line).strip()

    if cleaned:
        result = match_discipline(cleaned)
        if result:

            return result

    # Try the first non-empty line after label
    lines = after_label.split('\n')
    for line in lines[1:]:
        stripped = line.strip()

        if not stripped:
            continue

        # If we hit another label, the text order is garbled — bail
        if _ANY_LABEL.search(stripped):

            return None

        cleaned = re.sub(r'^[\s:.\-#]+', '', stripped).strip()
        if cleaned:

            return match_discipline(cleaned)

    return None


def match_discipline(value: str ) -> tuple[str, str] | None:

    """Match a sheet number string against category patterns

    Use for bare sheet numbers like 'G-101' where no label context exists.
    Returns (discipline, sheet_number) or None.
    """

    if not value or not value.strip():

        return None

    for discipline, pattern in CATEGORY_PATTERNS.items():
        match = re.search(pattern, value, re.IGNORECASE)
        if match:

            return discipline, match.group(0)

    return None

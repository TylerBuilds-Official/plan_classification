"""
Date regex patterns, parsing, and selection utilities.

Pure functions — no API dependencies, no I/O.
"""
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FoundDate:

    date_str:   str
    location:   str
    parsed:     datetime | None = None
    tier:       str             = ""
    page_index: int             = -1


# ── Regex patterns ────────────────────────────────────────────────
# Order matters — more specific patterns first to avoid partial matches.

DATE_PATTERNS: list[tuple[str, str]] = [
    # MM/DD/YYYY or MM-DD-YYYY or MM.DD.YYYY (4-digit year)
    (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', "numeric_long"),

    # MM/DD/YY or MM-DD-YY or MM.DD.YY (2-digit year)
    (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})(?!\d)', "numeric_short"),

    # YYYY-MM-DD (ISO format)
    (r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', "iso"),

    # Month DD, YYYY  (e.g. February 4, 2026 or Feb 4, 2026)
    (r'(January|February|March|April|May|June|July|August|September|October|November|December'
     r'|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})', "written_mdy"),

    # DD Month YYYY  (e.g. 4 February 2026)
    (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December'
     r'|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})', "written_dmy"),
]

# Context patterns that indicate a match is NOT a real date
FALSE_POSITIVE_CONTEXTS = [
    r'\d+[\'\"]\s*[-x×]\s*\d+',       # Dimensions  12' x 8'
    r'\d+\s*[-x×]\s*\d+\s*[-x×]',     # 3D dims
    r'scale\s*[:=]?\s*\d+',            # Scale refs
    r'#\d+',                           # Reference numbers
    r'spec\s',                         # Spec references
    r'detail\s',                       # Detail references
    r'aci\s+\d+',                      # ACI code refs  (ACI 318: 5.11-5.13)
    r'\d+-\d+/\d+\"',                  # Fractional dims  (5-3/16")
]


def extract_dates_from_text(
        text: str,
        source: str     = "native",
        page_index: int = -1 ) -> list[FoundDate]:

    """
    Run all date regex patterns against extracted text.

    Args:
        text:       Raw text from PDF page
        source:     Tier label (native / ocr)
        page_index: 0-based page number for tracking

    Returns:
        List of FoundDate with parsed datetimes
    """

    if not text or not text.strip():
        return []

    found: list[FoundDate] = []
    seen:  set[str]        = set()

    for pattern, fmt_name in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(0)

            if raw in seen:
                continue
            seen.add(raw)

            # Check surrounding context for false positives
            start = max(0, match.start() - 20)
            end   = min(len(text), match.end() + 20)
            ctx   = text[start:end].lower()

            if _is_false_positive(ctx):
                continue

            parsed = _parse_date_match(match, fmt_name)

            if parsed and _is_plausible_date(parsed):
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end   = text.find('\n', match.end())
                if line_end == -1:
                    line_end = min(len(text), match.end() + 60)

                location = text[line_start:line_end].strip()[:80]

                found.append(FoundDate(
                    date_str=raw,
                    location=location,
                    parsed=parsed,
                    tier=source,
                    page_index=page_index,
                ))

    return found


def select_best_date(
        all_dates: list[FoundDate] ) -> FoundDate | None:

    """
    Select the most recent date that isn't in the future.

    Falls back to the closest-to-now date if all are future.
    """

    now        = datetime.now()
    valid      = [d for d in all_dates if d.parsed is not None]
    non_future = [d for d in valid if d.parsed <= now]

    if non_future:
        return max(non_future, key=lambda d: d.parsed)

    if valid:
        return min(valid, key=lambda d: abs((d.parsed - now).days))

    return None


def format_mmddyy(
        dt: datetime ) -> str:

    """Format a datetime as MMDDYY string."""

    return dt.strftime("%m%d%y")


def try_parse_date(
        date_str: str ) -> datetime | None:

    """Try multiple formats to parse a date string."""

    if not date_str:
        return None

    formats = [
        "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
        "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


# ── Private helpers ───────────────────────────────────────────────

def _is_false_positive(
        context: str ) -> bool:

    """Check if surrounding context suggests this isn't a real date."""

    for fp_pattern in FALSE_POSITIVE_CONTEXTS:
        if re.search(fp_pattern, context, re.IGNORECASE):
            return True

    return False


def _is_plausible_date(
        dt: datetime ) -> bool:

    """
    Reject dates that are almost certainly not drawing dates.

    Keeps 1990+ and up to 1 year in the future.
    """

    now = datetime.now()

    if dt.year < 1990:
        return False

    if dt > now.replace(year=now.year + 1):
        return False

    return True


MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
    'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
    'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}


def _parse_date_match(
        match: re.Match,
        fmt_name: str ) -> datetime | None:

    """Parse a regex match into a datetime based on format type."""

    try:
        groups = match.groups()

        if fmt_name == "numeric_long":
            m, d, y = int(groups[0]), int(groups[1]), int(groups[2])
            return datetime(y, m, d)

        elif fmt_name == "numeric_short":
            m, d, y = int(groups[0]), int(groups[1]), int(groups[2])
            y += 2000 if y < 70 else 1900
            return datetime(y, m, d)

        elif fmt_name == "iso":
            y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
            return datetime(y, m, d)

        elif fmt_name == "written_mdy":
            month_str, d, y = groups[0], int(groups[1]), int(groups[2])
            m = MONTH_MAP.get(month_str.lower().rstrip('.'))
            if m:
                return datetime(y, m, d)

        elif fmt_name == "written_dmy":
            d, month_str, y = int(groups[0]), groups[1], int(groups[2])
            m = MONTH_MAP.get(month_str.lower().rstrip('.'))
            if m:
                return datetime(y, m, d)

    except (ValueError, TypeError):
        return None

    return None

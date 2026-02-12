"""
Tiered date extraction for construction drawings.

Tier 1: Native PDF text → regex  (free, milliseconds)
Tier 2: OCR text → regex         (free, seconds)
Tier 3: Opus vision + thinking   (expensive, fallback only)
"""
import json
import time
import base64
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime

import fitz
from PIL import Image

from ..utils.date.date_utils import (
    FoundDate,
    extract_dates_from_text,
    select_best_date,
    format_mmddyy,
    try_parse_date,
    _is_plausible_date,
)


OPUS = "claude-opus-4-5-20251101"

VISION_DATE_PROMPT = (
    "You are analyzing a full-page construction drawing to find EVERY date visible on the page.\n\n"
    "Construction drawings contain many dates in different locations:\n"
    "- Revision tables (revision number + date per entry)\n"
    "- Title block issue/drawing date\n"
    "- 'ISSUED FOR CONSTRUCTION' or 'RELEASED' stamps\n"
    "- Professional engineer seal dates\n"
    "- Permit stamps or approval dates\n"
    "- Copyright dates\n\n"
    "Your job is to find ALL of them. Examine the ENTIRE drawing carefully — "
    "title block, revision tables, stamps, seals, notes, everywhere.\n\n"
    "Return a JSON array of every date you find. For each date include:\n"
    "- \"date\": the date in MM/DD/YYYY format\n"
    "- \"location\": where on the drawing you found it (be specific)\n\n"
    "Example response:\n"
    "[\n"
    "  {\"date\": \"02/04/2026\", \"location\": \"Revision 3 in revision table\"},\n"
    "  {\"date\": \"09/30/2018\", \"location\": \"Original issue date in title block\"},\n"
    "  {\"date\": \"06/15/2020\", \"location\": \"Professional engineer seal\"}\n"
    "]\n\n"
    "Rules:\n"
    "- Return ONLY the JSON array, no markdown fences, no explanation\n"
    "- If you find NO dates at all, return exactly: []\n"
    "- Use MM/DD/YYYY format for all dates\n"
    "- Include every date you can find, even if partially legible\n"
    "- Be thorough — missing a date is worse than including an extra one"
)


@dataclass
class TierResult:

    tier:       str
    dates:      list[FoundDate] = field(default_factory=list)
    page_count: int             = 0
    elapsed_ms: float           = 0


@dataclass
class ExtractionResult:

    discipline:  str
    best_date:   FoundDate | None = None
    date_mmddyy: str | None      = None
    tiers:       list[TierResult] = field(default_factory=list)
    all_dates:   list[FoundDate]  = field(default_factory=list)


class DateExtractor:

    """
    Tiered date extraction: native text → OCR → Opus vision.

    Scans ALL pages per discipline for tiers 1-2 (free/fast).
    Only hits the API for tier 3 if earlier tiers find nothing.
    """

    def __init__(self,
            anthropic_client=None ) -> None:

        self.client = anthropic_client


    def extract_all(self,
            pdf_path: str,
            results: list,
            logger=None ) -> dict[str, str]:

        """
        Extract dates for every discipline found during classification.

        Args:
            pdf_path: Path to source PDF
            results:  list[PageResult] from ClassificationEngine
            logger:   Optional logger with .info() / .error()

        Returns:
            {discipline: "MMDDYY"} — same shape BreakoutHandler expects
        """

        discipline_pages = self._group_by_discipline(results)

        date_map:      dict[str, str | None] = {}
        fallback_date: str | None            = None

        for discipline, page_indices in discipline_pages.items():
            try:
                result = self.extract_for_discipline(
                    pdf_path, page_indices, discipline, logger=logger,
                )

                date_map[discipline] = result.date_mmddyy

                if result.date_mmddyy and not fallback_date:
                    fallback_date = result.date_mmddyy
                    if logger:
                        logger.info(
                            f'[DateExtractor] First date: {result.date_mmddyy} '
                            f'(from {discipline})'
                        )

            except Exception as e:
                if logger:
                    logger.error(f'[DateExtractor] Failed for {discipline}: {e}', exc=e)
                date_map[discipline] = None

        # Fill missing with first successful extraction or today
        final_fallback = fallback_date or datetime.now().strftime("%m%d%y")

        for discipline in date_map:
            if not date_map[discipline]:
                if logger:
                    logger.info(f'[DateExtractor] Filling {discipline} with fallback: {final_fallback}')
                date_map[discipline] = final_fallback

        return date_map


    def extract_for_discipline(self,
            pdf_path: str,
            page_indices: list[int],
            discipline: str,
            logger=None ) -> ExtractionResult:

        """
        Run tiered extraction across all pages in a discipline.

        Args:
            pdf_path:     Path to PDF
            page_indices: 0-based page indices for this discipline
            discipline:   Name for logging
            logger:       Optional logger with .info()

        Returns:
            ExtractionResult with best date and all tier details
        """

        result = ExtractionResult(discipline=discipline)

        def log(msg: str) -> None:
            if logger:
                logger.info(msg)

        doc = fitz.open(pdf_path)

        # ── Tier 1: Native text ───────────────────────────────────
        log(f'[Date:{discipline}] Tier 1 — Native text ({len(page_indices)} pages)')

        t1 = self._run_tier_native(doc, page_indices, log)
        result.tiers.append(t1)
        result.all_dates.extend(t1.dates)

        if t1.dates:
            best = select_best_date(t1.dates)
            if best:
                result.best_date   = best
                result.date_mmddyy = format_mmddyy(best.parsed)
                log(f'[Date:{discipline}] Tier 1 ✓ {len(t1.dates)} dates, selected: {result.date_mmddyy}')
                doc.close()
                return result

        log(f'[Date:{discipline}] Tier 1 ✗ No dates from native text')

        # ── Tier 2: OCR text ─────────────────────────────────────
        log(f'[Date:{discipline}] Tier 2 — OCR ({len(page_indices)} pages)')

        t2 = self._run_tier_ocr(doc, page_indices, log)
        result.tiers.append(t2)
        result.all_dates.extend(t2.dates)

        if t2.dates:
            best = select_best_date(t2.dates)
            if best:
                result.best_date   = best
                result.date_mmddyy = format_mmddyy(best.parsed)
                log(f'[Date:{discipline}] Tier 2 ✓ {len(t2.dates)} dates, selected: {result.date_mmddyy}')
                doc.close()
                return result

        log(f'[Date:{discipline}] Tier 2 ✗ No dates from OCR')

        # ── Tier 3: Opus vision ──────────────────────────────────
        if self.client:
            sample = self._pick_sample_pages(page_indices)
            log(f'[Date:{discipline}] Tier 3 — Opus vision ({len(sample)} sample pages)')

            t3 = self._run_tier_vision(doc, sample, discipline, log)
            result.tiers.append(t3)
            result.all_dates.extend(t3.dates)

            if t3.dates:
                best = select_best_date(t3.dates)
                if best:
                    result.best_date   = best
                    result.date_mmddyy = format_mmddyy(best.parsed)
                    log(f'[Date:{discipline}] Tier 3 ✓ {len(t3.dates)} dates, selected: {result.date_mmddyy}')
                    doc.close()
                    return result

            log(f'[Date:{discipline}] Tier 3 ✗ No dates from vision')
        else:
            log(f'[Date:{discipline}] Tier 3 — SKIPPED (no Anthropic client)')

        doc.close()

        # Final fallback: select from combined pool
        if result.all_dates:
            best = select_best_date(result.all_dates)
            if best:
                result.best_date   = best
                result.date_mmddyy = format_mmddyy(best.parsed)

        return result


    # ── Tier implementations ──────────────────────────────────────

    @staticmethod
    def _run_tier_native(
            doc: fitz.Document,
            page_indices: list[int],
            log ) -> TierResult:

        """Extract dates from native PDF text across all pages."""

        start = time.perf_counter()
        tier  = TierResult(tier="native", page_count=len(page_indices))

        for idx in page_indices:
            page  = doc.load_page(idx)
            text  = page.get_text("text")
            dates = extract_dates_from_text(text, source="native", page_index=idx)
            tier.dates.extend(dates)

        tier.elapsed_ms = (time.perf_counter() - start) * 1000
        log(f'    {len(tier.dates)} dates in {tier.elapsed_ms:.0f}ms')

        return tier


    @staticmethod
    def _run_tier_ocr(
            doc: fitz.Document,
            page_indices: list[int],
            log ) -> TierResult:

        """OCR each page and extract dates from the resulting text."""

        start = time.perf_counter()
        tier  = TierResult(tier="ocr", page_count=len(page_indices))

        for idx in page_indices:
            page = doc.load_page(idx)

            try:
                tp   = page.get_textpage_ocr(flags=fitz.TEXT_PRESERVE_WHITESPACE)
                text = page.get_text("text", textpage=tp)
            except Exception:
                text = page.get_text("text")

            dates = extract_dates_from_text(text, source="ocr", page_index=idx)
            tier.dates.extend(dates)

        tier.elapsed_ms = (time.perf_counter() - start) * 1000
        log(f'    {len(tier.dates)} dates in {tier.elapsed_ms:.0f}ms')

        return tier


    def _run_tier_vision(self,
            doc: fitz.Document,
            page_indices: list[int],
            discipline: str,
            log ) -> TierResult:

        """Send page images to Opus with extended thinking."""

        start = time.perf_counter()
        tier  = TierResult(tier="vision", page_count=len(page_indices))

        for idx in page_indices:
            log(f'    Sending page {idx + 1} to Opus...')
            page      = doc.load_page(idx)
            img_bytes = self._render_page(page)
            b64       = base64.b64encode(img_bytes).decode('utf-8')

            try:
                response = self.client.messages.create(
                    model=OPUS,
                    max_tokens=16000,
                    thinking={
                        "type":         "enabled",
                        "budget_tokens": 10000,
                    },
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": "image/jpeg",
                                    "data":       b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": VISION_DATE_PROMPT,
                            },
                        ],
                    }],
                )

                raw = ""
                for block in response.content:
                    if block.type == "text":
                        raw = block.text
                        break

                dates = self._parse_vision_response(raw, page_index=idx)
                tier.dates.extend(dates)
                log(f'    Page {idx + 1}: {len(dates)} dates')

            except Exception as e:
                log(f'    Page {idx + 1}: FAILED — {e}')

        tier.elapsed_ms = (time.perf_counter() - start) * 1000
        log(f'    {len(tier.dates)} total dates in {tier.elapsed_ms:.0f}ms')

        return tier


    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _group_by_discipline(
            results: list ) -> dict[str, list[int]]:

        """
        Group PageResult objects into {discipline: [page_indices]}.

        Accepts both PageResult objects and dicts.
        """

        discipline_pages: dict[str, list[int]] = {}

        for r in results:
            if hasattr(r, 'discipline'):
                disc = r.discipline
                idx  = r.page_index
            else:
                disc = r.get('discipline') or r.get('category')
                idx  = r.get('page_index', r.get('page_num', 0))

            if disc:
                discipline_pages.setdefault(disc, []).append(idx)

        return discipline_pages


    @staticmethod
    def _pick_sample_pages(
            page_indices: list[int] ) -> list[int]:

        """Pick first, middle, and last pages for vision sampling."""

        if len(page_indices) <= 3:
            return page_indices

        first  = page_indices[0]
        middle = page_indices[len(page_indices) // 2]
        last   = page_indices[-1]

        return list(dict.fromkeys([first, middle, last]))


    @staticmethod
    def _render_page(
            page: fitz.Page,
            zoom: float        = 3.0,
            max_dimension: int = 4096,
            quality: int       = 95 ) -> bytes:

        """Render a full page as high-quality JPEG."""

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.open(BytesIO(pix.tobytes('png')))

        if max(img.size) > max_dimension:
            ratio    = max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img      = img.resize(new_size, Image.Resampling.LANCZOS)

        output = BytesIO()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output, format='JPEG', quality=quality, optimize=True)

        return output.getvalue()


    @staticmethod
    def _parse_vision_response(
            raw: str,
            page_index: int = -1 ) -> list[FoundDate]:

        """Parse Opus JSON array response into FoundDate objects."""

        cleaned = raw.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]

        cleaned = cleaned.strip()

        if not cleaned or cleaned == "[]":
            return []

        try:
            entries = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(entries, list):
            return []

        found: list[FoundDate] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            date_str = entry.get("date", "")
            location = entry.get("location", "unknown")
            parsed   = try_parse_date(date_str)

            if parsed and _is_plausible_date(parsed):
                found.append(FoundDate(
                    date_str=date_str,
                    location=location,
                    parsed=parsed,
                    tier="vision",
                    page_index=page_index,
                ))

        return found

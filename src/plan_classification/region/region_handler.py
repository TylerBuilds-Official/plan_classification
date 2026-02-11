"""
Region Handler - Automatic Sheet Number Location Detection
AI-assisted detection with PyMuPDF precision and vision-based OCR fallback
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import fitz

from ..utils.pdf.pdf_utils import extract_text_from_region, extract_image_from_region, optimize_image_for_api
from ..constants import CATEGORY_PATTERNS, extract_sheet_number
from ..utils.ai.ocr_service import OCRService
from ..pipeline_config import PipelineConfig
from ._dataclass.region_result import RegionResult
from ._dataclass.validation_result import ValidationResult
from ._errors.region_detection_error import RegionDetectionError


# Predefined candidate regions grouped by area
# Multiple variants per area to account for different title block sizes
CANDIDATE_REGIONS = {
    "bottom-right": [
        {"x_ratio": 0.82, "y_ratio": 0.90, "w_ratio": 0.17, "h_ratio": 0.09, "name": "br-wide"},
        {"x_ratio": 0.85, "y_ratio": 0.92, "w_ratio": 0.14, "h_ratio": 0.07, "name": "br-standard"},
        {"x_ratio": 0.90, "y_ratio": 0.94, "w_ratio": 0.09, "h_ratio": 0.05, "name": "br-tight"},
        {"x_ratio": 0.70, "y_ratio": 0.90, "w_ratio": 0.29, "h_ratio": 0.09, "name": "br-extended"},
    ],
    "bottom-left": [
        {"x_ratio": 0.01, "y_ratio": 0.90, "w_ratio": 0.17, "h_ratio": 0.09, "name": "bl-wide"},
        {"x_ratio": 0.01, "y_ratio": 0.92, "w_ratio": 0.14, "h_ratio": 0.07, "name": "bl-standard"},
        {"x_ratio": 0.01, "y_ratio": 0.94, "w_ratio": 0.09, "h_ratio": 0.05, "name": "bl-tight"},
    ],
    "top-right": [
        {"x_ratio": 0.82, "y_ratio": 0.01, "w_ratio": 0.17, "h_ratio": 0.09, "name": "tr-wide"},
        {"x_ratio": 0.85, "y_ratio": 0.01, "w_ratio": 0.14, "h_ratio": 0.07, "name": "tr-standard"},
        {"x_ratio": 0.90, "y_ratio": 0.01, "w_ratio": 0.09, "h_ratio": 0.05, "name": "tr-tight"},
    ],
    "top-left": [
        {"x_ratio": 0.01, "y_ratio": 0.01, "w_ratio": 0.17, "h_ratio": 0.09, "name": "tl-wide"},
        {"x_ratio": 0.01, "y_ratio": 0.01, "w_ratio": 0.14, "h_ratio": 0.07, "name": "tl-standard"},
    ],
    "right-edge": [
        {"x_ratio": 0.90, "y_ratio": 0.35, "w_ratio": 0.09, "h_ratio": 0.25, "name": "re-mid"},
        {"x_ratio": 0.85, "y_ratio": 0.30, "w_ratio": 0.14, "h_ratio": 0.35, "name": "re-wide"},
    ],
    "bottom-center": [
        {"x_ratio": 0.35, "y_ratio": 0.90, "w_ratio": 0.30, "h_ratio": 0.09, "name": "bc-wide"},
        {"x_ratio": 0.40, "y_ratio": 0.92, "w_ratio": 0.20, "h_ratio": 0.07, "name": "bc-standard"},
    ],
}

# Area keyword mapping — maps AI descriptions to candidate groups
AREA_KEYWORDS = {
    "bottom-right":  ["bottom-right", "bottom right", "lower-right", "lower right", "br"],
    "bottom-left":   ["bottom-left", "bottom left", "lower-left", "lower left", "bl"],
    "top-right":     ["top-right", "top right", "upper-right", "upper right", "tr"],
    "top-left":      ["top-left", "top left", "upper-left", "upper left", "tl"],
    "right-edge":    ["right edge", "right side", "right margin", "vertical right"],
    "bottom-center": ["bottom center", "bottom middle", "lower center"],
}


class RegionHandler:
    """
    AI-assisted sheet number region detection

    Phase 1 — Region Detection (once per set):
        Path A (native text): Opus reads page → PyMuPDF locates exact text block → padded region
        Path B (image-based):  Opus reads page + area → candidate regions → Sonnet OCR → first hit

    Phase 2 — Per-page classification uses the locked region
    """

    def __init__(self, config: PipelineConfig):
        self.config     = config
        self.ocr        = OCRService(api_key=config.anthropic_api_key, model=config.ocr_model)
        self.total_cost = 0.0
        self.timings: dict[str, float] = {}


    def auto_detect_region(self, pdf_path: str, logger=None ) -> RegionResult:

        """Detect the sheet number region for a PDF set"""

        doc        = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()

        # Pick sample pages — middle first, then spread out
        sample_indices = self._pick_sample_pages(pdf_path, count=self.config.sample_count)

        if logger:
            logger.info(f"Sample pages selected: {[i + 1 for i in sample_indices]}")

        # Try each sample page until AI reads a sheet number
        ai_reading = None
        t0 = time.perf_counter()
        for page_idx in sample_indices:
            ai_reading = self._ai_read_page(pdf_path, page_idx, logger=logger)
            if ai_reading and ai_reading.get("sheet_number"):
                break
        self.timings["ai_read"] = time.perf_counter() - t0

        if not ai_reading or not ai_reading.get("sheet_number"):
            raise RegionDetectionError(
                "AI could not identify any sheet numbers on sample pages. "
                "This drawing set may have non-standard formatting."
            )

        sheet_number = ai_reading["sheet_number"]
        area         = ai_reading.get("area", "bottom-right")

        if logger:
            logger.info(f"AI detected: sheet_number={sheet_number!r}, area={area!r}")

        # Path A: Try native text block search
        t0 = time.perf_counter()
        region = self._find_native_text_block(pdf_path, sheet_number, logger=logger)
        self.timings["native_search"] = time.perf_counter() - t0

        if region:
            padded = self._pad_region(region)
            t0     = time.perf_counter()
            validation = self._validate_region(pdf_path, padded, logger=logger)
            self.timings["native_validation"] = time.perf_counter() - t0

            if logger:
                logger.info(f"Path A (native text) — validation: {validation.match_rate:.0%}")

            if validation.match_rate >= self.config.min_validation_rate:

                return RegionResult(
                    region=padded,
                    confidence=min(0.98, validation.match_rate + 0.05),
                    method='native_text',
                    cost_usd=self.total_cost,
                    detected_samples=validation.extracted_numbers,
                    validation_score=validation.match_rate
                )

        # Path B: Image-based — OCR candidate regions
        if logger:
            logger.info(f"Path B (OCR) — scanning candidates for area: {area!r}")

        candidates       = self._get_candidates_for_area(area)
        ocr_scan_total   = 0.0
        ocr_val_total    = 0.0

        for candidate in candidates:
            t0         = time.perf_counter()
            ocr_result = self._ocr_candidate_region(pdf_path, candidate, sample_indices[0], logger=logger)
            ocr_scan_total += time.perf_counter() - t0

            if ocr_result:
                padded = self._pad_region(candidate)
                t0     = time.perf_counter()
                validation = self._validate_region_ocr(pdf_path, padded, logger=logger)
                ocr_val_total += time.perf_counter() - t0

                if logger:
                    logger.info(f"  Hit on {candidate.get('name', '?')} — validation: {validation.match_rate:.0%}")

                if validation.match_rate >= self.config.min_validation_rate:
                    self.timings["ocr_scan"]       = ocr_scan_total
                    self.timings["ocr_validation"] = ocr_val_total

                    return RegionResult(
                        region=padded,
                        confidence=min(0.95, validation.match_rate + 0.05),
                        method='ocr_candidate',
                        cost_usd=self.total_cost,
                        detected_samples=validation.extracted_numbers,
                        validation_score=validation.match_rate
                    )

        self.timings["ocr_scan"]       = ocr_scan_total
        self.timings["ocr_validation"] = ocr_val_total

        # All paths exhausted
        raise RegionDetectionError(
            "Could not lock a sheet number region via native text or OCR. "
            "This drawing set may have non-standard formatting."
        )


    # ── AI Page Reading (Opus) ──────────────────────────────────────────────

    def _ai_read_page(self, pdf_path: str, page_idx: int, logger=None ) -> dict | None:

        """Ask Opus to read a full page and identify sheet number + area"""

        import base64
        from anthropic import Anthropic

        doc  = fitz.open(pdf_path)
        page = doc.load_page(page_idx)

        # Render at decent resolution
        mat       = fitz.Matrix(2.0, 2.0)
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes(output='png')
        img_opt   = optimize_image_for_api(img_bytes, max_dimension=self.config.max_image_dimension)
        doc.close()

        b64    = base64.b64encode(img_opt).decode('utf-8')
        client = Anthropic(api_key=self.config.anthropic_api_key)

        if logger:
            logger.info(f"  Sending page {page_idx + 1} to Opus for reading...")

        response = client.messages.create(
            model=self.config.vision_model,
            max_tokens=512,
            system=(
                "You are an expert at reading construction and architectural drawings. "
                "Your job is to find the SHEET NUMBER on the page. "
                "Sheet numbers use discipline prefixes like A-101, S-201, E-301, M-401, "
                "P-501, C-101, G-001, L-101, FP-101, FS-101, LS-101, SS-101, T-101. "
                "Separators may be hyphens, dots, or spaces (e.g. A-101, A.101, A 101, A1.01).\n\n"
                "Return ONLY valid JSON, no markdown:\n"
                '{"sheet_number": "exactly as printed", "area": "bottom-right"}\n\n'
                "For area, use one of: bottom-right, bottom-left, top-right, top-left, "
                "right-edge, bottom-center.\n\n"
                "If you cannot find a sheet number, return:\n"
                '{"sheet_number": null, "area": null}'
            ),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Find the sheet number on this construction drawing and tell me where on the page it's located."
                    }
                ]
            }]
        )

        # Cost tracking — Opus 4.5: $15/M input, $75/M output
        cost = (response.usage.input_tokens / 1000) * 0.015 + \
               (response.usage.output_tokens / 1000) * 0.075
        self.total_cost += cost

        # Parse response
        text = response.content[0].text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            result = json.loads(text.strip())

            if logger:
                logger.info(f"  Opus response: {result}")

            return result

        except (json.JSONDecodeError, KeyError) as e:
            if logger:
                logger.warning(f"  Failed to parse Opus response: {e} — raw: {text!r}")

            return None


    # ── Native Text Block Search (PyMuPDF) ──────────────────────────────────

    def _find_native_text_block(self,
            pdf_path: str,
            sheet_number: str,
            logger=None ) -> dict | None:

        """Search PyMuPDF text blocks for the exact sheet number string"""

        doc            = fitz.open(pdf_path)
        sample_indices = self._pick_sample_pages(pdf_path, count=self.config.validation_samples)
        found          = []

        # Clean the sheet number for flexible matching
        clean_target = re.sub(r'[\s\-.]', '', sheet_number).upper()

        for page_idx in sample_indices:
            page   = doc.load_page(page_idx)
            page_w = page.rect.width
            page_h = page.rect.height
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text  = span.get("text", "").strip()
                        clean_span = re.sub(r'[\s\-.]', '', span_text).upper()

                        if clean_target in clean_span:
                            bbox   = span["bbox"]
                            region = {
                                "x_ratio": bbox[0] / page_w,
                                "y_ratio": bbox[1] / page_h,
                                "w_ratio": (bbox[2] - bbox[0]) / page_w,
                                "h_ratio": (bbox[3] - bbox[1]) / page_h,
                            }
                            found.append(region)

                            if logger:
                                logger.debug(
                                    f"  Text block hit: page {page_idx + 1}, "
                                    f"text={span_text!r}, region={region}"
                                )

        doc.close()

        if not found:
            if logger:
                logger.info(f"  No native text blocks matched {sheet_number!r}")

            return None

        # Average the found positions for consistency
        avg_region = {
            "x_ratio": sum(r["x_ratio"] for r in found) / len(found),
            "y_ratio": sum(r["y_ratio"] for r in found) / len(found),
            "w_ratio": sum(r["w_ratio"] for r in found) / len(found),
            "h_ratio": sum(r["h_ratio"] for r in found) / len(found),
        }

        if logger:
            logger.info(f"  Averaged region from {len(found)} hits: {avg_region}")

        return avg_region


    # ── OCR Candidate Scanning ──────────────────────────────────────────────

    def _ocr_candidate_region(self,
            pdf_path: str,
            region: dict,
            page_idx: int,
            logger=None ) -> str | None:

        """Render a candidate region and OCR it, return text if sheet number found"""

        doc       = fitz.open(pdf_path)
        page      = doc.load_page(page_idx)
        img_bytes = extract_image_from_region(page, region, zoom=self.config.ocr_zoom, format='PNG')
        doc.close()

        ocr_text = self.ocr.extract_text(img_bytes, media_type="image/png")

        if logger:
            logger.debug(f"    OCR [{region.get('name', '?')}]: {ocr_text!r}")

        if extract_sheet_number(ocr_text):

            return ocr_text

        return None


    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_region(self, pdf_path: str, region: dict,
            sample_size: int | None = None,
            logger=None ) -> ValidationResult:

        """Validate region using native text extraction"""

        sample_size    = sample_size or self.config.validation_samples
        doc            = fitz.open(pdf_path)
        sample_indices = self._pick_sample_pages(pdf_path, count=sample_size)

        matched           = 0
        extracted_numbers = []
        failed_pages      = []

        for page_idx in sample_indices:
            page = doc.load_page(page_idx)
            text = extract_text_from_region(page, region)

            result = extract_sheet_number(text)
            if result:
                matched += 1
                extracted_numbers.append(result[1])
            else:
                failed_pages.append(page_idx)

        doc.close()

        match_rate = matched / len(sample_indices) if sample_indices else 0.0

        if logger:
            logger.debug(f"  Native validation: {matched}/{len(sample_indices)} pages matched")

        return ValidationResult(
            success=match_rate >= self.config.min_validation_rate,
            match_rate=match_rate,
            matched_pages=matched,
            total_pages=len(sample_indices),
            extracted_numbers=extracted_numbers,
            failed_pages=failed_pages
        )


    def _validate_region_ocr(self, pdf_path: str, region: dict,
            sample_size: int | None = None,
            logger=None ) -> ValidationResult:

        """Validate region using Sonnet OCR in parallel"""

        sample_size    = sample_size or self.config.validation_samples
        sample_indices = self._pick_sample_pages(pdf_path, count=sample_size)

        # Pre-render all region images (fast, CPU-bound)
        images = {}
        doc    = fitz.open(pdf_path)
        for page_idx in sample_indices:
            page              = doc.load_page(page_idx)
            images[page_idx]  = extract_image_from_region(page, region, zoom=self.config.ocr_zoom, format='PNG')
        doc.close()

        # OCR in parallel (IO-bound API calls)
        results = {}

        def _ocr_page(page_idx: int ) -> tuple[int, str]:
            ocr_text = self.ocr.extract_text(images[page_idx], media_type="image/png")

            return page_idx, ocr_text

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(_ocr_page, idx): idx for idx in sample_indices}
            for future in as_completed(futures):
                page_idx, ocr_text = future.result()
                results[page_idx]  = ocr_text

        # Tally
        matched           = 0
        extracted_numbers = []
        failed_pages      = []

        for page_idx in sample_indices:
            result = extract_sheet_number(results[page_idx])
            if result:
                matched += 1
                extracted_numbers.append(result[1])
            else:
                failed_pages.append(page_idx)

        match_rate = matched / len(sample_indices) if sample_indices else 0.0

        if logger:
            logger.debug(f"  OCR validation: {matched}/{len(sample_indices)} pages matched")

        return ValidationResult(
            success=match_rate >= self.config.min_validation_rate,
            match_rate=match_rate,
            matched_pages=matched,
            total_pages=len(sample_indices),
            extracted_numbers=extracted_numbers,
            failed_pages=failed_pages
        )


    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_sample_pages(pdf_path: str, count: int = 3 ) -> list[int]:

        """Pick sample pages from middle of the set, skipping likely covers"""

        doc        = fitz.open(pdf_path)
        page_count = doc.page_count

        if page_count <= count:
            doc.close()

            return list(range(page_count))

        # Skip page 0 if it looks like a cover
        start = 0
        if page_count > 3:
            first_text  = doc.load_page(0).get_text("text").lower()
            cover_words = ['cover', 'title sheet', 'index', 'table of contents',
                           'drawing list', 'sheet index', 'project directory']

            if any(w in first_text for w in cover_words) or len(first_text.strip()) < 100:
                start = 1

        doc.close()

        # Distribute samples across the available range
        available = page_count - start
        if available <= count:

            return list(range(start, page_count))

        step    = available / (count + 1)
        indices = [start + int(step * (i + 1)) for i in range(count)]

        return indices


    @staticmethod
    def _get_candidates_for_area(area: str ) -> list[dict]:

        """Map an AI-reported area string to candidate regions"""

        area_lower = area.lower().strip()

        # Direct match
        if area_lower in CANDIDATE_REGIONS:

            return CANDIDATE_REGIONS[area_lower]

        # Keyword fuzzy match
        for area_key, keywords in AREA_KEYWORDS.items():
            if any(kw in area_lower for kw in keywords):

                return CANDIDATE_REGIONS[area_key]

        # Default: try all bottom-right candidates (most common)
        return CANDIDATE_REGIONS["bottom-right"]


    def _pad_region(self, region: dict ) -> dict:

        """Add generous padding to a region, clamped to page bounds"""

        p = self.config.region_padding

        return {
            "x_ratio": max(0.0, region["x_ratio"] - p),
            "y_ratio": max(0.0, region["y_ratio"] - p),
            "w_ratio": min(1.0, region["w_ratio"] + p * 2),
            "h_ratio": min(1.0, region["h_ratio"] + p * 2),
        }


    def get_total_cost(self) -> float:

        """Get total API cost for this session"""

        return self.total_cost

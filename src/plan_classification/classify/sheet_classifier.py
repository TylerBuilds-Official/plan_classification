"""
Sheet Classifier — Parallel per-page classification using a locked region
4-step chain: native text → OCR → full page AI → unclassified
"""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
from anthropic import Anthropic

from ..constants import CATEGORY_PATTERNS, extract_sheet_number, match_discipline
from ..pipeline_config import PipelineConfig
from ..utils.pdf.pdf_utils import extract_text_from_region, extract_image_from_region, optimize_image_for_api
from ..utils.ai.ocr_service import OCRService
from ..region._dataclass.region_result import RegionResult
from ._dataclass.page_result import PageResult


# Discipline list for AI prompt
DISCIPLINE_LIST = "\n".join(f"- {name}" for name in CATEGORY_PATTERNS.keys())


class SheetClassifier:
    """
    Classifies every page in a PDF using a locked region

    Steps per page:
        1. Native text from region → regex match → done
        2. No native text → Sonnet OCR region → regex match → done
        3. OCR fails → full page to Opus with category list → done
        4. AI can't classify → unclassified
    """

    def __init__(self, config: PipelineConfig):
        self.config     = config
        self.ocr        = OCRService(api_key=config.anthropic_api_key, model=config.ocr_model)
        self.client     = Anthropic(api_key=config.anthropic_api_key)
        self.total_cost = 0.0
        self.timings: dict[str, float] = {}


    def classify_all(self,
            pdf_path: str,
            region_result: RegionResult,
            logger=None ) -> list[PageResult]:

        """Classify all pages in parallel using the locked region"""

        doc        = fitz.open(pdf_path)
        page_count = doc.page_count
        region     = region_result.region
        doc.close()

        if logger:
            logger.info(
                f"Classifying {page_count} pages with {self.config.max_workers} workers "
                f"(method: {region_result.method})"
            )

        t0      = time.perf_counter()
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {
                pool.submit(self._classify_page, pdf_path, i, region, logger): i
                for i in range(page_count)
            }

            for future in as_completed(futures):
                page_result         = future.result()
                results[page_result.page_index] = page_result

        self.timings["classify_all"] = time.perf_counter() - t0

        # Sum costs
        self.total_cost = sum(r.cost_usd for r in results.values())

        # Return ordered by page index
        ordered = [results[i] for i in range(page_count)]

        if logger:
            methods = {}
            for r in ordered:
                methods[r.method] = methods.get(r.method, 0) + 1
            logger.info(f"Classification complete: {methods}")

        return ordered


    def _classify_page(self,
            pdf_path: str,
            page_idx: int,
            region: dict,
            logger=None ) -> PageResult:

        """Run the 4-step chain for a single page"""

        doc  = fitz.open(pdf_path)
        page = doc.load_page(page_idx)

        # Step 1: Native text from region → label-aware extraction
        native_text = extract_text_from_region(page, region)
        result      = extract_sheet_number(native_text)

        if result:
            doc.close()

            return PageResult(
                page_index=page_idx,
                sheet_number=result[1],
                discipline=result[0],
                method="native",
                confidence=0.95,
            )

        # Step 2: OCR region → label-aware extraction
        img_bytes = extract_image_from_region(page, region, zoom=self.config.ocr_zoom, format='PNG')
        ocr_text  = self.ocr.extract_text(img_bytes, media_type="image/png")
        result    = extract_sheet_number(ocr_text)

        if result:
            doc.close()

            return PageResult(
                page_index=page_idx,
                sheet_number=result[1],
                discipline=result[0],
                method="ocr",
                confidence=0.90,
            )

        # Step 3: Full page → Opus (reads sheet number, we map discipline)
        ai_result = self._ai_classify_page(page, logger=logger)
        doc.close()

        if ai_result and ai_result.get("sheet_number"):
            sheet_num  = ai_result["sheet_number"]
            discipline = ai_result.get("discipline")

            # Trust our own prefix → discipline mapping over AI's guess
            our_result = match_discipline(sheet_num)
            if our_result:
                discipline = our_result[0]
                sheet_num  = our_result[1]

            return PageResult(
                page_index=page_idx,
                sheet_number=sheet_num,
                discipline=discipline,
                method="ai_vision",
                confidence=ai_result.get("confidence", 0.80),
                cost_usd=ai_result.get("cost", 0.0),
            )

        # Step 4: Unclassified
        return PageResult(
            page_index=page_idx,
            sheet_number=None,
            discipline=None,
            method="unclassified",
            confidence=0.0,
        )


    def _ai_classify_page(self, page: fitz.Page,
            logger=None ) -> dict | None:

        """Send full page to Opus for discipline classification"""

        mat       = fitz.Matrix(2.0, 2.0)
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes(output='png')
        img_opt   = optimize_image_for_api(img_bytes, max_dimension=self.config.max_image_dimension)
        b64       = base64.b64encode(img_opt).decode('utf-8')

        response = self.client.messages.create(
            model=self.config.classifier_model,
            max_tokens=256,
            system=(
                "You are classifying construction drawings by discipline.\n"
                "Find the sheet number and determine which discipline it belongs to.\n\n"
                f"Valid disciplines:\n{DISCIPLINE_LIST}\n\n"
                "Return ONLY valid JSON, no markdown:\n"
                '{"sheet_number": "A-101", "discipline": "Architectural", "confidence": 0.9}\n\n'
                "If you cannot determine the discipline, return:\n"
                '{"sheet_number": null, "discipline": null, "confidence": 0.0}'
            ),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": "Classify this construction drawing."
                    }
                ]
            }]
        )

        # Cost — Opus: $15/M input, $75/M output
        cost = (response.usage.input_tokens / 1000) * 0.015 + \
               (response.usage.output_tokens / 1000) * 0.075

        text = response.content[0].text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            parsed         = json.loads(text.strip())
            parsed["cost"] = cost

            if logger:
                logger.debug(f"  AI classify page: {parsed}")

            return parsed

        except (json.JSONDecodeError, KeyError) as e:
            if logger:
                logger.warning(f"  AI classify parse failed: {e} — raw: {text!r}")

            return None


    def get_total_cost(self ) -> float:

        """Total API cost across all pages"""

        return self.total_cost

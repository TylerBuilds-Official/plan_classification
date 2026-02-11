"""Region detection pipeline test — saves debug images and logs"""
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

import fitz
from dotenv import load_dotenv
import os

# Project root setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / '.env')

from src.plan_classification.region.region_handler import RegionHandler
from src.plan_classification.utils.pdf.pdf_utils import extract_text_from_region, extract_image_from_region
from src.plan_classification.utils.ai.ocr_service import OCRService

# ── Config ──────────────────────────────────────────────────────────────────
PDF_PATH    = Path(r"C:\Users\tylere.METALSFAB\Desktop\Dev stuff\PDFClassifyMCP\Testing\REV-M_2026-01-19_COMBINED.pdf")
DEBUG_DIR   = ROOT / "debug"
IMG_DIR     = DEBUG_DIR / "region_img"
LOG_FILE    = DEBUG_DIR / "region_detection.log"
API_KEY     = os.getenv("ANTHROPIC_API_KEY")
# ────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:

    """Configure file + console logging"""

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("region_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt       = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    file_h    = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    console_h = logging.StreamHandler(sys.stdout)

    file_h.setFormatter(fmt)
    console_h.setFormatter(fmt)
    logger.addHandler(file_h)
    logger.addHandler(console_h)

    return logger


def save_region_images(
        pdf_path: Path,
        region: dict,
        ocr: OCRService,
        logger: logging.Logger ) -> None:

    """Extract and save the detected region from every page, with native text and OCR"""

    doc        = fitz.open(str(pdf_path))
    page_count = doc.page_count

    logger.info(f"Saving region images for {page_count} pages...")

    for page_num in range(page_count):
        page        = doc.load_page(page_num)
        img_bytes   = extract_image_from_region(page, region, zoom=4.0, format='PNG')
        native_text = extract_text_from_region(page, region)
        out_path    = IMG_DIR / f"page_{page_num + 1:03d}.png"

        with open(out_path, 'wb') as f:
            f.write(img_bytes)

        # OCR the region if no native text
        ocr_text = ""
        if not native_text.strip():
            ocr_text = ocr.extract_text(img_bytes, media_type="image/png")

        display_text = native_text.strip() or ocr_text.strip() or "(empty)"
        source       = "native" if native_text.strip() else ("ocr" if ocr_text.strip() else "none")

        logger.debug(
            f"  Page {page_num + 1:3d} | [{source:6s}] {display_text!r:50s} | {out_path.name}"
        )

    doc.close()
    logger.info(f"All region images saved to {IMG_DIR}")


def run() -> None:

    """Main test entrypoint"""

    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("REGION DETECTION PIPELINE TEST")
    logger.info(f"PDF:       {PDF_PATH}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    if not PDF_PATH.exists():
        logger.error(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    if not API_KEY:
        logger.error("ANTHROPIC_API_KEY not set — cannot proceed")
        sys.exit(1)

    # Page count
    doc        = fitz.open(str(PDF_PATH))
    page_count = doc.page_count
    doc.close()
    logger.info(f"Page count: {page_count}")

    # Run detection
    handler = RegionHandler(anthropic_api_key=API_KEY)
    ocr     = OCRService(api_key=API_KEY)

    logger.info("-" * 70)
    logger.info("Starting region detection...")
    t_start = time.perf_counter()

    result = handler.auto_detect_region(str(PDF_PATH), logger=logger)

    elapsed = time.perf_counter() - t_start
    logger.info(f"Detection complete in {elapsed:.2f}s")
    logger.info("-" * 70)

    # Log results
    logger.info(f"Method:           {result.method}")
    logger.info(f"Confidence:       {result.confidence:.2%}")
    logger.info(f"Validation score: {result.validation_score:.2%}")
    logger.info(f"Cost:             ${result.cost_usd:.4f}")
    logger.info(f"Region:           {result.region}")
    logger.info(f"Samples:          {result.detected_samples}")

    # Save debug images with OCR
    logger.info("-" * 70)
    save_region_images(PDF_PATH, result.region, ocr, logger)

    logger.info("=" * 70)
    logger.info("DONE")


if __name__ == "__main__":
    run()

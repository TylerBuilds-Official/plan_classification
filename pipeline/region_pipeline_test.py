"""
Full pipeline test — region detection + parallel classification
TEST FILE
"""
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

from src.plan_classification.engine import ClassificationEngine
from src.plan_classification.pipeline_config import PipelineConfig

# ── Config ──────────────────────────────────────────────────────────────────
PDF_PATH  = Path(r"C:\Users\tylere.METALSFAB\Desktop\Dev stuff\PDFClassifyMCP\Testing\2026-02-04-REV3-COMBINED.pdf")
DEBUG_DIR = ROOT / "debug"
IMG_DIR   = DEBUG_DIR / "region_img"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = DEBUG_DIR / f"pipeline{TIMESTAMP}.log"
API_KEY   = os.getenv("ANTHROPIC_API_KEY")
# ────────────────────────────────────────────────────────────────────────────


def setup_logging() -> logging.Logger:

    """Configure file + console logging"""

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline_test")
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


def run() -> None:

    """Main test entrypoint"""

    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("FULL PIPELINE TEST")
    logger.info(f"PDF:       {PDF_PATH}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    if not PDF_PATH.exists():
        logger.error(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    if not API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Page count
    doc        = fitz.open(str(PDF_PATH))
    page_count = doc.page_count
    doc.close()
    logger.info(f"Page count: {page_count}")

    # Build engine
    config = PipelineConfig(anthropic_api_key=API_KEY)
    engine = ClassificationEngine(config)

    # Run full pipeline
    logger.info("-" * 70)
    logger.info("Phase 1: Region detection...")
    t_start = time.perf_counter()

    results = engine.classify(str(PDF_PATH), logger=logger)

    elapsed = time.perf_counter() - t_start

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("-" * 70)

    # Method breakdown
    methods = {}
    disciplines = {}
    for r in results:
        methods[r.method]         = methods.get(r.method, 0) + 1
        disc                      = r.discipline or "Unclassified"
        disciplines[disc]         = disciplines.get(disc, 0) + 1

    logger.info("By method:")
    for method, count in sorted(methods.items()):
        logger.info(f"  {method:15s} {count:4d} pages")

    logger.info("By discipline:")
    for disc, count in sorted(disciplines.items()):
        logger.info(f"  {disc:20s} {count:4d} pages")

    # Per-page detail
    logger.info("-" * 70)
    logger.info("Page details:")
    for r in results:
        logger.info(
            f"  Page {r.page_index + 1:3d} | "
            f"{r.method:12s} | "
            f"{r.sheet_number or '---':12s} | "
            f"{r.discipline or 'Unclassified':20s} | "
            f"conf={r.confidence:.0%}"
        )

    # Timing breakdown
    logger.info("-" * 70)
    logger.info("TIMING BREAKDOWN")
    for step, duration in engine.timings.items():
        logger.info(f"  {step:25s} {duration:6.2f}s")
    logger.info(f"  {'TOTAL':25s} {elapsed:6.2f}s")

    # Cost
    logger.info("-" * 70)
    logger.info(f"Total cost: ${engine.total_cost:.4f}")

    logger.info("=" * 70)
    logger.info("DONE")


if __name__ == "__main__":
    run()

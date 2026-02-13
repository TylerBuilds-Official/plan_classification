"""
Diagnostic — dump raw region text for problem pages
TEST FILE
"""
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / '.env')

from src.plan_classification.utils.pdf.pdf_utils import extract_text_from_region, extract_image_from_region
from src.plan_classification.utils.ai.ocr_service import OCRService
from src.plan_classification.constants.sheet_number_extractor import extract_sheet_number

PDF_PATH = Path(r"C:\Users\tylere.METALSFAB\Desktop\Dev stuff\PDFClassifyMCP\Testing\REV-M_2026-01-19_COMBINED.pdf")
API_KEY  = os.getenv("ANTHROPIC_API_KEY")

# The locked region from last run
REGION = {
    "x_ratio": 0.795,
    "y_ratio": 0.875,
    "w_ratio": 0.22,
    "h_ratio": 0.14,
}

PAGES = list(range(15))  # Pages 1-15

doc = fitz.open(str(PDF_PATH))
ocr = OCRService(api_key=API_KEY)

for idx in PAGES:
    page        = doc.load_page(idx)
    native_text = extract_text_from_region(page, REGION)
    img_bytes   = extract_image_from_region(page, REGION, zoom=4.0, format='PNG')
    ocr_text    = ocr.extract_text(img_bytes, media_type="image/png")
    result      = extract_sheet_number(native_text or ocr_text)

    print(f"\n{'='*60}")
    print(f"PAGE {idx + 1}")
    print(f"{'='*60}")
    print(f"NATIVE TEXT:\n{native_text!r}")
    print(f"\nOCR TEXT:\n{ocr_text!r}")
    print(f"\nEXTRACTOR RESULT: {result}")

doc.close()

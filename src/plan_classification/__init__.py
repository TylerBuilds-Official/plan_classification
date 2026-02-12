"""Classification package for PDF discipline classification"""

# Pipeline
from .engine import ClassificationEngine
from .pipeline_config import PipelineConfig

# Constants
from .constants import CATEGORY_PATTERNS

# Region detection
from .region.region_handler import RegionHandler
from .region._dataclass.region_result import RegionResult
from .region._dataclass.validation_result import ValidationResult
from .region._errors.region_detection_error import RegionDetectionError

# Classification
from .classify.sheet_classifier import SheetClassifier
from .classify._dataclass.page_result import PageResult
from .classify._errors.classification_error import ClassificationError

# Date extraction
from .date_extraction.date_extractor import DateExtractor, ExtractionResult, TierResult
from .utils.date.date_utils import FoundDate

# PDF utilities
from .utils.pdf.pdf_utils import (
    extract_text_from_region,
    extract_image_from_region,
    extract_full_page_image,
    optimize_image_for_api,
    get_pdf_page_count,
    load_pdf_page,
    make_pdf_rect,
    bulk_extract_text_from_regions,
)

# AI utilities
from .utils.ai.ocr_service import OCRService
from .utils.ai.ai_summary_service import AISummaryService, SummaryResult, AIDirResult

# Breakout
from .breakout_handler import BreakoutHandler

__all__ = [
    # Pipeline
    'ClassificationEngine',
    'PipelineConfig',

    # Constants
    'CATEGORY_PATTERNS',

    # Region
    'RegionHandler',
    'RegionResult',
    'ValidationResult',
    'RegionDetectionError',

    # Classification
    'SheetClassifier',
    'PageResult',
    'ClassificationError',

    # Date extraction
    'DateExtractor',
    'ExtractionResult',
    'TierResult',
    'FoundDate',

    # PDF utilities
    'extract_text_from_region',
    'extract_image_from_region',
    'extract_full_page_image',
    'optimize_image_for_api',
    'get_pdf_page_count',
    'load_pdf_page',
    'make_pdf_rect',
    'bulk_extract_text_from_regions',

    # AI utilities
    'OCRService',
    'AISummaryService',
    'SummaryResult',
    'AIDirResult',

    # Breakout
    'BreakoutHandler',
]

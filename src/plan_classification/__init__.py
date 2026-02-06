"""Classification package for PDF discipline classification"""
from .breakout_handler import BreakoutHandler
from .engine import (
    ClassificationEngine,
    ClassificationResult,
    ClassificationMethod,
    CATEGORY_PATTERNS
)

from .pdf_utils import (
    extract_text_from_region,
    extract_image_from_region,
    optimize_image_for_api,
    get_pdf_page_count,
    load_pdf_page,
    make_pdf_rect,
    bulk_extract_text_from_regions
)

from .ai_classifier import (
    AIClassifier, OpenAIClassifier, ClaudeClassifier, 
    ClassifierPool, PageClassification, create_classifier
)

from .region_handler import (
    RegionHandler,
    RegionResult,
    RegionDetectionError,
    COMMON_REGIONS
)

__all__ = [
    'ClassificationEngine',
    'ClassificationResult',
    'ClassificationMethod',
    'CATEGORY_PATTERNS',
    'extract_text_from_region',
    'extract_image_from_region',
    'optimize_image_for_api',
    'get_pdf_page_count',
    'load_pdf_page',
    'make_pdf_rect',
    'bulk_extract_text_from_regions',
    'AIClassifier',
    'OpenAIClassifier',
    'ClaudeClassifier',
    'ClassifierPool',
    'PageClassification',
    'create_classifier',
    'RegionHandler',
    'RegionResult',
    'RegionDetectionError',
    'COMMON_REGIONS',
    'BreakoutHandler',
]

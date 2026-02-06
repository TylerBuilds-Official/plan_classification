"""
Classification engine that coordinates text extraction, regex matching, and AI fallback
"""
import re
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class ClassificationMethod(Enum):
    """How a page was classified"""
    NATIVE_TEXT = "native_text"
    REGEX_MATCH = "regex_match"
    AI_VISION = "ai_vision"
    AI_TEXT = "ai_text"
    FAILED = "failed"


@dataclass
class ClassificationResult:
    """Result of classifying a single page"""
    page_num: int
    category: str
    confidence: float
    method: ClassificationMethod
    extracted_text: str
    sheet_number: Optional[str] = None
    cost_usd: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'page_num': self.page_num,
            'category': self.category,
            'confidence': self.confidence,
            'method': self.method.value,
            'extracted_text': self.extracted_text,
            'sheet_number': self.sheet_number,
            'cost_usd': self.cost_usd
        }


# Standard construction drawing discipline patterns
# Updated to allow optional spaces/separators between prefix and numbers
CATEGORY_PATTERNS = {
    'Architectural':    r'\bA\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Civil':            r'\bC\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Electrical':       r'\bE[LP]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',  # E, EL, EP
    'Fire Protection':  r'\bFP\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Food Service':     r'\bFS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'General':          r'\bG\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Landscape':        r'\bL[AS]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',  # L, LA, LS
    'Life Safety':      r'\bLS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Mechanical':       r'\bM[GP]?\s*(?:[-.\s]?\d+)+[A-Z]*\b',  # M, MG, MP
    'Plumbing':         r'\bP\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Security':         r'\bSS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Structural':       r'\bS\s*(?:[-.\s]?\d+)+[A-Z]*\b',
    'Technology':       r'\bT\s*(?:[-.\s]?\d+)+[A-Z]*\b'
}


class ClassificationEngine:
    """
    Hybrid classification engine
    Tier 1: Native text + regex (free, fast)
    Tier 2: AI vision/text (accurate, costs API credits)
    """
    
    def __init__(self, use_ai: bool = True, confidence_threshold: float = 0.8):
        self.use_ai = use_ai
        self.confidence_threshold = confidence_threshold
        self.total_cost = 0.0
        
    def classify_by_text(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        Classify using regex pattern matching

        Returns:
            (category, sheet_number, confidence)
        """
        if not text or not text.strip():
            return None, None, 0.0

        # Clean the text
        cleaned = re.sub(r'[^\w\-\.]', '', text.strip())

        # Try to match each category pattern
        for category, pattern in CATEGORY_PATTERNS.items():
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                sheet_number = match.group(0)
                confidence = 0.95  # High confidence for direct pattern match
                return category, sheet_number, confidence

        # Text exists but no pattern match
        return None, None, 0.3

    def classify_page_text_first(
        self,
        page_num: int,
        text: str,
        confidence_threshold: float = 0.90
    ) -> Tuple[Optional['ClassificationResult'], bool]:
        """
        Attempt text-first classification for a page.

        Args:
            page_num: Page number (0-indexed)
            text: Extracted text from title block region
            confidence_threshold: Minimum confidence to accept result

        Returns:
            (ClassificationResult or None, needs_ai_fallback: bool)
        """
        category, sheet_number, confidence = self.classify_by_text(text)

        if category and sheet_number and confidence >= confidence_threshold:
            result = ClassificationResult(
                page_num=page_num,
                category=category,
                confidence=confidence,
                method=ClassificationMethod.REGEX_MATCH,
                extracted_text=text,
                sheet_number=sheet_number,
                cost_usd=0.0
            )
            return result, False

        # Need AI fallback
        return None, True

    def get_total_cost(self) -> float:
        """Get total API cost for this session"""
        return self.total_cost

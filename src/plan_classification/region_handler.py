"""
Region Handler - Automatic Sheet Number Location Detection
Fully automated multi-tier detection system with validation
"""
import hashlib
import json
import re
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .pdf_utils import extract_text_from_region, extract_image_from_region, optimize_image_for_api
from .engine import CATEGORY_PATTERNS


@dataclass
class RegionResult:
    """Result of region detection"""
    region: Dict[str, float]
    confidence: float
    method: str  # cache, heuristic, ai_vision, ai_vision_validated, full_ocr
    cost_usd: float
    detected_samples: List[str] = None  # Sample sheet numbers found
    validation_score: float = 0.0  # % of test pages where region worked
    
    def to_dict(self) -> Dict:
        return {
            'region': self.region,
            'confidence': self.confidence,
            'method': self.method,
            'cost_usd': self.cost_usd,
            'detected_samples': self.detected_samples or [],
            'validation_score': self.validation_score
        }


@dataclass
class ValidationResult:
    """Result of validating a region against sample pages"""
    success: bool
    match_rate: float  # 0.0 - 1.0
    matched_pages: int
    total_pages: int
    extracted_numbers: List[str] = field(default_factory=list)
    failed_pages: List[int] = field(default_factory=list)


# Common sheet number locations (based on industry standards)
# Ordered by frequency in real-world drawings
COMMON_REGIONS = [
    # Bottom-right corner (most common - ~70% of drawings)
    {
        "x_ratio": 0.85,
        "y_ratio": 0.92,
        "w_ratio": 0.14,
        "h_ratio": 0.07,
        "name": "bottom-right"
    },
    # Bottom-right tight (smaller title blocks)
    {
        "x_ratio": 0.90,
        "y_ratio": 0.94,
        "w_ratio": 0.09,
        "h_ratio": 0.05,
        "name": "bottom-right-tight"
    },
    # Title block center-right
    {
        "x_ratio": 0.70,
        "y_ratio": 0.92,
        "w_ratio": 0.20,
        "h_ratio": 0.07,
        "name": "title-center-right"
    },
    # Top-right corner (some firms use this)
    {
        "x_ratio": 0.85,
        "y_ratio": 0.01,
        "w_ratio": 0.14,
        "h_ratio": 0.07,
        "name": "top-right"
    },
    # Bottom-left (rare but exists)
    {
        "x_ratio": 0.01,
        "y_ratio": 0.92,
        "w_ratio": 0.14,
        "h_ratio": 0.07,
        "name": "bottom-left"
    },
    # Right edge middle (vertical title blocks)
    {
        "x_ratio": 0.92,
        "y_ratio": 0.40,
        "w_ratio": 0.07,
        "h_ratio": 0.20,
        "name": "right-edge-middle"
    },
]

# Expanded regions for AI to consider (quadrants)
AI_SEARCH_QUADRANTS = [
    {"name": "bottom-right", "x_ratio": 0.60, "y_ratio": 0.75, "w_ratio": 0.40, "h_ratio": 0.25},
    {"name": "bottom-left", "x_ratio": 0.00, "y_ratio": 0.75, "w_ratio": 0.40, "h_ratio": 0.25},
    {"name": "top-right", "x_ratio": 0.60, "y_ratio": 0.00, "w_ratio": 0.40, "h_ratio": 0.25},
    {"name": "top-left", "x_ratio": 0.00, "y_ratio": 0.00, "w_ratio": 0.40, "h_ratio": 0.25},
    {"name": "right-strip", "x_ratio": 0.85, "y_ratio": 0.00, "w_ratio": 0.15, "h_ratio": 1.00},
    {"name": "bottom-strip", "x_ratio": 0.00, "y_ratio": 0.85, "w_ratio": 1.00, "h_ratio": 0.15},
]


class RegionCache:
    """Simple file-based cache for detected regions"""
    
    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_dir = Path.home() / ".pdfclassify_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / "region_cache.json"
        
        self.cache_file = Path(cache_file)
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save region cache: {e}")
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached region"""
        entry = self._cache.get(key)
        if entry:
            # Increment hit count
            entry['hit_count'] = entry.get('hit_count', 0) + 1
            self._save_cache()
            return entry
        return None
    
    def set(self, key: str, region: Dict, confidence: float, method: str, 
            samples: List[str] = None, validation_score: float = 0.0):
        """Cache a detected region"""
        from datetime import datetime
        
        self._cache[key] = {
            'region': region,
            'confidence': confidence,
            'method': method,
            'first_seen': datetime.now().isoformat(),
            'hit_count': 1,
            'sample_sheet_numbers': samples or [],
            'validation_score': validation_score
        }
        self._save_cache()


class RegionDetectionError(Exception):
    """Raised when region detection completely fails"""
    pass


class RegionHandler:
    """
    Fully automated sheet number region detection with validation
    
    Multi-tier strategy:
    - Tier 0: Cache lookup (instant, free)
    - Tier 1: Heuristic scan (fast, free)
    - Tier 2: AI vision search with validation (accurate, paid)
    - Tier 3: Smart full page search (comprehensive, free)
    """
    
    def __init__(self, openai_api_key: str = None, cache_file: str = None):
        self.openai_api_key = openai_api_key
        self.cache = RegionCache(cache_file)
        self.ai_classifier = None
        self.total_cost = 0.0
        
        # Initialize AI classifier if key provided
        if openai_api_key:
            from .ai_classifier import AIClassifier
            self.ai_classifier = AIClassifier(api_key=openai_api_key)
    
    def auto_detect_region(self, pdf_path: str, min_validation_rate: float = 0.6) -> RegionResult:
        """
        Fully automated region detection with validation
        
        Args:
            pdf_path: Path to PDF file
            min_validation_rate: Minimum % of pages that must validate (0.0-1.0)
            
        Returns:
            RegionResult with detected region and metadata
        """
        
        # Tier 0: Check cache
        cache_key = self._get_cache_key(pdf_path)
        if cached := self.cache.get(cache_key):
            # Quick validation check on cached region
            validation = self._validate_region(pdf_path, cached['region'], sample_size=3)
            if validation.success and validation.match_rate >= min_validation_rate:
                return RegionResult(
                    region=cached['region'],
                    confidence=cached['confidence'],
                    method='cache',
                    cost_usd=0.0,
                    detected_samples=validation.extracted_numbers,
                    validation_score=validation.match_rate
                )
            # Cache miss - region no longer valid, continue to detection
        
        # Tier 1: Heuristic scan (free & fast)
        heuristic_result = self._heuristic_detect(pdf_path, min_validation_rate)
        if heuristic_result and heuristic_result.confidence >= 0.8:
            self.cache.set(
                cache_key,
                heuristic_result.region,
                heuristic_result.confidence,
                'heuristic',
                heuristic_result.detected_samples,
                heuristic_result.validation_score
            )
            return heuristic_result
        
        # Tier 2: AI Vision with validation loop (accurate but costs)
        if self.openai_api_key:
            ai_result = self._ai_detect_with_validation(pdf_path, min_validation_rate)
            if ai_result and ai_result.confidence >= 0.7:
                self.cache.set(
                    cache_key,
                    ai_result.region,
                    ai_result.confidence,
                    ai_result.method,
                    ai_result.detected_samples,
                    ai_result.validation_score
                )
                return ai_result
        
        # Tier 3: Smart full page search (last resort)
        fallback_result = self._smart_full_page_search(pdf_path)
        if fallback_result:
            self.cache.set(
                cache_key,
                fallback_result.region,
                fallback_result.confidence,
                'full_ocr',
                fallback_result.detected_samples,
                fallback_result.validation_score
            )
            return fallback_result
        
        # Failed completely
        raise RegionDetectionError(
            "Could not automatically detect sheet number location. "
            "This drawing set may have non-standard formatting or no valid sheet numbers."
        )
    
    def _get_cache_key(self, pdf_path: str) -> str:
        """
        Generate cache key based on PDF fingerprint
        
        Same architect/firm = same sheet number location
        """
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        
        # Build fingerprint from metadata and page characteristics
        key_parts = [
            metadata.get('creator', 'unknown'),
            metadata.get('producer', 'unknown'),
            f"{doc[0].rect.width:.1f}x{doc[0].rect.height:.1f}",
        ]
        
        doc.close()
        
        cache_key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
        return cache_key
    
    def _get_sample_pages(self, pdf_path: str, sample_size: int = 5) -> List[int]:
        """
        Intelligently select sample pages, skipping likely cover sheets
        
        Args:
            pdf_path: Path to PDF
            sample_size: Number of pages to sample
            
        Returns:
            List of page indices (0-indexed)
        """
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        
        if page_count <= sample_size:
            doc.close()
            return list(range(page_count))
        
        # Skip page 0 if it looks like a cover sheet
        start_page = 0
        if page_count > 3:
            first_page = doc.load_page(0)
            first_text = first_page.get_text("text").lower()
            
            # Cover sheet indicators
            cover_indicators = ['cover', 'title sheet', 'index', 'table of contents', 
                               'drawing list', 'sheet index', 'project directory']
            if any(indicator in first_text for indicator in cover_indicators):
                start_page = 1
            
            # Also skip if first page has very little text (often a cover image)
            if len(first_text.strip()) < 100:
                start_page = 1
        
        # Sample evenly distributed pages
        available_pages = page_count - start_page
        if available_pages <= sample_size:
            indices = list(range(start_page, page_count))
        else:
            step = available_pages / sample_size
            indices = [start_page + int(i * step) for i in range(sample_size)]
        
        doc.close()
        return indices
    
    def _validate_region(self, pdf_path: str, region: Dict[str, float], 
                         sample_size: int = 5) -> ValidationResult:
        """
        Validate that a region actually contains sheet numbers
        
        Args:
            pdf_path: Path to PDF
            region: Region to validate
            sample_size: Number of pages to test
            
        Returns:
            ValidationResult with match statistics
        """
        doc = fitz.open(pdf_path)
        sample_indices = self._get_sample_pages(pdf_path, sample_size)
        
        matched_pages = 0
        extracted_numbers = []
        failed_pages = []
        
        for page_idx in sample_indices:
            page = doc.load_page(page_idx)
            text = extract_text_from_region(page, region)
            
            if sheet_num := self._extract_sheet_number(text):
                matched_pages += 1
                extracted_numbers.append(sheet_num)
            else:
                failed_pages.append(page_idx)
        
        doc.close()
        
        match_rate = matched_pages / len(sample_indices) if sample_indices else 0.0
        
        return ValidationResult(
            success=match_rate >= 0.5,
            match_rate=match_rate,
            matched_pages=matched_pages,
            total_pages=len(sample_indices),
            extracted_numbers=extracted_numbers,
            failed_pages=failed_pages
        )
    
    def _heuristic_detect(self, pdf_path: str, min_validation_rate: float) -> Optional[RegionResult]:
        """
        Tier 1: Test common locations with text extraction and validation
        
        Fast and free - works for 85-90% of typical drawings
        """
        doc = fitz.open(pdf_path)
        sample_indices = self._get_sample_pages(pdf_path, sample_size=5)
        
        best_result = None
        best_match_rate = 0.0
        
        # Test each common region
        for region_template in COMMON_REGIONS:
            matches = []
            
            for page_idx in sample_indices:
                page = doc.load_page(page_idx)
                text = extract_text_from_region(page, region_template)
                
                if sheet_num := self._extract_sheet_number(text):
                    matches.append(sheet_num)
            
            match_rate = len(matches) / len(sample_indices) if sample_indices else 0.0
            
            # Track best result even if below threshold
            if match_rate > best_match_rate:
                best_match_rate = match_rate
                best_result = RegionResult(
                    region={
                        'x_ratio': region_template['x_ratio'],
                        'y_ratio': region_template['y_ratio'],
                        'w_ratio': region_template['w_ratio'],
                        'h_ratio': region_template['h_ratio']
                    },
                    confidence=min(0.95, match_rate + 0.1),  # Boost confidence slightly
                    method='heuristic',
                    cost_usd=0.0,
                    detected_samples=matches,
                    validation_score=match_rate
                )
            
            # If we found a great match, return early
            if match_rate >= 0.8:
                doc.close()
                return best_result
        
        doc.close()
        
        # Return best result if it meets minimum threshold
        if best_result and best_match_rate >= min_validation_rate:
            return best_result
        
        return None
    
    def _extract_sheet_number(self, text: str) -> Optional[str]:
        """Check if text contains a valid sheet number"""
        if not text or not text.strip():
            return None
        
        # Try each pattern
        for category, pattern in CATEGORY_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _ai_detect_with_validation(self, pdf_path: str, min_validation_rate: float,
                                    max_retries: int = 3) -> Optional[RegionResult]:
        """
        Tier 2: Use AI vision with validation loop
        
        Two-stage approach:
        1. Coarse detection - identify which area of the page
        2. Fine detection - get precise coordinates
        3. Validation - verify it actually works
        4. Retry with feedback if validation fails
        """
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        sample_indices = self._get_sample_pages(pdf_path, sample_size=3)
        
        # Stage 1: Render sample pages for AI analysis
        images = []
        for idx in sample_indices:
            page = doc.load_page(idx)
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes(output='png')
            img_optimized = optimize_image_for_api(img_bytes, max_dimension=1500)
            images.append(img_optimized)
        
        doc.close()
        
        previous_attempts = []
        
        for attempt in range(max_retries):
            try:
                # Call AI to detect region
                ai_region = self._call_ai_region_detection(
                    images, 
                    previous_attempts=previous_attempts
                )
                
                if not ai_region:
                    continue
                
                # Validate the result
                validation = self._validate_region(pdf_path, ai_region['region'], sample_size=5)
                
                if validation.success and validation.match_rate >= min_validation_rate:
                    return RegionResult(
                        region=ai_region['region'],
                        confidence=min(0.95, validation.match_rate + 0.1),
                        method='ai_vision_validated',
                        cost_usd=ai_region['cost'],
                        detected_samples=validation.extracted_numbers,
                        validation_score=validation.match_rate
                    )
                
                # Validation failed - record for retry
                previous_attempts.append({
                    'region': ai_region['region'],
                    'match_rate': validation.match_rate,
                    'failed_pages': validation.failed_pages,
                    'ai_detected_numbers': ai_region.get('detected_numbers', []),
                    'actual_extracted': validation.extracted_numbers
                })
                
            except Exception as e:
                print(f"AI detection attempt {attempt + 1} failed: {e}")
                continue
        
        # All retries exhausted - return best attempt if any
        if previous_attempts:
            best_attempt = max(previous_attempts, key=lambda x: x['match_rate'])
            if best_attempt['match_rate'] >= min_validation_rate * 0.5:  # Lower threshold for partial success
                return RegionResult(
                    region=best_attempt['region'],
                    confidence=best_attempt['match_rate'],
                    method='ai_vision_partial',
                    cost_usd=self.total_cost,
                    detected_samples=best_attempt['actual_extracted'],
                    validation_score=best_attempt['match_rate']
                )
        
        return None
    
    def _call_ai_region_detection(self, images: List[bytes], 
                                   previous_attempts: List[Dict] = None) -> Optional[Dict]:
        """
        Call OpenAI to detect region from images with detailed prompting
        
        Args:
            images: List of page images
            previous_attempts: Previous failed attempts for context
            
        Returns:
            Dict with 'region', 'cost', 'detected_numbers'
        """
        import base64
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_api_key)
        encoded_images = [base64.b64encode(img).decode('utf-8') for img in images]
        
        # Build detailed prompt
        prompt = self._build_region_detection_prompt(previous_attempts)
        
        # Build message content
        content = [{"type": "text", "text": prompt}]
        for i, img_b64 in enumerate(encoded_images):
            content.append({
                "type": "text",
                "text": f"\n--- Page {i + 1} of {len(encoded_images)} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at analyzing construction and architectural drawings.
Your task is to precisely locate the sheet number identifier on drawing pages.

Sheet numbers follow standard discipline prefixes:
- A-### = Architectural (e.g., A-101, A1.01, A.2.03)
- S-### = Structural (e.g., S-201, S1.02)
- E-### = Electrical (e.g., E-301)
- M-### = Mechanical (e.g., M-401)
- P-### = Plumbing (e.g., P-501)
- C-### = Civil (e.g., C-101)
- G-### = General (e.g., G-001)
- L-### = Landscape (e.g., L-101)
- FP-### = Fire Protection
- And similar patterns...

The sheet number is typically located in the title block, usually in a corner of the drawing.
It should be consistent across all pages in a set."""
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1500,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Calculate cost
        cost = (response.usage.prompt_tokens / 1000) * 0.005 + \
               (response.usage.completion_tokens / 1000) * 0.015
        self.total_cost += cost
        
        # Validate the returned coordinates are sensible
        try:
            region = {
                'x_ratio': float(result['x_ratio']),
                'y_ratio': float(result['y_ratio']),
                'w_ratio': float(result['w_ratio']),
                'h_ratio': float(result['h_ratio'])
            }
            
            # Sanity checks
            for key, value in region.items():
                if not (0.0 <= value <= 1.0):
                    print(f"Warning: AI returned invalid {key}={value}, clamping to 0-1")
                    region[key] = max(0.0, min(1.0, value))
            
            # Ensure width/height are reasonable (not too small or too large)
            if region['w_ratio'] < 0.02:
                region['w_ratio'] = 0.05  # Minimum 5% width
            if region['h_ratio'] < 0.02:
                region['h_ratio'] = 0.05  # Minimum 5% height
            if region['w_ratio'] > 0.5:
                region['w_ratio'] = 0.3  # Maximum 30% width
            if region['h_ratio'] > 0.5:
                region['h_ratio'] = 0.2  # Maximum 20% height
            
            return {
                'region': region,
                'cost': cost,
                'detected_numbers': result.get('detected_numbers', []),
                'location_description': result.get('location_description', ''),
                'confidence': float(result.get('confidence', 0.8))
            }
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"Failed to parse AI response: {e}")
            return None
    
    def _build_region_detection_prompt(self, previous_attempts: List[Dict] = None) -> str:
        """Build detailed prompt for region detection"""
        
        base_prompt = """Analyze these construction drawing pages to find the SHEET NUMBER location.

TASK: Identify the bounding box that contains the sheet number (e.g., A-101, S-202, E-301).

COORDINATE SYSTEM:
- x_ratio: Distance from LEFT edge (0.0 = left edge, 1.0 = right edge)
- y_ratio: Distance from TOP edge (0.0 = top edge, 1.0 = bottom edge)
- w_ratio: Width of the box as fraction of page width
- h_ratio: Height of the box as fraction of page height

EXAMPLE COORDINATES:
- Bottom-right corner: x_ratio=0.88, y_ratio=0.93, w_ratio=0.10, h_ratio=0.05
- Top-right corner: x_ratio=0.88, y_ratio=0.02, w_ratio=0.10, h_ratio=0.05

IMPORTANT GUIDELINES:
1. The sheet number is usually in the TITLE BLOCK (typically a bordered area in a corner)
2. Look for text matching patterns like: A-101, S.201, M1.02, E-301A, etc.
3. The region should be TIGHT around the sheet number - not the entire title block
4. Add a small buffer (~1-2% of page) around the number for OCR tolerance
5. The location should be CONSISTENT across all pages shown

OUTPUT FORMAT (JSON only, no markdown):
{
    "x_ratio": 0.88,
    "y_ratio": 0.93,
    "w_ratio": 0.10,
    "h_ratio": 0.05,
    "confidence": 0.95,
    "detected_numbers": ["A-101", "A-102", "A-103"],
    "location_description": "bottom-right corner of title block"
}"""

        # Add context from previous failed attempts
        if previous_attempts:
            feedback = "\n\nPREVIOUS ATTEMPTS THAT FAILED VALIDATION:"
            for i, attempt in enumerate(previous_attempts):
                feedback += f"""
                
Attempt {i + 1}:
- Region tried: x={attempt['region']['x_ratio']:.2f}, y={attempt['region']['y_ratio']:.2f}, w={attempt['region']['w_ratio']:.2f}, h={attempt['region']['h_ratio']:.2f}
- Match rate: {attempt['match_rate']:.0%} (need at least 60%)
- AI detected: {attempt.get('ai_detected_numbers', [])}
- Actually extracted: {attempt.get('actual_extracted', [])}
- Failed on pages: {attempt.get('failed_pages', [])}"""
            
            feedback += """

Based on these failed attempts, please:
1. Adjust the coordinates to better capture the sheet number
2. Consider if the sheet number might be in a different location
3. Make sure the region is large enough to capture text reliably
4. The region may need to be larger or positioned differently"""
            
            base_prompt += feedback
        
        return base_prompt
    
    def _smart_full_page_search(self, pdf_path: str) -> Optional[RegionResult]:
        """
        Tier 3: Comprehensive full page search with intelligent prioritization
        
        Strategy:
        1. Extract all text blocks with positions from multiple pages
        2. Find sheet number patterns
        3. Calculate consistent region across pages
        4. Validate result
        """
        doc = fitz.open(pdf_path)
        sample_indices = self._get_sample_pages(pdf_path, sample_size=5)
        
        # Collect all sheet number locations across pages
        found_locations = []  # List of (page_idx, bbox, sheet_number)
        
        for page_idx in sample_indices:
            page = doc.load_page(page_idx)
            page_width = page.rect.width
            page_height = page.rect.height
            
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") != 0:  # Not a text block
                    continue
                    
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if sheet_num := self._extract_sheet_number(text):
                            bbox = span["bbox"]
                            
                            # Convert to ratios
                            location = {
                                'x_ratio': bbox[0] / page_width,
                                'y_ratio': bbox[1] / page_height,
                                'x1_ratio': bbox[2] / page_width,
                                'y1_ratio': bbox[3] / page_height
                            }
                            
                            found_locations.append((page_idx, location, sheet_num))
        
        doc.close()
        
        if not found_locations:
            return None
        
        # Find the most consistent location across pages
        # Group by approximate position
        position_groups = self._group_locations_by_position(found_locations)
        
        if not position_groups:
            return None
        
        # Take the largest group (most pages agree on this location)
        best_group = max(position_groups, key=len)
        
        # Calculate average region with padding
        region = self._calculate_average_region(best_group)
        
        # Validate
        validation = self._validate_region(pdf_path, region, sample_size=5)
        
        if validation.match_rate < 0.3:
            return None
        
        return RegionResult(
            region=region,
            confidence=min(0.85, validation.match_rate + 0.1),
            method='full_ocr',
            cost_usd=0.0,
            detected_samples=validation.extracted_numbers,
            validation_score=validation.match_rate
        )
    
    def _group_locations_by_position(self, locations: List[Tuple], 
                                      tolerance: float = 0.1) -> List[List[Tuple]]:
        """
        Group sheet number locations by approximate position
        
        Args:
            locations: List of (page_idx, bbox_ratios, sheet_number)
            tolerance: Maximum position difference to be considered same group
            
        Returns:
            List of groups, each group is a list of locations
        """
        if not locations:
            return []
        
        groups = []
        used = set()
        
        for i, loc in enumerate(locations):
            if i in used:
                continue
            
            group = [loc]
            used.add(i)
            
            # Find similar locations
            for j, other in enumerate(locations):
                if j in used:
                    continue
                
                # Compare center positions
                x1 = (loc[1]['x_ratio'] + loc[1]['x1_ratio']) / 2
                y1 = (loc[1]['y_ratio'] + loc[1]['y1_ratio']) / 2
                x2 = (other[1]['x_ratio'] + other[1]['x1_ratio']) / 2
                y2 = (other[1]['y_ratio'] + other[1]['y1_ratio']) / 2
                
                if abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_average_region(self, locations: List[Tuple], 
                                   padding: float = 0.02) -> Dict[str, float]:
        """
        Calculate average region from a group of locations with padding
        
        Args:
            locations: List of (page_idx, bbox_ratios, sheet_number)
            padding: Padding to add around the region (as ratio)
            
        Returns:
            Region dict with x_ratio, y_ratio, w_ratio, h_ratio
        """
        # Calculate bounds
        min_x = min(loc[1]['x_ratio'] for loc in locations)
        min_y = min(loc[1]['y_ratio'] for loc in locations)
        max_x = max(loc[1]['x1_ratio'] for loc in locations)
        max_y = max(loc[1]['y1_ratio'] for loc in locations)
        
        # Add padding
        min_x = max(0.0, min_x - padding)
        min_y = max(0.0, min_y - padding)
        max_x = min(1.0, max_x + padding)
        max_y = min(1.0, max_y + padding)
        
        return {
            'x_ratio': min_x,
            'y_ratio': min_y,
            'w_ratio': max_x - min_x,
            'h_ratio': max_y - min_y
        }
    
    def get_total_cost(self) -> float:
        """Get total API cost for this session"""
        return self.total_cost

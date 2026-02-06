"""
Multi-Provider Vision Classification
Supports OpenAI GPT-4o and Anthropic Claude for redundancy and rate limit management
"""
import base64
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Literal
from dataclasses import dataclass

from ...engine import CATEGORY_PATTERNS


# Cost estimates per 1K tokens (approximate)
COSTS = {
    # GPT-4.1 family
    'gpt-4.1': {'input': 0.002, 'output': 0.008},
    'gpt-4.1-mini': {'input': 0.0004, 'output': 0.0016},
    'gpt-4.1-nano': {'input': 0.0001, 'output': 0.0004},

    # GPT-5 family
    'gpt-5-mini': {'input': 0.001, 'output': 0.004},
    'gpt-5-nano': {'input': 0.0002, 'output': 0.0008},

    # Legacy
    'gpt-4o': {'input': 0.0025, 'output': 0.01},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},

    # Claude
    'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
}

ASSISTANT_PROMPT = ""


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm"""
    
    def __init__(self, calls_per_second: float = 2.0):
        self.min_interval = 1.0 / calls_per_second
        self.lock = threading.Lock()
        self.last_call = 0.0
    
    def acquire(self):
        """Wait until we can make another call"""
        with self.lock:
            now = time.time()
            wait_time = self.last_call + self.min_interval - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call = time.time()


@dataclass
class PageClassification:
    """Result of classifying a single page"""
    page_num: int
    category: str
    sheet_number: Optional[str]
    confidence: float
    validated: bool
    provider: str = "unknown"
    raw_response: Optional[str] = None
    cost_usd: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'page_num': self.page_num,
            'category': self.category,
            'sheet_number': self.sheet_number,
            'confidence': self.confidence,
            'method': f'ai_vision_{self.provider}' if not self.error else 'failed',
            'validated': self.validated,
            'cost_usd': self.cost_usd,
            'error': self.error
        }


class BaseClassifier:
    """Base class for vision classifiers"""
    
    categories = list(CATEGORY_PATTERNS.keys())
    
    def __init__(self, calls_per_second: float = 2.0, max_retries: int = 3):
        self.rate_limiter = RateLimiter(calls_per_second=calls_per_second)
        self.max_retries = max_retries
    
    def classify_page(self, page_num: int, image_bytes: bytes) -> PageClassification:
        raise NotImplementedError
    
    def _build_prompt(self) -> str:
        """Build the classification prompt"""
        return f"""Look at this construction drawing title block image and identify the SHEET NUMBER.

Sheet numbers follow discipline prefixes:
- A = Architectural (A-101, A1.01, A.2.03)
- S = Structural (S-201, S1.02)
- E = Electrical (E-301)
- M = Mechanical (M-401)
- P = Plumbing (P-501)
- C = Civil (C-101)
- G = General (G-001)
- L = Landscape (L-101)
- FP = Fire Protection
- FS = Food Service
- LS = Life Safety
- SS = Security
- T = Technology

Available categories: {', '.join(self.categories)}

Return JSON only, no markdown:
{{"sheet_number": "exactly as shown", "category": "category name", "confidence": 0.95}}

If no valid sheet number visible:
{{"sheet_number": "", "category": "Unclassified", "confidence": 0.0}}"""

    @staticmethod
    def _validate_sheet_number(sheet_number: str) -> tuple[bool, Optional[str]]:
        """Validate sheet number against known patterns"""
        if not sheet_number:
            return False, None
        for category, pattern in CATEGORY_PATTERNS.items():
            if re.search(pattern, sheet_number, re.IGNORECASE):
                return True, category
        return False, None

    @staticmethod
    def _build_batch_prompt(page_count: int) -> str:
        """Build prompt for batch classification"""

        return f"""
                Analyze these {page_count} construction drawing title block images.
                For EACH image, identify the sheet number.
                
                Sheet number patterns:
                - A = Architectural (A-101, A1.01)
                - S = Structural, E = Electrical, M = Mechanical, P = Plumbing
                - C = Civil, G = General, L = Landscape
                - FP = Fire Protection, FS = Food Service, LS = Life Safety, SS = Security, T = Technology
                
                Return a JSON object with a "results" array containing one object per image, in order:
                {{"results": [
                  {{"image": 1, "sheet_number": "A-101", "category": "Architectural", "confidence": 0.95}},
                  {{"image": 2, "sheet_number": "S-201", "category": "Structural", "confidence": 0.90}}
                ]}}
                
                If no sheet number visible for an image:
                {{"image": N, "sheet_number": "", "category": "Unclassified", "confidence": 0.0}}
                        
                """

    def classify_batch(
        self,
        pages: List[tuple],
        batch_size: int = 8,
        max_concurrent: int = 4,
        on_progress: callable = None
    ) -> List[PageClassification]:
        """
        Classify multiple pages, batching API calls with parallel execution.

        Args:
            pages: List of (page_num, image_bytes) tuples
            batch_size: Max images per API call
            max_concurrent: Max number of batches to process in parallel
            on_progress: Optional callback(completed_pages, total_pages) called after each batch completes

        Returns:
            List of PageClassification results
        """
        if not pages:
            return []

        total_pages = len(pages)

        # Split pages into batches
        batches = []
        for i in range(0, len(pages), batch_size):
            batches.append(pages[i:i + batch_size])

        if len(batches) == 1:
            # Single batch - no parallelism needed
            results = self._classify_batch_internal(batches[0])
            if on_progress:
                on_progress(total_pages, total_pages)
            return results

        # Process batches in parallel
        results = []
        batch_results_map = {}
        completed_pages = 0
        progress_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all batches
            future_to_batch_idx = {
                executor.submit(self._classify_batch_internal, batch): idx
                for idx, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    batch_results = future.result()
                    batch_results_map[batch_idx] = batch_results
                except Exception as e:
                    # If a batch completely fails, create error results
                    batch = batches[batch_idx]
                    batch_results_map[batch_idx] = [
                        PageClassification(
                            page_num=page_num,
                            category="Unclassified",
                            sheet_number=None,
                            confidence=0.0,
                            validated=False,
                            provider=getattr(self, 'provider', 'unknown'),
                            error=str(e)
                        )
                        for page_num, _ in batch
                    ]

                # Report progress after each batch completes
                if on_progress:
                    with progress_lock:
                        completed_pages += len(batches[batch_idx])
                        on_progress(completed_pages, total_pages)

        # Reassemble results in original order
        for idx in range(len(batches)):
            results.extend(batch_results_map[idx])

        return results

    def _classify_batch_internal(self, batch: List[tuple]) -> List[PageClassification]:
        """Override in subclass to implement batched API call"""
        # Default: fall back to sequential classification
        return [self.classify_page(page_num, image) for page_num, image in batch]


class OpenAIClassifier(BaseClassifier):
    """OpenAI GPT Vision classifier"""
    
    def __init__(self, api_key: str, model: str = "gpt-5-mini",
                 calls_per_second: float = 8.0, max_retries: int = 3):
        super().__init__(calls_per_second, max_retries)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"
        
    def classify_page(self, page_num: int, image_bytes: bytes) -> PageClassification:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(min(2 ** attempt, 10))
                
                self.rate_limiter.acquire()
                
                # GPT-5 uses max_completion_tokens, older models use max_tokens
                token_param = "max_completion_tokens" if "gpt-5" in self.model else "max_tokens"
                
                # Build kwargs - GPT-5 may have different temp handling
                kwargs = {
                    token_param: 500,  # Increased from 200 - GPT-5 needs more
                    "response_format": {"type": "json_object"}
                }


                # Only add temperature for non-GPT-5 models (or test if GPT-5 accepts it)
                if "gpt-5" in self.model:
                    kwargs["temperature"] = 1.0
                else:
                    kwargs["temperature"] = 0.5 # 0.5 for consistent response format

                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at reading construction drawing sheet numbers. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self._build_prompt()},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    **kwargs
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Get costs - handle model name variations
                model_key = self.model
                if model_key not in COSTS:
                    # Try to find a matching base model
                    for key in COSTS:
                        if key in model_key or model_key.startswith(key.split('-')[0]):
                            model_key = key
                            break
                costs = COSTS.get(model_key, {'input': 0.001, 'output': 0.004})  # Default fallback
                cost = (
                    (response.usage.prompt_tokens / 1000) * costs['input'] +
                    (response.usage.completion_tokens / 1000) * costs['output']
                )
                
                sheet_number = result.get('sheet_number', '').strip()
                category = result.get('category', 'Unclassified')
                confidence = float(result.get('confidence', 0.0))
                
                validated, validated_category = self._validate_sheet_number(sheet_number)
                if validated and validated_category and validated_category != category:
                    category = validated_category
                
                return PageClassification(
                    page_num=page_num,
                    category=category,
                    sheet_number=sheet_number if sheet_number else None,
                    confidence=confidence,
                    validated=validated,
                    provider=self.provider,
                    raw_response=response.choices[0].message.content,
                    cost_usd=cost
                )
                
            except Exception as e:
                last_error = str(e)
                # Log detailed error for debugging 400s
                if "400" in last_error or "Bad Request" in last_error or "invalid" in last_error.lower():
                    print(f"\n[ERROR] API rejected request: {last_error[:200]}")
                    print(f"        Model: {self.model}")
                    break  # Don't retry 400 errors
                if attempt < self.max_retries:
                    continue
        
        return PageClassification(
            page_num=page_num,
            category="Unclassified",
            sheet_number=None,
            confidence=0.0,
            validated=False,
            provider=self.provider,
            cost_usd=0.0,
            error=last_error
        )

    def _classify_batch_internal(self, batch: List[tuple]) -> List[PageClassification]:
        """Classify multiple images in a single API call with retry logic."""
        if not batch:
            return []

        try:
            return self._classify_batch_internal_impl(batch)
        except Exception as e:
            # Batch failed - retry once before falling back to sequential
            print(f"[WARN] Batch classification failed: {e}, retrying batch...")
            try:
                time.sleep(1)  # Brief pause before retry
                return self._classify_batch_internal_impl(batch)
            except Exception as retry_error:
                print(f"[WARN] Batch retry failed: {retry_error}, falling back to sequential")
                return [self.classify_page(page_num, image) for page_num, image in batch]

    def _classify_batch_internal_impl(self, batch: List[tuple]) -> List[PageClassification]:
        """Internal implementation of batch classification (for retry logic)."""
        if not batch:
            return []

        self.rate_limiter.acquire()

        # Build content array with all images
        content = []
        for idx, (page_num, image_bytes) in enumerate(batch):
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            content.append({
                "type": "text",
                "text": f"Image {idx + 1} (Page {page_num + 1}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low"
                }
            })

        content.append({"type": "text", "text": self._build_batch_prompt(len(batch))})

        token_param = "max_completion_tokens" if "gpt-5" in self.model else "max_tokens"
        kwargs = {
            token_param: 2000,
            "response_format": {"type": "json_object"}
        }
        if "gpt-5" not in self.model:
            kwargs["temperature"] = 0.1

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": content}
            ],
            **kwargs
        )

        result_data = json.loads(response.choices[0].message.content)
        results_array = result_data.get('results', result_data)
        if not isinstance(results_array, list):
            results_array = [results_array]

        # Calculate cost
        model_key = self.model
        if model_key not in COSTS:
            for key in COSTS:
                if key in model_key:
                    model_key = key
                    break
        costs = COSTS.get(model_key, {'input': 0.001, 'output': 0.004})
        total_cost = (
            (response.usage.prompt_tokens / 1000) * costs['input'] +
            (response.usage.completion_tokens / 1000) * costs['output']
        )
        cost_per_page = total_cost / len(batch)

        # Map results back to pages
        classifications = []
        for idx, (page_num, _) in enumerate(batch):
            if idx < len(results_array):
                item = results_array[idx]
                sheet_number = str(item.get('sheet_number', '')).strip()
                category = item.get('category', 'Unclassified')
                confidence = float(item.get('confidence', 0.0))

                validated, validated_category = self._validate_sheet_number(sheet_number)
                if validated and validated_category and validated_category != category:
                    category = validated_category

                classifications.append(PageClassification(
                    page_num=page_num,
                    category=category,
                    sheet_number=sheet_number if sheet_number else None,
                    confidence=confidence,
                    validated=validated,
                    provider=self.provider,
                    cost_usd=cost_per_page
                ))
            else:
                classifications.append(PageClassification(
                    page_num=page_num,
                    category="Unclassified",
                    sheet_number=None,
                    confidence=0.0,
                    validated=False,
                    provider=self.provider,
                    error="Missing result in batch response"
                ))

        return classifications


class ClaudeClassifier(BaseClassifier):
    """Anthropic Claude Vision classifier"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 calls_per_second: float = 4.0, max_retries: int = 3):
        super().__init__(calls_per_second, max_retries)
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.provider = "anthropic"
        
    def classify_page(self, page_num: int, image_bytes: bytes) -> PageClassification:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(min(2 ** attempt, 10))
                
                self.rate_limiter.acquire()
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_b64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self._build_prompt()
                                }
                            ]
                        }
                    ]
                )
                
                # Parse JSON from response
                text = response.content[0].text
                # Handle potential markdown wrapping
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                result = json.loads(text.strip())
                
                # Estimate cost (Claude pricing)
                costs = COSTS.get('claude-3-5-sonnet', COSTS['claude-3-5-sonnet'])
                cost = (
                    (response.usage.input_tokens / 1000) * costs['input'] +
                    (response.usage.output_tokens / 1000) * costs['output']
                )
                
                sheet_number = result.get('sheet_number', '').strip()
                category = result.get('category', 'Unclassified')
                confidence = float(result.get('confidence', 0.0))
                
                validated, validated_category = self._validate_sheet_number(sheet_number)
                if validated and validated_category and validated_category != category:
                    category = validated_category
                
                return PageClassification(
                    page_num=page_num,
                    category=category,
                    sheet_number=sheet_number if sheet_number else None,
                    confidence=confidence,
                    validated=validated,
                    provider=self.provider,
                    raw_response=text,
                    cost_usd=cost
                )
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    continue
        
        return PageClassification(
            page_num=page_num,
            category="Unclassified",
            sheet_number=None,
            confidence=0.0,
            validated=False,
            provider=self.provider,
            cost_usd=0.0,
            error=last_error
        )

    def _classify_batch_internal(self, batch: List[tuple]) -> List[PageClassification]:
        """Classify multiple images in a single API call with retry logic."""
        if not batch:
            return []

        try:
            return self._classify_batch_internal_impl(batch)
        except Exception as e:
            # Batch failed - retry once before falling back to sequential
            print(f"[WARN] Batch classification failed: {e}, retrying batch...")
            try:
                time.sleep(1)  # Brief pause before retry
                return self._classify_batch_internal_impl(batch)
            except Exception as retry_error:
                print(f"[WARN] Batch retry failed: {retry_error}, falling back to sequential")
                return [self.classify_page(page_num, image) for page_num, image in batch]

    def _classify_batch_internal_impl(self, batch: List[tuple]) -> List[PageClassification]:
        """Internal implementation of batch classification (for retry logic)."""
        self.rate_limiter.acquire()

        # Build content array with all images
        content = []
        for idx, (page_num, image_bytes) in enumerate(batch):
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            content.append({
                "type": "text",
                "text": f"Image {idx + 1} (Page {page_num + 1}):"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64
                }
            })

        content.append({"type": "text", "text": self._build_batch_prompt(len(batch))})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": content}]
        )

        # Parse JSON from response
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result_data = json.loads(text.strip())
        results_array = result_data.get('results', result_data)
        if not isinstance(results_array, list):
            results_array = [results_array]

        # Calculate cost
        costs = COSTS.get('claude-3-5-sonnet', COSTS['claude-3-5-sonnet'])
        total_cost = (
            (response.usage.input_tokens / 1000) * costs['input'] +
            (response.usage.output_tokens / 1000) * costs['output']
        )
        cost_per_page = total_cost / len(batch)

        # Map results back to pages
        classifications = []
        for idx, (page_num, _) in enumerate(batch):
            if idx < len(results_array):
                item = results_array[idx]
                sheet_number = str(item.get('sheet_number', '')).strip()
                category = item.get('category', 'Unclassified')
                confidence = float(item.get('confidence', 0.0))

                validated, validated_category = self._validate_sheet_number(sheet_number)
                if validated and validated_category and validated_category != category:
                    category = validated_category

                classifications.append(PageClassification(
                    page_num=page_num,
                    category=category,
                    sheet_number=sheet_number if sheet_number else None,
                    confidence=confidence,
                    validated=validated,
                    provider=self.provider,
                    cost_usd=cost_per_page
                ))
            else:
                classifications.append(PageClassification(
                    page_num=page_num,
                    category="Unclassified",
                    sheet_number=None,
                    confidence=0.0,
                    validated=False,
                    provider=self.provider,
                    error="Missing result in batch response"
                ))

        return classifications


class ClassifierPool:
    """Pool of classifiers with shared rate limiting"""
    
    def __init__(self, classifiers: List[BaseClassifier]):
        self.classifiers = classifiers
        self._index = 0
        self._lock = threading.Lock()
        
    def get_classifier(self) -> BaseClassifier:
        with self._lock:
            classifier = self.classifiers[self._index]
            self._index = (self._index + 1) % len(self.classifiers)
            return classifier
    
    def classify_page(self, page_num: int, image_bytes: bytes) -> PageClassification:
        classifier = self.get_classifier()
        return classifier.classify_page(page_num, image_bytes)
    
    @property
    def effective_calls_per_second(self) -> float:
        return sum(1.0 / c.rate_limiter.min_interval for c in self.classifiers)


def create_classifier(
    provider: Literal["openai", "anthropic", "auto"] = "auto",
    openai_key: str = None,
    anthropic_key: str = None,
    model: str = None,
    calls_per_second: float = 4.0
) -> BaseClassifier:
    """
    Factory function to create appropriate classifier
    
    Args:
        provider: "openai", "anthropic", or "auto" (tries both)
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
        model: Override default model
        calls_per_second: Rate limit
        
    Returns:
        Configured classifier instance
    """
    if provider == "anthropic" or (provider == "auto" and anthropic_key):
        return ClaudeClassifier(
            api_key=anthropic_key,
            model=model or "claude-sonnet-4-20250514",
            calls_per_second=calls_per_second
        )
    elif provider == "openai" or (provider == "auto" and openai_key):
        return OpenAIClassifier(
            api_key=openai_key,
            model=model or "gpt-5-mini",  # 500K TPM
            calls_per_second=calls_per_second
        )
    else:
        raise ValueError("No API key provided for any supported provider")


# Keep old class name for backwards compatibility
AIClassifier = OpenAIClassifier

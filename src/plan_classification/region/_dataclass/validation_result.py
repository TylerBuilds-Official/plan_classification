from dataclasses import dataclass, field
from typing import List

@dataclass
class ValidationResult:
    """Result of validating a region against sample pages"""
    success: bool
    match_rate: float  # 0.0 - 1.0
    matched_pages: int
    total_pages: int
    extracted_numbers: List[str] = field(default_factory=list)
    failed_pages: List[int] = field(default_factory=list)


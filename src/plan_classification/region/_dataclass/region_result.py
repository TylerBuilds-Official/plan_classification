from dataclasses import dataclass

from typing import Dict, List

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

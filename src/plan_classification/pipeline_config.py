"""Shared configuration for the classification pipeline"""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """All tunable parameters for region detection and classification"""

    # API
    anthropic_api_key: str = ""

    # Models
    vision_model:     str = "claude-opus-4-5-20251101"
    ocr_model:        str = "claude-sonnet-4-5-20250929"
    classifier_model: str = "claude-opus-4-5-20251101"

    # Parallelization
    max_workers: int = 8

    # Image rendering
    max_image_dimension: int   = 2048
    ocr_zoom:            float = 4.0

    # Region detection
    region_padding:      float = 0.025
    sample_count:        int   = 3
    validation_samples:  int   = 5
    min_validation_rate: float = 0.5

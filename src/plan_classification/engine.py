"""
Classification Engine — Clean orchestrator for the 2-phase pipeline

Phase 1: Region detection (RegionHandler)
Phase 2: Per-page classification (SheetClassifier)
"""
from .pipeline_config import PipelineConfig
from .region.region_handler import RegionHandler
from .region._dataclass.region_result import RegionResult
from .classify.sheet_classifier import SheetClassifier
from .classify._dataclass.page_result import PageResult


class ClassificationEngine:
    """
    Two-phase construction drawing classification

    Phase 1 — Lock the sheet number region (once per set)
    Phase 2 — Classify every page using the locked region (parallel)
    """

    def __init__(self, config: PipelineConfig):
        self.config          = config
        self.region_handler  = RegionHandler(config)
        self.sheet_classifier = SheetClassifier(config)


    def classify(self,
            pdf_path: str,
            logger=None,
            on_progress: callable = None ) -> list[PageResult]:

        """Run the full pipeline: detect region → classify all pages"""

        if on_progress:
            on_progress({"phase": "region_detection", "status": "started"})

        # Phase 1
        region = self.region_handler.auto_detect_region(pdf_path, logger=logger)

        if on_progress:
            on_progress({"phase": "region_detection", "status": "completed"})

        # Phase 2
        if on_progress:
            on_progress({"phase": "classification", "status": "started"})

        results = self.sheet_classifier.classify_all(
            pdf_path, region, logger=logger, on_progress=on_progress,
        )

        if on_progress:
            on_progress({"phase": "classification", "status": "completed"})

        return results


    @property
    def region_result(self ) -> RegionResult | None:

        """Access the region result after classification"""

        return getattr(self.region_handler, '_last_region', None)


    @property
    def timings(self ) -> dict[str, float]:

        """Combined timings from both phases"""

        combined = dict(self.region_handler.timings)
        combined.update(self.sheet_classifier.timings)

        return combined


    @property
    def total_cost(self ) -> float:

        """Total API cost across both phases"""

        return self.region_handler.get_total_cost() + self.sheet_classifier.get_total_cost()

"""Result of classifying a single page"""
from dataclasses import dataclass


@dataclass
class PageResult:
    """Classification result for one page"""

    page_index:   int
    sheet_number: str | None
    discipline:   str | None
    method:       str
    confidence:   float = 0.0
    cost_usd:     float = 0.0

    def to_dict(self ) -> dict:

        return {
            'page_index':   self.page_index,
            'sheet_number': self.sheet_number,
            'discipline':   self.discipline,
            'method':       self.method,
            'confidence':   self.confidence,
            'cost_usd':     self.cost_usd,
        }

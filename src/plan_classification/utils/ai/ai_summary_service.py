"""
AI Summary Service — Anthropic-only

Handles dirname generation and result summaries for classification output.
Date extraction has moved to date_extraction.DateExtractor.
"""
import json
from dataclasses import dataclass

from anthropic import Anthropic


@dataclass
class SummaryResult:

    summary:    str
    confidence: float


@dataclass
class AIDirResult:

    dir_name: str


# ── Models ────────────────────────────────────────────────────────
SONNET = "claude-sonnet-4-5-20250929"

# ── Prompts ───────────────────────────────────────────────────────
DIRNAME_PROMPT = (
    "You are a construction document classification expert.\n"
    "Given a PDF filename, generate a concise directory name representing "
    "the project's scope and discipline.\n\n"
    "Rules:\n"
    "- Use PascalCase_With_Underscores (e.g. Municipal_Water_Treatment_Expansion)\n"
    "- No spaces, special characters, or trailing underscores\n"
    "- 1-6 words maximum\n"
    "- Use standard terminology, preferably a direct concise reference to the filename\n"
    "- Do not guess specific addresses, dates, or project numbers unless clearly present\n"
    "- Return ONLY the directory name, nothing else\n"
    "\nExample:\n"
    "- Input: '2023-04-15-Project-Plan.pdf'\n"
    "- Output: 'Project_Plans'"
)

SUMMARY_PROMPT = (
    "Summarize the following construction plan classification results "
    "for a non-technical construction professional.\n\n"
    "Rules:\n"
    "- 1-2 short paragraphs (3 only if truly necessary)\n"
    "- 500 character hard limit\n"
    "- Explain what was found, how it was categorized, and any notable patterns\n"
    "- Do not restate raw data — summarize meaning and outcomes\n"
    "- Use plain language, not technical jargon"
)


class AISummaryService:

    """
    AI-powered summary and directory naming.

    Uses Sonnet for text tasks (dirname, summary).
    """

    def __init__(self,
            client: Anthropic ) -> None:

        self.client = client


    def create_dirname(self,
            filename: str ) -> AIDirResult:

        """Generate a project directory name from a PDF filename."""

        prompt  = DIRNAME_PROMPT + "\nFilename:\n" + filename
        dirname = self._text_request(prompt, model=SONNET)

        return AIDirResult(dirname)


    def create_summary(self,
            json_data: dict,
            confidence_results: float ) -> SummaryResult:

        """Generate a human-readable summary of classification results."""

        prompt = (
            SUMMARY_PROMPT
            + "\nData:\n" + json.dumps(json_data)
            + "\nSystem confidence in the results by regex validation:\n"
            + str(confidence_results)
        )

        response = self._text_request(prompt, model=SONNET)

        return SummaryResult(response, confidence=confidence_results)


    # ── Private API helpers ───────────────────────────────────────

    def _text_request(self,
            prompt: str,
            model: str ) -> str:

        """Simple text completion."""

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

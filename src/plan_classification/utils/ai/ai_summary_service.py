import json
from dataclasses import dataclass
from openai import OpenAI
from anthropic import Anthropic


@dataclass
class SummaryResult:
    summary: str
    confidence: float

@dataclass
class AIDirResult:
    dir_name: str

class AISummaryService:
    def __init__(self, client: OpenAI | Anthropic = None):
        self.client = client
        self.model = None
        self.provider: str = None

        self.summary_prompts = {
            "directory": (
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
            ),

            "results": (
                "Summarize the following construction plan classification results "
                "for a non-technical construction professional.\n\n"
                "Rules:\n"
                "- 1-2 short paragraphs (3 only if truly necessary)\n"
                "- 500 character hard limit\n"
                "- Explain what was found, how it was categorized, and any notable patterns\n"
                "- Do not restate raw data — summarize meaning and outcomes\n"
                "- Use plain language, not technical jargon"
            ),
        }

        if isinstance(self.client, Anthropic):
            self.model = "claude-sonnet-4-5-20250929"
            self.provider = "anthropic"

        elif isinstance(self.client, OpenAI):
            self.model = "gpt-5-mini"
            self.provider = "openai"

        else:
            raise ValueError("Invalid client type")


    def create_dirname(self, filename: str) -> AIDirResult:

        if self.provider == "anthropic":
            dirname = self._get_anthropic_response(self.summary_prompts["directory"] + '\nFilename:\n' + filename)
            return AIDirResult(dirname)

        elif self.provider == "openai":
            dirname = self._get_openai_response(self.summary_prompts["directory"] + '\nFilename:\n' + filename)
            return AIDirResult(dirname)

        else:
            raise ValueError("Invalid provider type. Check provider key.")


    def create_summary(self, json_data: dict, confidence_results: float) -> SummaryResult:
        if self.provider == "anthropic":
            response = self._get_anthropic_response(self.summary_prompts["results"] + '\nData:\n' + json.dumps(json_data) + '\nSystem confidence in the results by regex validation:\n' + str(confidence_results))
            return SummaryResult(response, confidence=confidence_results)

        elif self.provider == "openai":
            response = self._get_openai_response(self.summary_prompts["results"] + '\nData:\n' + json.dumps(json_data) + '\nSystem confidence in the results by regex validation:\n' + str(confidence_results))
            return SummaryResult(response, confidence=confidence_results)

        else:
            raise ValueError("Invalid provider type. Check provider key.")


    def _get_anthropic_response(self, prompt: str) -> str:
        request = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}])
        response = request.content[0].text
        return response

    def _get_openai_response(self, prompt: str) -> str:
        request = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}])

        response = request.choices[0].message.content
        return response
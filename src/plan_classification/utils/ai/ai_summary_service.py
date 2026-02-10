import json
import base64
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

@dataclass
class DateResult:
    date: str | None       # MMDDYY format or None if not found
    raw_response: str      # Raw model response for debugging
    category: str          # Which category this was extracted from


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

            "date": (
                "Extract the latest revision or release date from this construction drawing title block.\n\n"
                "Rules:\n"
                "- Look for revision tables, issue dates, or date stamps in the title block area\n"
                "- If multiple dates exist, return the MOST RECENT one\n"
                "- Return ONLY the date in MMDDYY format (e.g. 020626 for Feb 6, 2026)\n"
                "- If no date is found, return exactly: NONE\n"
                "- Do not return any other text, explanation, or formatting"
            )
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


    def extract_date_from_image(self, image_bytes: bytes, category: str, logger=None) -> DateResult:
        """
        Extract the drawing release date from a title block image using AI vision.
        
        Args:
            image_bytes: JPEG/PNG bytes of the title block region
            category: Category name (for tracking which class this came from)
            logger: Optional debug_logger module for logging
            
        Returns:
            DateResult with date in MMDDYY format, or None if not found
        """
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = self.summary_prompts["date"]

        if logger:
            logger.info(f'[Date:{category}] Sending vision request to {self.provider} ({self.model})')
            logger.debug(f'[Date:{category}] Image size: {len(image_bytes)} bytes, b64 length: {len(b64_image)}')

        if self.provider == "anthropic":
            raw = self._get_anthropic_vision(prompt, b64_image)
        elif self.provider == "openai":
            raw = self._get_openai_vision(prompt, b64_image)
        else:
            raise ValueError("Invalid provider type.")

        raw = raw.strip()
        date = self._parse_date_response(raw)

        if logger:
            logger.info(f'[Date:{category}] Raw response: "{raw}" -> Parsed: {date}')

        return DateResult(date=date, raw_response=raw, category=category)


    def extract_dates_from_classes(
        self,
        category_images: dict[str, bytes],
        logger=None
    ) -> dict[str, str]:
        """
        Extract dates from one representative page per category.
        Fills missing dates using the first successful date found.
        
        Args:
            category_images: {category_name: image_bytes} — one page per category
            logger: Optional debug_logger module for logging
            
        Returns:
            {category_name: "MMDDYY"} for ALL categories (filled with fallback if needed)
        """
        date_map: dict[str, str | None] = {}
        fallback_date: str | None = None

        # Extract date from each category
        for category, image_bytes in category_images.items():
            try:
                result = self.extract_date_from_image(image_bytes, category, logger=logger)
                date_map[category] = result.date
                if result.date and not fallback_date:
                    fallback_date = result.date
                    if logger:
                        logger.info(f'[Date] First successful date found: {result.date} (from {category})')
            except Exception as e:
                if logger:
                    logger.error(f'[Date] Vision call failed for {category}: {e}', exc=e)
                print(f"[Date] Failed for {category}: {e}")
                date_map[category] = None

        # Fill missing with fallback from another category, then today's date
        from datetime import datetime
        final_fallback = fallback_date or datetime.now().strftime("%m%d%y")

        if logger:
            logger.info(f'[Date] Pre-fill date_map: {date_map}')
            logger.info(f'[Date] Fallback date: {final_fallback} (from_extraction={fallback_date is not None})')

        for category in date_map:
            if not date_map[category]:
                if logger:
                    logger.info(f'[Date] Filling {category} with fallback: {final_fallback}')
                date_map[category] = final_fallback

        if logger:
            logger.info(f'[Date] Final date_map: {date_map}')

        return date_map


    @staticmethod
    def _parse_date_response(raw: str) -> str | None:
        """Parse the model's response into a clean MMDDYY string or None."""
        import re
        cleaned = raw.strip().upper()

        if cleaned == "NONE" or not cleaned:
            return None

        # Try to extract exactly 6 digits
        match = re.search(r'\b(\d{6})\b', cleaned)
        if match:
            return match.group(1)

        # Try common formats and convert: MM/DD/YY, MM-DD-YY, MM.DD.YY
        date_match = re.search(r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})', cleaned)
        if date_match:
            m, d, y = date_match.groups()
            y = y[-2:]  # Take last 2 digits if 4-digit year
            return f"{int(m):02d}{int(d):02d}{y}"

        return None


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

    def _get_anthropic_vision(self, prompt: str, b64_image: str) -> str:
        """Send a vision request to Anthropic with an image."""
        request = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }])
        return request.content[0].text

    def _get_openai_vision(self, prompt: str, b64_image: str) -> str:
        """Send a vision request to OpenAI with an image."""
        request = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=16000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }])
        return request.choices[0].message.content

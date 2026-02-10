import fitz
import os
from datetime import datetime
from pathlib import Path


class BreakoutHandler:
    def __init__(self, classification_results: list, pdf_path: str, output_dir: str=None):
        """
        Args:
            classification_results: The full list of dicts from classify_pdf (has 'results', 'summary', etc.)
            pdf_path: Path to the source PDF
            output_dir: Where to save the breakout PDFs
        """
        self.results = classification_results
        self.pdf_path = pdf_path
        # Convert output_dir to Path if it's a string
        self.output_dir = Path(output_dir) if output_dir else None
        self.doc = fitz.open(pdf_path)

    @staticmethod
    def _get_contiguous_ranges(sorted_pages: list) -> list:
        """Group sorted page numbers into contiguous ranges.
        e.g. [0,1,2,5,6,9] -> [(0,2), (5,6), (9,9)]
        """
        if not sorted_pages:
            return []
        ranges = []
        start = sorted_pages[0]
        end = start
        for page in sorted_pages[1:]:
            if page == end + 1:
                end = page
            else:
                ranges.append((start, end))
                start = page
                end = page
        ranges.append((start, end))
        return ranges

    def breakout(self, on_progress: callable = None, date_map: dict = None):
        """Main method to split PDF by discipline.
        
        Args:
            on_progress: Optional callback(current_category, total_categories, category_name)
                         called after each category file is created.
            date_map: Optional {category: "MMDDYY"} for per-category date prefixes.
                      Falls back to today's date if not provided.
        """
        # Group pages by category
        # results[i] has: page_num, category, confidence, method, etc.

        category_pages = {} # Dict of lists of page numbers, keyed by category
        created_files = [] # List of dicts with category, page_count, and output_path

        for result in self.results:
            category = result['category']
            page_num = result['page_num']

            if category not in category_pages:
                category_pages[category] = []
            category_pages[category].append(page_num)

        # Default date prefix (used when date_map doesn't have a category)
        default_date = datetime.now().strftime("%m%d%y")

        # Ensure output dir is set
        if self.output_dir is None:
            self.output_dir = Path(Path.home()) / 'Desktop' / 'breakout'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output PDFs
        total_categories = len(category_pages)
        for cat_idx, (category, page_nums) in enumerate(category_pages.items()):
            output_pdf = fitz.open()
            pages_added = 0

            # Batch contiguous pages into ranges for fewer insert_pdf calls
            for range_start, range_end in self._get_contiguous_ranges(sorted(page_nums)):
                try:
                    output_pdf.insert_pdf(self.doc, from_page=range_start, to_page=range_end)
                    pages_added += (range_end - range_start + 1)
                except Exception as e:
                    print(f"Warning: Could not add pages {range_start}-{range_end} to {category}: {e}")
            
            if pages_added == 0:
                output_pdf.close()
                continue

            date_prefix = (date_map or {}).get(category, default_date)
            filename = f"{date_prefix}_{category}.pdf"
            output_path = os.path.join(self.output_dir, filename)
            
            # Save with reasonable middle-ground options, fallback to minimal
            saved = False
            try:
                output_pdf.save(output_path, garbage=2, deflate=True)
                saved = True
            except Exception:
                try:
                    output_pdf.save(output_path)
                    saved = True
                except Exception as e:
                    print(f"Warning: Could not save {category} PDF: {e}")
            
            output_pdf.close()

            if saved:
                display_pages = [p + 1 for p in sorted(page_nums)]
                created_files.append({
                    'category': category,
                    'page_count': pages_added,
                    'output_path': output_path,
                    'page_numbers': display_pages
                })

            if on_progress:
                on_progress(cat_idx + 1, total_categories, category)

        self.doc.close()

        return { 'category_pages': category_pages, 'created_files': created_files }
import json
from pathlib import Path
from typing import Dict, List, Optional


class RegionCache:
    """Simple file-based cache for detected regions"""

    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_dir = Path.home() / ".pdfclassify_cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / "region_cache.json"

        self.cache_file = Path(cache_file)
        self._cache = self._load_cache()


    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}


    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save region cache: {e}")


    def get(self, key: str) -> Optional[Dict]:
        """Get cached region"""
        entry = self._cache.get(key)
        if entry:
            # Increment hit count
            entry['hit_count'] = entry.get('hit_count', 0) + 1
            self._save_cache()
            return entry
        return None


    def set(self, key: str, region: Dict, confidence: float, method: str,
            samples: List[str] = None, validation_score: float = 0.0):
        """Cache a detected region"""
        from datetime import datetime

        self._cache[key] = {
            'region': region,
            'confidence': confidence,
            'method': method,
            'first_seen': datetime.now().isoformat(),
            'hit_count': 1,
            'sample_sheet_numbers': samples or [],
            'validation_score': validation_score
        }
        self._save_cache()


    def clear(self):
        """Clear the entire cache"""
        self._cache = {}
        self._save_cache()
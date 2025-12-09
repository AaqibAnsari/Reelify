import os
import re
import time
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup


class TrendScraper:
    """
    Render-safe scraper:
    - Uses requests + BeautifulSoup (no Selenium/Chrome)
    - Optional in-memory cache to reduce requests
    """

    BASE_URLS = {
        "pakistan": "https://trends24.in/pakistan/",
        "united-states": "https://trends24.in/united-states/",
    }

    def __init__(self, cache_ttl_seconds: int = 900):
        # In-memory cache (Render filesystem can reset; this is safer for demo)
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}

        # Use a realistic UA (helps against basic blocking)
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _normalize_location(self, location: str) -> str:
        loc = (location or "").strip().lower()
        return loc if loc in self.BASE_URLS else "pakistan"

    def _is_cached_fresh(self, location: str) -> bool:
        if location not in self._cache:
            return False
        ts, _ = self._cache[location]
        return (time.time() - ts) < self.cache_ttl_seconds

    def _fetch_html(self, url: str) -> str:
        # retry lightly for demo stability
        last_err = None
        for _ in range(3):
            try:
                resp = requests.get(url, headers=self.headers, timeout=20)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                last_err = e
                time.sleep(1.2)
        raise RuntimeError(f"Failed to fetch trends page: {last_err}")

    def scrape_trends(self, location: str) -> List[Dict]:
        """
        Returns list of:
          {rank, name, count, numeric_count}
        """
        loc = self._normalize_location(location)

        # Cache
        if self._is_cached_fresh(loc):
            return self._cache[loc][1]

        url = self.BASE_URLS[loc]
        html_content = self._fetch_html(url)

        soup = BeautifulSoup(html_content, "html.parser")
        items = soup.select(".trend-card__list li")

        trends = []
        for i, li in enumerate(items[:20]):
            a = li.select_one("a.trend-link")
            if not a:
                continue

            name = a.get_text(strip=True)
            span = li.select_one("span")
            raw_count = span.get_text(strip=True) if span else ""

            numeric_count = 0
            if raw_count:
                m = re.search(r"(\d+)", raw_count.replace(",", ""))
                if m:
                    numeric_count = int(m.group(1))
                    raw_count = f"{numeric_count}K"

            trends.append(
                {
                    "rank": i + 1,
                    "name": name,
                    "count": raw_count,
                    "numeric_count": numeric_count,
                }
            )

        # Cache result
        self._cache[loc] = (time.time(), trends)
        return trends

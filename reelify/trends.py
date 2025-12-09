# reelify/trends.py
import re
import time
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup


class TrendScraper:
    """
    Render-safe scraper:
    - Uses requests + BeautifulSoup (no Selenium/Chrome)
    - In-memory cache (Render FS can be ephemeral)
    - Forces UTF-8 decode + normalizes text to avoid mojibake/garbled Unicode
    """

    BASE_URLS = {
        "pakistan": "https://trends24.in/pakistan/",
        "united-states": "https://trends24.in/united-states/",
    }

    def __init__(self, cache_ttl_seconds: int = 900):
        self.cache_ttl_seconds = int(cache_ttl_seconds)
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ur;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _normalize_location(self, location: str) -> str:
        loc = (location or "").strip().lower()
        return loc if loc in self.BASE_URLS else "pakistan"

    def _is_cached_fresh(self, location: str) -> bool:
        if location not in self._cache:
            return False
        ts, _ = self._cache[location]
        return (time.time() - ts) < self.cache_ttl_seconds

    @staticmethod
    def _clean_text(s: str) -> str:
        """
        Normalize whitespace and fix common mojibake cases that can happen if UTF-8
        bytes were decoded as latin-1/cp1252 somewhere upstream.
        """
        if not s:
            return ""
        s = s.replace("\u00a0", " ")              # NBSP -> space
        s = re.sub(r"\s+", " ", s).strip()

        # Heuristic fix: if it looks like mojibake, try latin1->utf8 roundtrip.
        # (Safe: if it fails, we keep original.)
        try:
            # If string contains lots of Â/Ã/Ø/Ù artifacts, it's often mojibake.
            if any(ch in s for ch in ("Ã", "Â", "Ø", "Ù", "Ð", "Ñ")):
                s2 = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
                s2 = re.sub(r"\s+", " ", s2).strip()
                if s2:
                    s = s2
        except Exception:
            pass

        return s

    def _fetch_html(self, url: str) -> str:
        last_err = None
        for _ in range(3):
            try:
                resp = requests.get(url, headers=self.headers, timeout=20)
                resp.raise_for_status()

                # Force UTF-8 to avoid garbled characters on some hosts
                resp.encoding = "utf-8"
                return resp.text
            except Exception as e:
                last_err = e
                time.sleep(1.2)
        raise RuntimeError(f"Failed to fetch trends page: {last_err}")

    def scrape_trends(self, location: str) -> List[Dict]:
        """
        Returns list of dicts:
          {rank, name, count, numeric_count}
        """
        loc = self._normalize_location(location)

        # Cache
        if self._is_cached_fresh(loc):
            return self._cache[loc][1]

        url = self.BASE_URLS[loc]
        html = self._fetch_html(url)

        soup = BeautifulSoup(html, "html.parser")
        items = soup.select(".trend-card__list li")

        trends: List[Dict] = []
        for i, li in enumerate(items[:20]):
            a = li.select_one("a.trend-link")
            if not a:
                continue

            name = self._clean_text(a.get_text(strip=True))

            span = li.select_one("span")
            raw_count = self._clean_text(span.get_text(strip=True)) if span else ""

            numeric_count = 0
            count_display = raw_count

            # Trends24 often shows like "35K Tweets" etc; keep just N + "K"
            if raw_count:
                m = re.search(r"(\d[\d,]*)", raw_count)
                if m:
                    numeric_count = int(m.group(1).replace(",", ""))
                    count_display = f"{numeric_count}K"

            trends.append(
                {
                    "rank": i + 1,
                    "name": name,
                    "count": count_display,
                    "numeric_count": numeric_count,
                }
            )

        # Cache result
        self._cache[loc] = (time.time(), trends)
        return trends

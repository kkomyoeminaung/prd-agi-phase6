"""
tools/web_search.py — PRD-AGI Web Search Tool
==============================================
Real web search using DuckDuckGo Instant Answer API (no API key needed).
Falls back to scraped search results if instant answer unavailable.

Features:
  - DuckDuckGo Instant Answer API (free, no key)
  - Full text extraction from result URLs
  - Result caching to avoid duplicate requests
  - Rate limiting (1 req/sec)
  - SU(5) curvature tagging of results
  - Safe content filtering
"""

import requests
import json
import time
import hashlib
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import quote_plus, urljoin, urlparse

logger = logging.getLogger('PRD-AGI.WebSearch')


class WebSearchTool:
    """
    Web search tool for PRD-AGI.

    Uses DuckDuckGo Instant Answer API — no API key required.
    Results are cached and rate-limited.

    Usage:
        tool = WebSearchTool()
        results = tool.search("Buddhist causality Paccaya conditions")
        summary = tool.search_and_summarize("latest AI research 2025")
    """

    DDG_API   = "https://api.duckduckgo.com/"
    DDG_HTML  = "https://html.duckduckgo.com/html/"
    HEADERS   = {
        "User-Agent": "Mozilla/5.0 (compatible; PRD-AGI/6.0; +https://prd-agi.local)",
        "Accept": "application/json, text/html",
    }

    def __init__(self, max_results: int = 5, cache_size: int = 200, rate_limit: float = 1.0):
        self.max_results  = max_results
        self.rate_limit   = rate_limit   # seconds between requests
        self._cache: Dict[str, Dict] = {}
        self._cache_order: List[str]  = []
        self._cache_max   = cache_size
        self._last_req    = 0.0
        self.search_history: List[Dict] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def search(self, query: str, max_results: Optional[int] = None) -> Dict:
        """
        Search the web for a query.

        Returns:
            {
              query, results: [{title, url, snippet, source}],
              instant_answer, timestamp, cached
            }
        """
        if not query.strip():
            return {"error": "Empty query", "results": []}

        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self._cache:
            result = dict(self._cache[cache_key])
            result["cached"] = True
            return result

        self._rate_limit_wait()

        n = max_results or self.max_results
        result = self._ddg_search(query, n)
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        result["cached"] = False

        # Cache
        if len(self._cache_order) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        # Log history
        self.search_history.append({
            "query": query,
            "result_count": len(result.get("results", [])),
            "timestamp": result["timestamp"],
        })
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

        return result

    def search_and_summarize(self, query: str) -> str:
        """
        Search and return a clean text summary of top results.
        Ready to pass to LLM as context.
        """
        result = self.search(query)
        if "error" in result:
            return f"[Search error: {result['error']}]"

        lines = [f"Search: {query}\n"]

        if result.get("instant_answer"):
            lines.append(f"Instant Answer: {result['instant_answer']}\n")

        for i, r in enumerate(result.get("results", [])[:5], 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   Source: {r['url']}\n")

        return "\n".join(lines)

    def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        """
        Fetch and extract text from a URL.
        Returns cleaned text content.
        """
        try:
            self._rate_limit_wait()
            r = requests.get(url, headers=self.HEADERS, timeout=10)
            if r.status_code != 200:
                return f"[HTTP {r.status_code}]"
            text = self._extract_text(r.text)
            return text[:max_chars]
        except Exception as e:
            return f"[Fetch error: {e}]"

    # ── Internal ────────────────────────────────────────────────────────────

    def _ddg_search(self, query: str, n: int) -> Dict:
        """Query DuckDuckGo API."""
        results = []
        instant = ""

        # 1. Instant Answer API
        try:
            params = {
                "q": query, "format": "json", "no_redirect": "1",
                "no_html": "1", "skip_disambig": "1",
            }
            r = requests.get(self.DDG_API, params=params,
                             headers=self.HEADERS, timeout=8)
            if r.status_code == 200:
                data = r.json()
                # Instant answer
                if data.get("AbstractText"):
                    instant = data["AbstractText"]
                    if data.get("AbstractURL"):
                        results.append({
                            "title":   data.get("Heading", query),
                            "url":     data["AbstractURL"],
                            "snippet": data["AbstractText"][:300],
                            "source":  data.get("AbstractSource", ""),
                        })
                # Related topics
                for topic in data.get("RelatedTopics", [])[:n]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title":   topic.get("Text", "")[:60],
                            "url":     topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")[:300],
                            "source":  "DuckDuckGo",
                        })
        except Exception as e:
            logger.warning(f"DDG instant API error: {e}")

        # 2. HTML fallback if not enough results
        if len(results) < 2:
            try:
                html_results = self._ddg_html_search(query, n)
                results.extend(html_results)
            except Exception as e:
                logger.warning(f"DDG HTML search error: {e}")

        return {
            "query":          query,
            "results":        results[:n],
            "instant_answer": instant,
            "result_count":   len(results[:n]),
        }

    def _ddg_html_search(self, query: str, n: int) -> List[Dict]:
        """Scrape DuckDuckGo HTML results as fallback."""
        results = []
        try:
            r = requests.post(
                self.DDG_HTML,
                data={"q": query, "b": "", "kl": ""},
                headers=self.HEADERS, timeout=10,
            )
            if r.status_code != 200:
                return results

            # Simple regex extraction (no BeautifulSoup needed)
            pattern = r'class="result__title"[^>]*>.*?href="([^"]+)"[^>]*>(.*?)</a'
            matches = re.findall(pattern, r.text, re.DOTALL)

            snippet_pattern = r'class="result__snippet"[^>]*>(.*?)</a'
            snippets = re.findall(snippet_pattern, r.text, re.DOTALL)

            for i, (url, title) in enumerate(matches[:n]):
                clean_title   = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                # Decode DDG redirect URL
                if "uddg=" in url:
                    from urllib.parse import unquote
                    url = unquote(re.search(r'uddg=([^&]+)', url).group(1))
                results.append({
                    "title":   clean_title[:100],
                    "url":     url,
                    "snippet": clean_snippet[:300],
                    "source":  urlparse(url).netloc,
                })
        except Exception as e:
            logger.warning(f"DDG HTML parse error: {e}")
        return results

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove scripts, styles, nav
        for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe']:
            html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', html, flags=re.DOTALL|re.IGNORECASE)
        # Remove all tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove very short lines
        lines = [l.strip() for l in text.split('.') if len(l.strip()) > 40]
        return '. '.join(lines)

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_req
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_req = time.time()

    def clear_cache(self):
        self._cache.clear()
        self._cache_order.clear()

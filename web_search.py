from typing import Final, List, Dict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


class WebSearch:
    MAX_RESULTS: Final[int] = 3
    REGION: Final[str] = "ua-uk"
    SAFESEARCH: Final[str] = "moderate"
    REQUEST_TIMEOUT: Final[int] = 10
    MAX_PAGE_CHARS: Final[int] = 2500
    USER_AGENT: Final[str] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    UNWANTED_TAGS: Final[List[str]] = [
        "script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe"
    ]

    def search(self, query: str) -> List[Dict[str, str]]:
        if not query or not query.strip():
            return []

        results: List[Dict[str, str]] = []
        try:
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    region=self.REGION,
                    safesearch=self.SAFESEARCH,
                    max_results=self.MAX_RESULTS,
                )
                if not raw_results:
                    return []

                for item in raw_results:
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                        }
                    )
        except Exception:
            return []

        return results

    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except Exception:
            return False

    def read_page(self, url: str) -> str:
        if not url or not self._is_valid_url(url):
            return ""

        try:
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(self.UNWANTED_TAGS):
                tag.decompose()

            clean_text = " ".join(soup.get_text(separator=" ").split())
            return clean_text[: self.MAX_PAGE_CHARS]

        except requests.RequestException:
            return ""
        except Exception:
            return ""


if __name__ == "__main__":
    searcher = WebSearch()
    test_query = "Python мова програмування"
    print(f"Пошук за запитом: '{test_query}'...\n")

    search_results = searcher.search(test_query)
    if not search_results:
        print("Пошук не повернув жодного результату.")
    else:
        for idx, res in enumerate(search_results, start=1):
            print(f"{idx}. {res['title']} ({res['url']})")
            print(f"   Опис: {res['snippet']}\n")

        first_url = search_results["url"]
        print(f"Завантаження вмісту сторінки: {first_url}\n")
        page_content = searcher.read_page(first_url)

        if page_content:
            print("Отриманий текст сторінки:")
            print("-" * 50)
            print(page_content)
        else:
            print("Не вдалося отримати текст HTML-сторінки.")
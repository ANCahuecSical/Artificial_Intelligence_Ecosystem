"""Simple scraper that fetches a page and extracts text from its <main> tag.

Requirements: requests, beautifulsoup4

Saves extracted text to `Selected_Document.txt` (UTF-8) and returns it.
"""
from typing import Optional
import re
import requests
from bs4 import BeautifulSoup

OUTPUT_FILE = "Selected_Document.txt"


def fetch_and_extract(url: str) -> str:
    """Fetch and extract text from the page.

    - If the URL is a Wikipedia page, use the MediaWiki API (explaintext) for a clean article extract.
    - Otherwise, fall back to HTML extraction (previous behavior).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Detect Wikimedia/Wikipedia pages and use the API
    try:
        from urllib.parse import urlparse, unquote
    except Exception:
        urlparse = None
        unquote = None

    is_wikipedia = False
    page_title = None
    if urlparse:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if "wikipedia.org" in host or "wikimedia.org" in host:
            is_wikipedia = True
            # URL form: /wiki/Title
            path = parsed.path or ""
            if path.startswith("/wiki/"):
                page_title = unquote(path[len("/wiki/"):])

    if is_wikipedia and page_title:
        api = f"https://{parsed.netloc}/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": page_title,
            "format": "json",
            "redirects": "1",
        }
        try:
            r = requests.get(api, headers=headers, params=params, timeout=15)
        except requests.RequestException as exc:
            print(f"API request failed: {exc}")
            # Fall back to HTML extraction below
            r = None

        if r is not None and r.status_code == 200:
            try:
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                # Pages is a dict keyed by pageid
                if pages:
                    first = next(iter(pages.values()))
                    extract = first.get("extract", "")
                    if extract:
                        # Write and return
                        try:
                            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                                f.write(extract)
                        except OSError as exc:
                            print(f"Failed to write to {OUTPUT_FILE}: {exc}")
                            return extract
                        print(f"Wrote API extract to {OUTPUT_FILE} (via MediaWiki API)")
                        return extract
            except ValueError:
                print("Failed to parse API JSON response; falling back to HTML extraction.")
        else:
            print(f"API request failed or returned HTTP {getattr(r, 'status_code', 'N/A')} - falling back to HTML extraction.")

    # Fallback: original HTML extraction
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return ""

    if resp.status_code != 200:
        print(f"Failed: HTTP {resp.status_code} while fetching {url}")
        return ""

    print(f"Success: HTTP {resp.status_code} fetched {url} (HTML fallback)")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Prefer Wikipedia's article container when available to avoid navigation/headers
    container = soup.find("div", id="mw-content-text")
    if container is None:
        main_tag = soup.find("main")
        if main_tag is None:
            print("Warning: <main> tag not found in the document.")
            body_tag = soup.find("body")
            if body_tag is None:
                return ""
            container = body_tag
        else:
            container = main_tag

    # Extract heading/paragraph/list-like tags in document order and stop before References/External links
    parts = []
    stop_headings = {"references", "external links", "notes", "see also", "further reading"}

    # Consider these block-level tags to preserve reading order
    interesting_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "li", "blockquote", "pre", "figcaption", "caption"
    ]

    for tag in container.find_all(interesting_tags, recursive=True):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        # Filter out navigation-like short strings
        low = text.lower()
        if "table of contents" in low or "jump to" in low or "edit" == low:
            continue
        # Stop if we encounter a References/External links heading
        if tag.name.startswith("h"):
            heading_text = re.sub(r"\[.*?\]", "", text).strip().lower()
            if heading_text in stop_headings:
                break
        # Avoid consecutive duplicates
        if parts and parts[-1] == text:
            continue
        parts.append(text)

    extracted = "\n\n".join(parts)

    # Write to file as UTF-8
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(extracted)
    except OSError as exc:
        print(f"Failed to write to {OUTPUT_FILE}: {exc}")
        return extracted

    print(f"Wrote extracted content to {OUTPUT_FILE} (HTML fallback)")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a URL")
    parser.add_argument("url", help="URL to extract text from")
    args = parser.parse_args()

    extracted = fetch_and_extract(args.url)
    if extracted:
        print("Extraction completed (non-empty result).")
    else:
        print("Extraction completed but returned no content.")


if __name__ == "__main__":
    main()

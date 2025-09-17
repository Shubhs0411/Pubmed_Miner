# extractor.py
from __future__ import annotations

import re
import json
import time
from typing import Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"
HEADERS = {
    "User-Agent": "PMID-FullText/1.0 (+mailto:you@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# -------------------------
# Robust HTTP session
# -------------------------
_SESSION = requests.Session()
_RETRY = Retry(
    total=6,
    backoff_factor=0.6,                  # 0.6, 1.2, 2.4, ...
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False,
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

def _clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# >>> CHANGED: normalize troublesome unicode so quotes match PAPER_TEXT
_UNICODE_NORMALIZATION_MAP = {
    # quotes
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
    # dashes
    "\u2013": "-", "\u2014": "-", "\u2212": "-",  # en/em/real minus
    # spaces
    "\u00A0": " ", "\u2009": " ", "\u202F": " ", "\u200A": " ", "\u200B": "",
    # arrows
    "\u2192": "->", "\u27F6": "->",
}
_UNICODE_TRANS = str.maketrans(_UNICODE_NORMALIZATION_MAP)

def _normalize_text_for_ie(text: str) -> str:
    """
    Keep scientific tokens (Δ, Greek letters) intact, but normalize punctuation,
    spaces, and arrows so the LLM's short quotes are more likely to be substrings.
    """
    if not text:
        return text
    t = text.translate(_UNICODE_TRANS)
    # collapse excessive whitespace once more after replacements
    return _clean_space(t)

def _make_soup(html_text: str):
    """
    Build a BeautifulSoup object trying robust parsers in order:
    lxml -> html5lib -> html.parser (stdlib).
    Avoids hard dependency errors: 'Couldn't find a tree builder... lxml'.
    """
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(html_text, parser)
        except Exception:
            continue
    # last resort
    return BeautifulSoup(html_text, "html.parser")

def _safe_get_json(url: str, params: dict, *, tries: int = 3, sleep_s: float = 0.7) -> dict:
    """
    GET JSON with retries + tolerant parsing to handle:
      - control characters / stray bytes
      - truncated/chunked responses
      - 429/5xx with backoff
    """
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = _SESSION.get(url, params=params, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "application/json"}, timeout=60)
            if r.status_code >= 400:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        time.sleep(float(ra))
                    except Exception:
                        pass
                r.raise_for_status()
            txt = (r.text or "").replace("\x00", "")
            return json.loads(txt, strict=False)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * attempt)
            continue
    raise last_err

# -------------------------
# Core: PMID -> PMCID
# -------------------------
def pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Return PMCID for a PMID if it exists (free full text in PMC), else None."""
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "json",
        "tool": "pmid_fulltext_tool",
        "email": "you@example.com",
    }
    # Optional API key via env: NCBI_API_KEY (if set by caller’s environment)
    import os
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key

    data = _safe_get_json(f"{EUTILS}/elink.fcgi", params=params)
    # Typical structures:
    # data["linksets"][0]["linksetdbs"][0]["links"] -> ["1234567", ...]  # then PMCID = "PMC" + id
    try:
        linksets = data.get("linksets", []) or []
        if not linksets:
            return None
        dbs = (linksets[0].get("linksetdbs") or [])
        if not dbs:
            return None
        links = dbs[0].get("links") or []
        if links:
            return f"PMC{links[0]}"
    except Exception:
        pass
    return None

# -------------------------
# Helpers for HTML pruning
# -------------------------

# >>> CHANGED: labels we should remove to avoid background/noise
_SECTION_KILLWORDS = {
    "references", "reference", "footnotes", "acknowledgements", "acknowledgments",
    "author contributions", "authors' information", "funding", "ethics",
    "competing interests", "conflict of interest", "supplementary", "supporting information",
}

def _has_killword(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(kw in t for kw in _SECTION_KILLWORDS)

def _prune_noise(main) -> None:
    """
    Remove obvious chrome and back matter (refs, footnotes, acknowledgements, etc.).
    """
    # Remove generic chrome
    for sel in ["nav", "header", "footer", "aside", "script", "style",
                ".ncbi-alerts", ".page-header", ".page-navbar"]:
        for node in main.select(sel):
            try:
                node.decompose()
            except Exception:
                pass

    # Remove sections by heading text (e.g., <h2>References</h2> and following siblings if grouped)
    for h in main.find_all(["h2", "h3", "h4"]):
        try:
            if _has_killword(h.get_text(" ", strip=True)):
                # If the heading parent is a section-ish container, drop that container; else drop heading and next siblings until next heading.
                parent = h.find_parent(lambda tag: tag.name in {"section", "div", "article"} and len(tag.find_all(["h2","h3"], recursive=False)) <= 1)
                if parent:
                    parent.decompose()
                else:
                    h.decompose()
        except Exception:
            pass

    # Remove elements by id/class heuristics
    for el in list(main.find_all(True)):
        try:
            idtxt = " ".join([el.get("id",""), *el.get("class", [])]).lower()
            if any(kw in idtxt for kw in ["ref-list", "references", "footnote", "acknowledg", "supplement"]):
                el.decompose()
        except Exception:
            pass

# -------------------------
# Parse PMC HTML to text
# -------------------------
def _parse_main_and_title(html: str) -> Tuple[str, Optional[str]]:
    soup = _make_soup(html)

    # Title
    title = None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = _clean_space(og["content"])
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = _clean_space(h1.get_text(" ", strip=True))

    # Main content area (PMC uses #maincontent but fallback to <main> or whole doc)
    main = soup.find(id="maincontent") or soup.find("main") or soup

    # >>> CHANGED: aggressive pruning of noise/back matter
    _prune_noise(main)

    # Prefer body-like blocks that carry scientific content
    parts = []
    for tag in main.find_all(["h1","h2","h3","h4","p","li","td","th","caption"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)

    text = _clean_space(" ".join(parts))

    # >>> CHANGED: normalize unicode so the LLM’s quotes match this text
    text = _normalize_text_for_ie(text)

    return text, title

def fetch_pmc_html_text_and_title(pmcid: str, retries: int = 3) -> Tuple[str, Optional[str]]:
    """
    Fetch PMC article HTML and parse to (text, title).
    Retries and cools down on 403 (rate-limit / access throttle).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = _SESSION.get(f"{PMC_ARTICLE_URL}/{pmcid}/", headers=HEADERS, timeout=60)
            # 403 handling: throttle & retry
            if r.status_code == 403:
                # polite cool-down (grows with attempts)
                time.sleep(4.0 * attempt)
                last_err = requests.HTTPError(f"403 Forbidden for {pmcid}")
                continue
            r.raise_for_status()
            return _parse_main_and_title(r.text)
        except Exception as e:
            last_err = e
            # backoff for other transient errors
            time.sleep(0.6 * attempt)
            continue
    # Give up
    raise last_err if last_err else RuntimeError(f"Failed to fetch PMC article for {pmcid}")

# -------------------------
# Public API (unchanged signatures)
# -------------------------
def get_pmc_fulltext_with_meta(pmid: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Given a PMID, return (PMCID, full_text, title). PMCID may be None if not found.
    """
    pmcid = pmid_to_pmcid(pmid)
    if not pmcid:
        return None, "", None
    text, title = fetch_pmc_html_text_and_title(pmcid)
    return pmcid, text, title

def get_pmc_fulltext(pmid: str) -> Tuple[Optional[str], str]:
    """
    Back-compat wrapper: return (PMCID, full_text). Title omitted here.
    """
    pmcid, text, _title = get_pmc_fulltext_with_meta(pmid)
    return pmcid, text


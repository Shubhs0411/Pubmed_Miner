
import re
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"
HEADERS = {"User-Agent": "PMID-FullText/1.0 (+mailto:you@example.com)"}

def _clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

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
    r = requests.get(f"{EUTILS}/elink.fcgi", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    try:
        links = data["linksets"][0]["linksetdbs"][0]["links"]
        if links:
            return f"PMC{links[0]}"
    except Exception:
        pass
    return None

def _parse_main_and_title(html: str) -> Tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = None
    # Try meta og:title first
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = _clean_space(og["content"])
    # Fallback to first h1
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = _clean_space(h1.get_text(" ", strip=True))
    # Main content extraction
    main = soup.find(id="maincontent") or soup.find("main") or soup
    for sel in ["nav", "header", "footer", "aside", "script", "style", ".ncbi-alerts", ".page-header", ".page-navbar"]:
        for node in main.select(sel):
            node.decompose()
    parts = []
    for tag in main.find_all(["h1","h2","h3","h4","p","li","td","th","caption"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = _clean_space(" ".join(parts))
    return text, title

def fetch_pmc_html_text_and_title(pmcid: str) -> Tuple[str, Optional[str]]:
    r = requests.get(f"{PMC_ARTICLE_URL}/{pmcid}/", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return _parse_main_and_title(r.text)

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

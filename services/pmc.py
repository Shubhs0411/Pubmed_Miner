from __future__ import annotations

import os
import re
import json
import time
from typing import Optional, Tuple, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")
HEADERS_HTML = {"User-Agent": f"SVF-PMC-Fetch/1.3 (+mailto:{CONTACT_EMAIL})"}

# Lower threshold - we want more JATS content even if shorter
MIN_JATS_BODY_CHARS = int(os.getenv("PMC_MIN_JATS_CHARS", "800"))

_LAST_SOURCE: Dict[str, str] = {}


def get_last_fetch_source(key: str) -> Optional[str]:
    return _LAST_SOURCE.get(str(key))


def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


_SESS = _make_session()


def _get_json(url: str, params: dict, tries: int = 3, sleep_s: float = 0.6) -> dict:
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params = dict(params)
        params["api_key"] = api_key
        
    last_exc = None
    for i in range(tries):
        try:
            r = _SESS.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(sleep_s * (2 ** i))
                continue
            r.raise_for_status()
            
            try:
                return r.json()
            except Exception:
                txt = r.text
                start, end = txt.find("{"), txt.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(txt[start:end + 1])
                raise
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s * (i + 1))
            
    raise last_exc or RuntimeError("EUtils request failed")


def pmid_to_pmcid(pmid: str) -> Optional[str]:
    j = _get_json(f"{EUTILS}/elink.fcgi", {
        "dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json",
        "tool": "svf_pmc_mapper", "email": CONTACT_EMAIL
    })
    try:
        linksets = j["linksets"][0].get("linksetdbs", [])
        for ls in linksets:
            if ls.get("dbto") == "pmc":
                links = ls.get("links") or []
                if links:
                    return "PMC" + str(links[0])
    except Exception:
        pass
    return None


def pmcid_to_pmid(pmcid: str) -> Optional[str]:
    pmcid_num = re.sub(r"^PMC", "", str(pmcid))
    j = _get_json(f"{EUTILS}/elink.fcgi", {
        "dbfrom": "pmc", "db": "pubmed", "id": pmcid_num, "retmode": "json",
        "tool": "svf_pmc_mapper", "email": CONTACT_EMAIL
    })
    try:
        linksets = j["linksets"][0].get("linksetdbs", [])
        for ls in linksets:
            if ls.get("dbto") == "pubmed":
                links = ls.get("links") or []
                if links:
                    return str(links[0])
    except Exception:
        pass
    return None


_EMBARGO_RE = re.compile(r"PMCID:\s*PMC\d+\s*\(available on\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\)", re.I)


def _pubmed_embargo_date_for_pmid(pmid: str) -> Optional[str]:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    r = _SESS.get(url, headers=HEADERS_HTML, timeout=30)
    if r.status_code == 200:
        m = _EMBARGO_RE.search(r.text)
        if m:
            return m.group(1)
    return None


def get_publisher_links(pmid: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "doi": None, 
        "publisher_url": None, 
        "publisher_free_url": None, 
        "is_publisher_oa": None
    }
    
    try:
        j = _get_json(f"{EUTILS}/esummary.fcgi", {
            "db": "pubmed", "id": pmid, "retmode": "json"
        })
        rec = j["result"][str(pmid)]
        for aid in rec.get("articleids", []):
            if aid.get("idtype") == "doi":
                out["doi"] = aid.get("value")
                break
    except Exception:
        pass
        
    try:
        html = _SESS.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", 
                        headers=HEADERS_HTML, timeout=30).text
        m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*(?:Full text|Publisher)\s*', 
                     html, re.I)
        if m:
            out["publisher_url"] = m.group(1)
    except Exception:
        pass
        
    email = os.getenv("UNPAYWALL_EMAIL")
    if out["doi"] and email:
        try:
            u = f"https://api.unpaywall.org/v2/{out['doi']}"
            r = _SESS.get(u, params={"email": email}, timeout=30)
            if r.status_code == 200:
                upw = r.json()
                out["is_publisher_oa"] = bool(upw.get("is_oa"))
                best = upw.get("best_oa_location") or {}
                if best.get("url"):
                    out["publisher_free_url"] = best["url"]
        except Exception:
            pass
            
    return out


def fetch_pmc_jats_xml(pmcid: str, *, tries: int = 3, sleep_s: float = 0.8) -> str:
    pmc_num = re.sub(r"^PMC", "", str(pmcid), flags=re.I)
    url = f"{EUTILS}/efetch.fcgi"
    params = {"db": "pmc", "id": pmc_num, "retmode": "xml"}
    
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
        
    last_exc = None
    for i in range(tries):
        try:
            r = _SESS.get(url, params=params, timeout=60)
            txt = (r.text or "").lstrip()
            if r.status_code == 200 and txt.startswith("<") and "</" in txt:
                return txt
            if r.status_code in (429, 502, 503, 504):
                time.sleep(sleep_s * (i + 1))
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s * (i + 1))
            
    raise last_exc or RuntimeError("pmc_efetch_jats_failed")


def _jats_to_text_and_title(xml_str: str) -> Tuple[str, Optional[str]]:
    """
    Extract comprehensive text from JATS XML.
    Captures: abstract, body (all sections), methods, results, discussion.
    Priority: get as much relevant text as possible for mutation mining.
    """
    try:
        soup = BeautifulSoup(xml_str, "lxml-xml")
    except Exception:
        soup = BeautifulSoup(xml_str, "xml")
    
    # Extract title
    title_el = (soup.find("article-title")
                or (soup.find("title-group").find("article-title") 
                    if soup.find("title-group") else None)
                or soup.find("title"))
    title = title_el.get_text(" ", strip=True) if title_el else None
    
    parts: List[str] = []
    
    # Abstract(s) - critical for mutation summaries
    for ab in soup.find_all("abstract"):
        t = ab.get_text(" ", strip=True)
        if t:
            parts.append(t)
    
    # Body - main content
    body = soup.find("body")
    if body:
        # Section titles
        for st in body.find_all("title"):
            t = st.get_text(" ", strip=True)
            if t:
                parts.append(t)
        
        # Paragraphs - main text content
        for p in body.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t:
                parts.append(t)
        
        # Lists (methods often in lists)
        for li in body.find_all("li"):
            t = li.get_text(" ", strip=True)
            if t:
                parts.append(t)
        
        # Tables (mutations often documented in tables)
        for table in body.find_all("table"):
            t = table.get_text(" ", strip=True)
            if t:
                parts.append(t)
    
    # Methods section (specific mutations often detailed here)
    methods = soup.find("sec", {"sec-type": "methods"})
    if methods:
        t = methods.get_text(" ", strip=True)
        if t:
            parts.append(t)
    
    # Results section (mutation effects reported here)
    results = soup.find("sec", {"sec-type": "results"})
    if results:
        t = results.get_text(" ", strip=True)
        if t:
            parts.append(t)
    
    # Discussion (mutation interpretation)
    discussion = soup.find("sec", {"sec-type": "discussion"})
    if discussion:
        t = discussion.get_text(" ", strip=True)
        if t:
            parts.append(t)
    
    # Materials and methods (another common location)
    mat_methods = soup.find("sec", {"sec-type": "materials|methods"})
    if mat_methods:
        t = mat_methods.get_text(" ", strip=True)
        if t:
            parts.append(t)
    
    # Figure captions (mutations sometimes only in figures)
    for fig in soup.find_all("fig"):
        caption = fig.find("caption")
        if caption:
            t = caption.get_text(" ", strip=True)
            if t:
                parts.append(t)
    
    # Combine and normalize whitespace
    text = " ".join(" ".join(parts).split())
    
    return text, title


def _html_to_text_and_title(html: str) -> Tuple[str, Optional[str]]:
    """Extract text from PMC HTML fallback."""
    soup = BeautifulSoup(html, "html.parser")
    
    t_el = soup.find(["h1", "title"])
    title = t_el.get_text(strip=True) if t_el else None
    
    article = soup.find("article") or soup.find(id="maincontent") or soup
    
    # Remove non-content elements
    for sel in ["nav", "header", "footer", "script", "style", 
                ".fig", ".figures", ".fig-popup", ".figure-viewer",
                ".tsec", ".table-wrap", ".sidebar", ".ref-list", ".references"]:
        for n in article.select(sel):
            n.decompose()
    
    text = " ".join(article.get_text(separator=" ", strip=True).split())
    
    return text, title


def fetch_pmc_html_text_and_title(pmcid: str, retries: int = 3) -> Tuple[str, Optional[str]]:
    """Fetch and parse PMC HTML."""
    url = f"{PMC_ARTICLE_URL}/{pmcid}/"
    last_status = None
    
    for i in range(retries):
        r = _SESS.get(url, headers=HEADERS_HTML, timeout=45)
        last_status = r.status_code
        
        if r.status_code == 200:
            return _html_to_text_and_title(r.text)
        if r.status_code == 403:
            raise RuntimeError("pmc_embargo_or_blocked")
        if r.status_code in (429, 502, 503, 504):
            time.sleep(0.7 * (2 ** i))
            continue
        r.raise_for_status()
        
    raise RuntimeError(f"pmc_fetch_failed status={last_status}")


def get_pmc_fulltext_with_meta(pmid: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Fetch PMC full text with priority: JATS > HTML fallback.
    Strategy:
    1. Always try JATS first (most structured, best for mutation mining)
    2. If JATS is too short or fails, try HTML as fallback
    3. Use whichever provides more comprehensive text
    
    Returns: (pmcid, text, title)
    """
    pmcid = pmid_to_pmcid(pmid)
    if not pmcid:
        _LAST_SOURCE[pmid] = "none"
        return None, "", None
    
    # Verify PMCID maps back to correct PMID
    back = pmcid_to_pmid(pmcid)
    if back and str(back) != str(pmid):
        pass  # Continue anyway, but noted
    
    jats_text: Optional[Tuple[str, Optional[str]]] = None
    jats_success = False
    
    # Try JATS first (priority)
    try:
        jats_xml = fetch_pmc_jats_xml(pmcid)
        if jats_xml:
            j_text, j_title = _jats_to_text_and_title(jats_xml)
            if j_text and len(j_text) >= MIN_JATS_BODY_CHARS:
                jats_text = (j_text, j_title)
                jats_success = True
    except Exception:
        jats_text = None
    
    # If JATS succeeded with good content, use it
    if jats_success and jats_text:
        _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "jats"
        return pmcid, jats_text[0], jats_text[1]
    
    # JATS failed or too short - try HTML fallback
    html_text: Optional[Tuple[str, Optional[str]]] = None
    try:
        html_text, html_title = fetch_pmc_html_text_and_title(pmcid)
    except RuntimeError as e:
        if "pmc_embargo_or_blocked" in str(e):
            _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "none"
            return pmcid, "", None
        html_text = None
    
    # If we have HTML and it's better than JATS, use HTML
    if html_text:
        if jats_text:
            # Compare lengths - use whichever is more complete
            if len(html_text[0]) > len(jats_text[0]) * 1.3:
                _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "html"
                return pmcid, html_text[0], html_text[1]
            else:
                # JATS is comparable, prefer it (better structure)
                _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "jats"
                return pmcid, jats_text[0], jats_text[1]
        else:
            # Only HTML available
            _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "html"
            return pmcid, html_text[0], html_text[1]
    
    # Last resort: use whatever JATS we got, even if short
    if jats_text:
        _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "jats"
        return pmcid, jats_text[0], jats_text[1]
    
    # Nothing worked
    _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "none"
    return pmcid, "", None


def get_pmc_fulltext(pmid: str) -> Tuple[Optional[str], str]:
    """Legacy wrapper - returns (pmcid, text) without title."""
    pmcid, text, _ = get_pmc_fulltext_with_meta(pmid)
    return pmcid, text


def get_free_publisher_fallback(pmid: str) -> Dict[str, Any]:
    """Get publisher info and embargo status."""
    info = get_publisher_links(pmid)
    info["embargo_until"] = _pubmed_embargo_date_for_pmid(pmid)
    return info


__all__ = [
    "get_pmc_fulltext_with_meta",
    "get_pmc_fulltext",
    "get_free_publisher_fallback",
    "get_last_fetch_source",
]
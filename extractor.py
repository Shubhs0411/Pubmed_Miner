# extractor.py
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

# Identify yourself to NCBI (recommended)
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")
HEADERS_HTML = {
    "User-Agent": f"SVF-PMC-Fetch/1.1 (+mailto:{CONTACT_EMAIL})"
}

# ---------- Session with retries ----------
def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
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
            r = _SESS.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(sleep_s * (2 ** i))
                continue
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                # tolerate stray chars around JSON
                txt = r.text
                start, end = txt.find("{"), txt.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(txt[start:end + 1])
                raise
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s * (i + 1))
    raise last_exc or RuntimeError("EUtils request failed")

# ---------- PMID <-> PMCID mapping ----------
def pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Exact mapping via ELink pubmed->pmc. Returns 'PMCxxxx' or None."""
    j = _get_json(f"{EUTILS}/elink.fcgi", {
        "dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json", "tool": "svf_pmc_mapper", "email": CONTACT_EMAIL
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
    """Reverse mapping to validate that a PMCID belongs to the PMID we think it does."""
    pmcid_num = re.sub(r"^PMC", "", str(pmcid))
    j = _get_json(f"{EUTILS}/elink.fcgi", {
        "dbfrom": "pmc", "db": "pubmed", "id": pmcid_num, "retmode": "json", "tool": "svf_pmc_mapper", "email": CONTACT_EMAIL
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

# ---------- PubMed HTML helpers (embargo & publisher links) ----------
_EMBARGO_RE = re.compile(r"PMCID:\s*PMC\d+\s*\(available on\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\)", re.I)

def _pubmed_embargo_date_for_pmid(pmid: str) -> Optional[str]:
    """Find 'PMCID: PMCxxxx (available on YYYY-MM-DD)' on the PubMed HTML page."""
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    r = _SESS.get(url, headers=HEADERS_HTML, timeout=30)
    if r.status_code == 200:
        m = _EMBARGO_RE.search(r.text)
        if m:
            return m.group(1)
    return None

def get_publisher_links(pmid: str) -> Dict[str, Any]:
    """
    Returns DOI, publisher_url, and if configured with UNPAYWALL_EMAIL, publisher_free_url + is_publisher_oa.
    """
    out: Dict[str, Any] = {"doi": None, "publisher_url": None, "publisher_free_url": None, "is_publisher_oa": None}
    # ESummary for DOI
    try:
        j = _get_json(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": pmid, "retmode": "json"})
        rec = j["result"][str(pmid)]
        for aid in rec.get("articleids", []):
            if aid.get("idtype") == "doi":
                out["doi"] = aid.get("value")
                break
    except Exception:
        pass
    # PubMed HTML to get "Full text" publisher link if present
    try:
        html = _SESS.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", headers=HEADERS_HTML, timeout=30).text
        m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*(?:Full text|Publisher)\s*', html, re.I)
        if m:
            out["publisher_url"] = m.group(1)
    except Exception:
        pass

    # Optional: Unpaywall
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

# ---------- PMC fetch & HTML->text ----------
def _html_to_text_and_title(html: str) -> Tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title = None
    t_el = soup.find(["h1", "title"])
    if t_el:
        title = t_el.get_text(strip=True)

    # Prefer the article content region on PMC pages
    article = soup.find("article")
    main = article or soup.find(id="maincontent") or soup

    # Remove nav/figures/scripts/references sidebars
    for sel in [
        "nav", "header", "footer", "script", "style",
        ".fig", ".figures", ".fig-popup", ".figure-viewer",
        ".tsec", ".table-wrap", ".sidebar", ".ref-list", ".references"
    ]:
        for n in main.select(sel):
            n.decompose()

    text = " ".join(main.get_text(separator=" ", strip=True).split())
    return text, title

def fetch_pmc_html_text_and_title(pmcid: str, retries: int = 3) -> Tuple[str, Optional[str]]:
    """
    Fetch PMC HTML for an article ID like 'PMC5925603' and return (plain_text, title).
    Raises RuntimeError('pmc_embargo_or_blocked') on 403, or RuntimeError with status on other failures.
    """
    url = f"{PMC_ARTICLE_URL}/{pmcid}/"
    last_status = None
    for i in range(retries):
        r = _SESS.get(url, headers=HEADERS_HTML, timeout=45)
        last_status = r.status_code
        if r.status_code == 200:
            return _html_to_text_and_title(r.text)
        if r.status_code == 403:
            # Don’t retry forever; likely embargo/robots block
            raise RuntimeError("pmc_embargo_or_blocked")
        if r.status_code in (429, 502, 503, 504):
            time.sleep(0.7 * (2 ** i))
            continue
        r.raise_for_status()
    raise RuntimeError(f"pmc_fetch_failed status={last_status}")

# ---------- Public: fetch with mapping/embargo/OA metadata ----------
def get_pmc_fulltext_with_meta(pmid: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns (pmcid, text, title). If no PMCID, text='' and title=None.
    - Validates pmcid<->pmid to avoid wrong-article 403s.
    - If PMC blocks with embargo (403), returns empty text but keeps pmcid so you can show embargo info via `_pubmed_embargo_date_for_pmid`.
    """
    pmcid = pmid_to_pmcid(pmid)
    if not pmcid:
        # No PMC – nothing to fetch here.
        return None, "", None

    # sanity: ensure pmcid maps back to this pmid (avoid mismatched PMCID)
    back = pmcid_to_pmid(pmcid)
    if back and str(back) != str(pmid):
        # Wrong PMCID for this PMID – treat as not in PMC for safety
        return None, "", None

    try:
        text, title = fetch_pmc_html_text_and_title(pmcid)
        return pmcid, text, title
    except RuntimeError as e:
        if "pmc_embargo_or_blocked" in str(e):
            # PMC exists but access blocked (often embargo). Surface empty text; callers can show embargo date using helper.
            return pmcid, "", None
        raise

def get_pmc_fulltext(pmid: str) -> Tuple[Optional[str], str]:
    """
    Back-compat wrapper: return (PMCID, full_text). Title omitted.
    """
    pmcid, text, _title = get_pmc_fulltext_with_meta(pmid)
    return pmcid, text

# ---------- Optional helper for your UI (publisher fallback when no PMC) ----------
def get_free_publisher_fallback(pmid: str) -> Dict[str, Any]:
    """
    If an article is not in PMC, use PubMed + Unpaywall to report publisher access:
      { 'doi', 'publisher_url', 'publisher_free_url', 'is_publisher_oa', 'embargo_until' }
    """
    info = get_publisher_links(pmid)
    info["embargo_until"] = _pubmed_embargo_date_for_pmid(pmid)
    return info

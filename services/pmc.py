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
HEADERS_HTML = {"User-Agent": f"SVF-PMC-Fetch/1.2 (+mailto:{CONTACT_EMAIL})"}

MIN_JATS_BODY_CHARS = int(os.getenv("PMC_MIN_JATS_CHARS", "1200"))

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
        params = dict(params); params["api_key"] = api_key
    last_exc = None
    for i in range(tries):
        try:
            r = _SESS.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(sleep_s * (2 ** i)); continue
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
            last_exc = e; time.sleep(sleep_s * (i + 1))
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
        if m: return m.group(1)
    return None

def get_publisher_links(pmid: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"doi": None, "publisher_url": None, "publisher_free_url": None, "is_publisher_oa": None}
    try:
        j = _get_json(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": pmid, "retmode": "json"})
        rec = j["result"][str(pmid)]
        for aid in rec.get("articleids", []):
            if aid.get("idtype") == "doi":
                out["doi"] = aid.get("value"); break
    except Exception:
        pass
    try:
        html = _SESS.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", headers=HEADERS_HTML, timeout=30).text
        m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*(?:Full text|Publisher)\s*', html, re.I)
        if m: out["publisher_url"] = m.group(1)
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
                if best.get("url"): out["publisher_free_url"] = best["url"]
        except Exception:
            pass
    return out

def fetch_pmc_jats_xml(pmcid: str, *, tries: int = 3, sleep_s: float = 0.8) -> str:
    pmc_num = re.sub(r"^PMC", "", str(pmcid), flags=re.I)
    url = f"{EUTILS}/efetch.fcgi"
    params = {"db": "pmc", "id": pmc_num, "retmode": "xml"}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key: params["api_key"] = api_key
    last_exc = None
    for i in range(tries):
        try:
            r = _SESS.get(url, params=params, timeout=60)
            txt = (r.text or "").lstrip()
            if r.status_code == 200 and txt.startswith("<") and "</" in txt:
                return txt
            if r.status_code in (429, 502, 503, 504):
                time.sleep(sleep_s * (i + 1)); continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e; time.sleep(sleep_s * (i + 1))
    raise last_exc or RuntimeError("pmc_efetch_jats_failed")

def _jats_to_text_and_title(xml_str: str) -> Tuple[str, Optional[str]]:
    try:
        soup = BeautifulSoup(xml_str, "lxml-xml")
    except Exception:
        soup = BeautifulSoup(xml_str, "xml")
    title_el = (soup.find("article-title")
                or (soup.find("title-group").find("article-title") if soup.find("title-group") else None)
                or soup.find("title"))
    title = title_el.get_text(" ", strip=True) if title_el else None
    parts: List[str] = []
    for ab in soup.find_all("abstract"):
        t = ab.get_text(" ", strip=True)
        if t: parts.append(t)
    body = soup.find("body")
    if body:
        for st in body.find_all("title"):
            t = st.get_text(" ", strip=True)
            if t: parts.append(t)
        for p in body.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t: parts.append(t)
        for li in body.find_all("li"):
            t = li.get_text(" ", strip=True)
            if t: parts.append(t)
    text = " ".join(" ".join(parts).split())
    return text, title

def _html_to_text_and_title(html: str) -> Tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    t_el = soup.find(["h1", "title"])
    title = t_el.get_text(strip=True) if t_el else None
    article = soup.find("article") or soup.find(id="maincontent") or soup
    for sel in ["nav","header","footer","script","style",".fig",".figures",".fig-popup",".figure-viewer",
                ".tsec",".table-wrap",".sidebar",".ref-list",".references"]:
        for n in article.select(sel): n.decompose()
    text = " ".join(article.get_text(separator=" ", strip=True).split())
    return text, title

def fetch_pmc_html_text_and_title(pmcid: str, retries: int = 3) -> Tuple[str, Optional[str]]:
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
            time.sleep(0.7 * (2 ** i)); continue
        r.raise_for_status()
    raise RuntimeError(f"pmc_fetch_failed status={last_status}")

def get_pmc_fulltext_with_meta(pmid: str) -> Tuple[Optional[str], str, Optional[str]]:
    pmcid = pmid_to_pmcid(pmid)
    if not pmcid:
        _LAST_SOURCE[pmid] = "none"
        return None, "", None
    back = pmcid_to_pmid(pmcid)
    if back and str(back) != str(pmid):
        pass
    jats_text: Optional[Tuple[str, Optional[str]]] = None
    try:
        jats_xml = fetch_pmc_jats_xml(pmcid)
        if jats_xml:
            tmp_text, tmp_title = _jats_to_text_and_title(jats_xml)
            if tmp_text and len(tmp_text) > 200:
                jats_text = (tmp_text, tmp_title)
    except Exception:
        jats_text = None

    if jats_text:
        j_text, j_title = jats_text
        html_text = None
        html_title = None
        try_html = (len(j_text) < max(200, MIN_JATS_BODY_CHARS))
        if not try_html:
            # Still consider HTML if it can provide substantially more content.
            try_html = True
        if try_html:
            try:
                html_text, html_title = fetch_pmc_html_text_and_title(pmcid)
            except RuntimeError:
                html_text = None
                html_title = None

        if html_text and len(html_text) > len(j_text) + 400:
            _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "html"
            return pmcid, html_text, html_title

        _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "jats"
        return pmcid, j_text, j_title

    try:
        text, title = fetch_pmc_html_text_and_title(pmcid)
        _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "html"
        return pmcid, text, title
    except RuntimeError as e:
        if "pmc_embargo_or_blocked" in str(e):
            _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "none"
            return pmcid, "", None
        _LAST_SOURCE[pmid] = _LAST_SOURCE[pmcid] = "none"
        raise

def get_pmc_fulltext(pmid: str) -> Tuple[Optional[str], str]:
    pmcid, text, _ = get_pmc_fulltext_with_meta(pmid)
    return pmcid, text

def get_free_publisher_fallback(pmid: str) -> Dict[str, Any]:
    info = get_publisher_links(pmid)
    info["embargo_until"] = _pubmed_embargo_date_for_pmid(pmid)
    return info

__all__ = [
    "get_pmc_fulltext_with_meta",
    "get_pmc_fulltext",
    "get_free_publisher_fallback",
    "get_last_fetch_source",
]

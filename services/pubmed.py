# services/pubmed.py
from __future__ import annotations

import os
import re
import time
import json
import calendar
from typing import Dict, List, Optional, Tuple
from datetime import date

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# -------------------- Robust HTTP session + tolerant JSON --------------------
_SESSION = requests.Session()
_RETRY = Retry(
    total=6,
    backoff_factor=0.6,  # 0.6, 1.2, 2.4, ...
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
    raise_on_status=False,
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

_DEFAULT_HEADERS = {
    "User-Agent": "pmid_fulltext_tool/1.0 (contact: you@example.com)",
    "Accept": "application/json",
}


def _safe_get_json(url: str, params: dict, *, tries: int = 3, sleep_s: float = 0.7) -> dict:
    """
    GET JSON with retries and tolerant parsing to avoid transient issues like:
      - 'Invalid control character' (stray bytes)
      - 'Response ended prematurely' (truncated responses)
    """
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = _SESSION.get(url, params=params, headers=_DEFAULT_HEADERS, timeout=60)
            if r.status_code >= 400:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        time.sleep(float(ra))
                    except Exception:
                        pass
                r.raise_for_status()

            txt = r.text.replace("\x00", "")  # strip stray NULs just in case
            return json.loads(txt, strict=False)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * attempt)  # simple backoff
            continue
    raise last_err


# -------------------- PubDate parsing & date utilities --------------------
_MON = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}


def parse_pubdate_interval(pubdate: str) -> Tuple[Optional[date], Optional[date]]:
    """
    Turn PubMed 'pubdate' strings like '2019', '2019 Nov', '2021 May 28'
    into an (inclusive) date interval: (start_date, end_date).
    Returns (None, None) if it can't parse.
    """
    if not pubdate or not isinstance(pubdate, str):
        return (None, None)

    s = pubdate.strip()

    # YYYY
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return (date(y, 1, 1), date(y, 12, 31))

    # YYYY Mon
    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]{3,})", s)
    if m:
        y = int(m.group(1))
        mon_str = m.group(2)[:3].lower()
        if mon_str in _MON:
            mm = _MON[mon_str]
            last_day = calendar.monthrange(y, mm)[1]
            return (date(y, mm, 1), date(y, mm, last_day))

    # YYYY Mon DD
    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]{3,})\s+(\d{1,2})", s)
    if m:
        y = int(m.group(1))
        mon_str = m.group(2)[:3].lower()
        d = int(m.group(3))
        if mon_str in _MON:
            mm = _MON[mon_str]
            try:
                dt = date(y, mm, d)
                return (dt, dt)
            except ValueError:
                return (None, None)

    # Extend here for other shapes like '2019 Winter' if needed
    return (None, None)


def overlaps(a: Tuple[Optional[date], Optional[date]],
             b: Tuple[Optional[date], Optional[date]]) -> bool:
    """Inclusive overlap for possibly None endpoints."""
    a0, a1 = a
    b0, b1 = b
    if not a0 or not a1 or not b0 or not b1:
        return False
    return not (a1 < b0 or b1 < a0)


def to_pdat(d: Optional[date]) -> Optional[str]:
    """Convert date object to PubMed pdat string 'YYYY/MM' (required by NCBI API)."""
    if not d:
        return None
    return f"{d.year:04d}/{d.month:02d}"


# -------------------- E-utilities wrappers --------------------
def _ncbi_params(extra: Optional[Dict] = None) -> Dict:
    params = {
        "retmode": "json",
        "tool": "pmid_fulltext_tool",
        "email": "you@example.com",
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    if extra:
        params.update(extra)
    return params


def as_review_query(base_query: str, open_access_only: bool = False) -> str:
    """
    Force query to only return 'Review' articles.
    If open_access_only=True, also require PMC Open Access to reduce 403/blocked hits.
    """
    q = f"({base_query}) AND (review[pt] OR review[Publication Type])"
    if open_access_only:
        q = f"({q}) AND (pmc open access[filter])"
    return q


def esearch_reviews(query: str, *,
                    mindate: Optional[str],
                    maxdate: Optional[str],
                    sort: str = "relevance",
                    retmax: int = 200,
                    cap: int = 2000,
                    open_access_only: bool = False) -> List[str]:
    """
    Search PubMed for review PMIDs matching the query + date bounds.
    Pages through results up to 'cap'.
    If open_access_only=True, adds PMC Open Access filter to the query.
    
    Args:
        query: Search query string
        mindate: Minimum date in YYYY/MM format (e.g., "2020/01")
        maxdate: Maximum date in YYYY/MM format (e.g., "2020/12")
        sort: Sort order ("relevance" or "pub+date")
        retmax: Results per page
        cap: Maximum total results
        open_access_only: If True, restrict to open access papers
    
    Note: Query is normalized to lowercase to avoid PubMed parser issues with uppercase + hyphens.
    """
    # Normalize query to lowercase to avoid PubMed parser issues with uppercase + hyphens
    normalized_query = query.lower()
    q = as_review_query(normalized_query, open_access_only=open_access_only)
    
    ids: List[str] = {}
    ids = []
    retstart = 0
    while True:
        j = _safe_get_json(
            f"{EUTILS}/esearch.fcgi",
            _ncbi_params({
                "db": "pubmed",
                "term": q,
                "retstart": str(retstart),
                "retmax": str(retmax),
                "sort": sort,
                **({"datetype": "pdat"} if (mindate or maxdate) else {}),
                **({"mindate": mindate} if mindate else {}),
                **({"maxdate": maxdate} if maxdate else {}),
            }),
        )
        page = j.get("esearchresult", {}).get("idlist", []) or []
        if not page:
            break
        ids.extend(page)
        retstart += len(page)
        if len(ids) >= cap:
            break
        time.sleep(0.34)  # be nice to NCBI
    return ids


def esearch_all(query: str, *,
                mindate: Optional[str],
                maxdate: Optional[str],
                sort: str = "relevance",
                retmax: int = 200,
                cap: int = 2000,
                open_access_only: bool = False) -> List[str]:
    """
    Search PubMed for all article types (not just reviews) matching the query + date bounds.
    Pages through results up to 'cap'.
    If open_access_only=True, adds PMC Open Access filter to the query.
    
    Args:
        query: Search query string
        mindate: Minimum date in YYYY/MM format (e.g., "2020/01")
        maxdate: Maximum date in YYYY/MM format (e.g., "2020/12")
        sort: Sort order ("relevance" or "pub+date")
        retmax: Results per page
        cap: Maximum total results
        open_access_only: If True, restrict to open access papers
    
    Note: Query is normalized to lowercase to avoid PubMed parser issues with uppercase + hyphens.
    """
    # Normalize query to lowercase to avoid PubMed parser issues with uppercase + hyphens
    normalized_query = query.lower()
    q = normalized_query
    if open_access_only:
        q = f"({q}) AND (pmc open access[filter])"
    
    ids: List[str] = []
    retstart = 0
    while True:
        j = _safe_get_json(
            f"{EUTILS}/esearch.fcgi",
            _ncbi_params({
                "db": "pubmed",
                "term": q,
                "retstart": str(retstart),
                "retmax": str(retmax),
                "sort": sort,
                **({"datetype": "pdat"} if (mindate or maxdate) else {}),
                **({"mindate": mindate} if mindate else {}),
                **({"maxdate": maxdate} if maxdate else {}),
            }),
        )
        page = j.get("esearchresult", {}).get("idlist", []) or []
        if not page:
            break
        ids.extend(page)
        retstart += len(page)
        if len(ids) >= cap:
            break
        time.sleep(0.34)  # be nice to NCBI
    return ids


def esummary(pmids: List[str]) -> Dict[str, Dict]:
    """
    Get concise metadata (title, source, pubdate) for a list of PMIDs.
    """
    out: Dict[str, Dict] = {}
    if not pmids:
        return out
    CHUNK = 200
    for i in range(0, len(pmids), CHUNK):
        chunk_ids = ",".join(pmids[i:i + CHUNK])
        data = _safe_get_json(
            f"{EUTILS}/esummary.fcgi",
            _ncbi_params({"db": "pubmed", "id": chunk_ids}),
        )
        uids = data.get("result", {}).get("uids", []) or []
        for uid in uids:
            it = data["result"].get(uid, {})
            out[uid] = {
                "title": it.get("title"),
                "source": it.get("source"),
                "pubdate": it.get("pubdate"),
            }
        time.sleep(0.25)
    return out

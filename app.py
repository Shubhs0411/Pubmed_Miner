import os
import time
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from extractor import get_pmc_fulltext_with_meta, get_pmc_fulltext
from llm_groq import run_on_paper, clean_and_ground
import requests  # for PubMed API callsfrom datetime import date, datetime
from datetime import date, datetime
import calendar
import re

# ---------- Robust PubDate parsing to intervals ----------
_MON = {m.lower(): i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1
)}

def _parse_pubdate_interval(pubdate: str) -> tuple[date | None, date | None]:
    """
    Turn a PubMed 'pubdate' string like:
      '2019', '2019 Nov', '2021 May 28'
    into a (start_date, end_date) interval.
    Returns (None, None) if it can't parse.
    """
    if not pubdate or not isinstance(pubdate, str):
        return (None, None)

    s = pubdate.strip()
    # YYYY
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return (date(y,1,1), date(y,12,31))

    # YYYY Mon
    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]{3,})", s)
    if m:
        y = int(m.group(1))
        mon_str = m.group(2)[:3].lower()
        if mon_str in _MON:
            mnum = _MON[mon_str]
            last_day = calendar.monthrange(y, mnum)[1]
            return (date(y,mnum,1), date(y,mnum,last_day))

    # YYYY Mon DD
    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]{3,})\s+(\d{1,2})", s)
    if m:
        y = int(m.group(1))
        mon_str = m.group(2)[:3].lower()
        d = int(m.group(3))
        if mon_str in _MON:
            mnum = _MON[mon_str]
            try:
                dt = date(y, mnum, d)
                return (dt, dt)
            except ValueError:
                return (None, None)

    return (None, None)



def _overlaps(a_start: date | None, a_end: date | None, b_start: date | None, b_end: date | None) -> bool:
    """Inclusive overlap check for intervals that may have None endpoints."""
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return not (a_end < b_start or b_end < a_start)


def _to_pdat_string(d: date | None) -> str | None:
    """Convert a date object to PubMed pdat 'YYYY/MM/DD' string."""
    if not d:
        return None
    return f"{d.year:04d}/{d.month:02d}/{d.day:02d}"


# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="PMID → Full Text (PMC) + LLM Extraction", layout="wide")
st.title("📄 PubMed Full Text Extractor (PMC) + 🧠 Groq LLM Miner")
st.caption("Enter a PMID to fetch PMC full text, and optionally mine sequence features with Groq. Or switch to the Review Paper Search tab to find review articles with a PubMed-style query builder.")

# ------------------------------
# Sidebar Controls
# ------------------------------
with st.sidebar:
    st.header("Settings")
    st.info("Extraction uses PubMed E-Utilities and PMC. Some PMIDs may not have a PMCID (no free full text).")
    st.caption("Tip: After extraction, you can mine sequence features with Groq and download results.")
    st.divider()

    st.subheader("Groq Filters")
    virus_filter = st.text_input("Virus filter (optional)", value="Dengue virus", help="E.g., 'Dengue virus'")
    protein_filter = st.text_input("Protein filter (optional)", value="protein", help="E.g., 'NS2A' or leave 'protein'")

    st.divider()
    st.subheader("Extraction Mode")
    exhaustive = st.checkbox("Exhaustive mode (slower, more complete)", value=True, help="Runs multiple passes and merges results; adds small delays to avoid rate limits")

    with st.expander("Advanced", expanded=False):
        chunk_chars = st.slider("Max characters per chunk", min_value=8000, max_value=24000, value=16000, step=1000)
        overlap_chars = st.slider("Overlap per chunk", min_value=200, max_value=1500, value=500, step=50)
        delay_ms = st.slider("Delay between chunk calls (ms)", min_value=0, max_value=1500, value=400, step=50)
        min_conf = st.slider("Minimum confidence to keep", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        require_mut_quote = st.checkbox(
            "Require mutation token in a quote",
            value=True,
            help="Stricter grounding; keeps only entries whose quotes include the mutation text (e.g., D125A)."
        )

# ==============================
# Tabs
# ==============================
tab_pmids, tab_reviews = st.tabs(["PMID → Full Text / LLM", "Review Paper Search"])

# ==========================================================
# TAB 1: Existing flow (PMID → PMC full text → LLM extract)
# ==========================================================
with tab_pmids:
    # ------------------------------
    # PMID Input
    # ------------------------------
    pmid = st.text_input("PMID", value="", placeholder="e.g., 25392211", help="Paper must have a free full text in PMC.")
    col_go, col_reset = st.columns([1, 1])
    with col_go:
        go = st.button("Fetch Full Text", type="primary")
    with col_reset:
        if st.button("Reset"):
            for k in ["pmid", "pmcid", "fulltext", "paper_title", "extract_result"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

    # ------------------------------
    # Fetch Full Text
    # ------------------------------
    if go:
        if not pmid.strip().isdigit():
            st.error("Please enter a numeric PMID.")
        else:
            with st.spinner("Looking up PMCID and fetching full text…"):
                try:
                    pmcid, text, title = get_pmc_fulltext_with_meta(pmid.strip())
                except Exception:
                    # fallback for compatibility
                    try:
                        pmcid, text = get_pmc_fulltext(pmid.strip())
                        title = None
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.stop()

            if pmcid is None:
                st.error("No PMCID found — this PMID likely has no free full text on PMC.")
            elif not text:
                st.warning(f"PMCID {pmcid} found, but no parsable text was extracted.")
            else:
                st.success(f"Fetched {pmcid} • Characters: {len(text):,}")
                st.session_state["pmid"] = pmid
                st.session_state["pmcid"] = pmcid
                st.session_state["fulltext"] = text
                st.session_state["paper_title"] = title

    # ------------------------------
    # Display Full Text & Download
    # ------------------------------
    if "fulltext" in st.session_state and st.session_state.get("fulltext"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PMID", st.session_state.get("pmid", ""))
        with c2:
            st.metric("PMCID", st.session_state.get("pmcid", ""))
        with c3:
            st.metric("Characters", len(st.session_state["fulltext"]))

        if st.session_state.get("paper_title"):
            st.markdown(f"**Title:** {st.session_state['paper_title']}")

        with st.expander("Show extracted full text (click to toggle)", expanded=False):
            st.text_area("Full Text", st.session_state["fulltext"], height=400)

        st.download_button(
            label="⬇️ Download full text (.txt)",
            data=st.session_state["fulltext"].encode("utf-8"),
            file_name=f"{st.session_state.get('pmid','PMID')}_{st.session_state.get('pmcid','PMCID')}_fulltext.txt",
            mime="text/plain"
        )

    st.divider()

    # ------------------------------
    # LLM Extraction
    # ------------------------------
    st.subheader("🧠 LLM (Groq) — Extract Sequence Features")

    if "fulltext" in st.session_state and st.session_state.get("fulltext"):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            do_llm = st.button("🔎 Extract with Groq", type="secondary")
        with col_b:
            clear_llm = st.button("Clear results")

        if clear_llm and "extract_result" in st.session_state:
            del st.session_state["extract_result"]

        if do_llm:
            if not os.getenv("GROQ_API_KEY"):
                st.warning("GROQ_API_KEY not set in environment. Put it in your .env file.")
            with st.spinner("Analyzing paper with Groq…"):
                try:
                    result = run_on_paper(
                        st.session_state["fulltext"],
                        meta={
                            "pmid": st.session_state.get("pmid"),
                            "pmcid": st.session_state.get("pmcid"),
                            "virus_filter": virus_filter or None,
                            "protein_filter": protein_filter or None,
                            "exhaustive": exhaustive,
                            "chunk_chars": chunk_chars,
                            "overlap_chars": overlap_chars,
                            "delay_ms": delay_ms,
                        }
                    )
                    # Sanitize & ground to the paper text
                    result = clean_and_ground(
                        result,
                        st.session_state["fulltext"],
                        restrict_to_paper=True,
                        require_mutation_in_quote=require_mut_quote,
                        min_confidence=min_conf
                    )
                    # Inject title if we have it
                    if "paper" in result:
                        result["paper"]["title"] = st.session_state.get("paper_title")
                    st.session_state["extract_result"] = result
                except Exception as e:
                    st.error(f"Groq extraction failed: {e}")

    # ------------------------------
    # Results Display
    # ------------------------------
    if "extract_result" in st.session_state and st.session_state["extract_result"]:
        result = st.session_state["extract_result"]
        st.markdown("#### JSON Output")
        st.json(result)

        feats = result.get("sequence_features", []) if isinstance(result, dict) else []
        if feats:
            rows = []
            for f in feats:
                rows.append({
                    "virus": f.get("virus"),
                    "source_strain": f.get("source_strain"),
                    "protein": f.get("protein"),
                    "mutation": f.get("mutation"),
                    "position": f.get("position"),
                    "effect_category": f.get("effect_category"),
                    "confidence": f.get("confidence"),
                    "effect_summary": f.get("effect_summary"),
                })
            df = pd.DataFrame(rows)

            st.markdown("#### Sequence Features (summary table)")
            st.dataframe(df, use_container_width=True)

            # CSV Download
            csv_name = f"{st.session_state.get('pmid','PMID')}_features.csv"
            st.download_button(
                label="⬇️ Download features (.csv)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=csv_name,
                mime="text/csv"
            )

            # JSON Download
            st.download_button(
                label="⬇️ Download features (.json)",
                data=json.dumps(result, ensure_ascii=True, indent=2).encode("utf-8"),
                file_name=f"{st.session_state.get('pmid','PMID')}_features.json",
                mime="application/json"
            )
        else:
            st.info("No sequence features found or extraction returned an empty list.")
    else:
        st.caption("Tip: Fetch the full text first, then click **Extract with Groq** to mine sequence features.")

# =================================================================
# TAB 2: Review Paper Search with PubMed-like Query Builder
# =================================================================
with tab_reviews:
    st.subheader("🔎 PubMed Review Paper Search (Advanced Builder)")
    st.caption("Build a PubMed query step-by-step (like the Advanced Search Builder). We automatically restrict to review articles.")

    # --- PubMed E-utilities helpers ---
    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _ncbi_params(extra=None):
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

    def _as_review_query(base_query: str) -> str:
        # Constrain to review articles
        return f"({base_query}) AND (review[pt] OR review[Publication Type])"

    def pubmed_search_reviews(query: str, *, sort="relevance", mindate=None, maxdate=None,
                              retmax=200, max_records=1000):
        """Return all review PMIDs for the query (paged)."""
        q = _as_review_query(query)
        r = requests.get(
            f"{EUTILS}/esearch.fcgi",
            params=_ncbi_params({
                "db": "pubmed",
                "term": q,
                "retstart": "0",
                "retmax": str(retmax),
                "sort": sort,
                **({"datetype": "pdat"} if (mindate or maxdate) else {}),
                **({"mindate": mindate} if mindate else {}),
                **({"maxdate": maxdate} if maxdate else {}),
            }),
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        total = int(j.get("esearchresult", {}).get("count", "0"))
        ids = list(j.get("esearchresult", {}).get("idlist", []) or [])
        target_total = min(total, max_records) if max_records is not None else total

        while len(ids) < target_total:
            retstart = len(ids)
            r2 = requests.get(
                f"{EUTILS}/esearch.fcgi",
                params=_ncbi_params({
                    "db": "pubmed",
                    "term": q,
                    "retstart": str(retstart),
                    "retmax": str(retmax),
                    "sort": sort,
                    **({"datetype": "pdat"} if (mindate or maxdate) else {}),
                    **({"mindate": mindate} if mindate else {}),
                    **({"maxdate": maxdate} if maxdate else {}),
                }),
                timeout=60,
            )
            r2.raise_for_status()
            page = r2.json().get("esearchresult", {}).get("idlist", []) or []
            if not page:
                break
            ids.extend(page)
            time.sleep(0.34)  # be nice to NCBI
        if max_records is not None and len(ids) > max_records:
            ids = ids[:max_records]
        return {
            "query": query,
            "query_with_review_filter": q,
            "count": total,
            "returned": len(ids),
            "pmids": ids,
            "sort": sort,
            "mindate": mindate,
            "maxdate": maxdate,
        }

    def pubmed_summaries(pmids):
        """ESummary for titles/journals/dates to show a nice list."""
        if not pmids:
            return {}
        out = {}
        CHUNK = 200
        for i in range(0, len(pmids), CHUNK):
            chunk_ids = ",".join(pmids[i:i+CHUNK])
            r = requests.get(
                f"{EUTILS}/esummary.fcgi",
                params=_ncbi_params({"db": "pubmed", "id": chunk_ids}),
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
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

    # ------------------------------
    # Query Builder UI (PubMed-like)
    # ------------------------------
    if "query_parts" not in st.session_state:
        st.session_state["query_parts"] = []

    st.markdown("**Add terms to the query box**")
    field_options = {
        "All Fields": "All Fields",
        "Title": "Title",
        "Abstract": "Abstract",
        "Author": "Author",
        "Journal": "Journal",
        "MeSH Terms": "MeSH Terms",
        "Text Word": "Text Word",
    }
    field = st.selectbox("Field", list(field_options.keys()), index=0)
    keyword = st.text_input("Enter a search term")
    operator = st.selectbox("Operator to prepend", ["AND", "OR", "NOT"], index=0, help="Used when appending to existing query")

    c_add, c_clear_last, c_clear_all = st.columns([1, 1, 1])
    with c_add:
        if st.button("ADD"):
            if keyword.strip():
                # Build clause like "term"[Field]
                field_tag = field_options[field]
                if field_tag == "All Fields":
                    clause = f"\"{keyword}\"[All Fields]"
                else:
                    clause = f"\"{keyword}\"[{field_tag}]"
                # If query already has parts, prepend operator (e.g., AND "term"[Field])
                if st.session_state["query_parts"]:
                    clause = f"{operator} {clause}"
                st.session_state["query_parts"].append(clause)
    with c_clear_last:
        if st.button("Remove last"):
            if st.session_state["query_parts"]:
                st.session_state["query_parts"].pop()
    with c_clear_all:
        if st.button("Clear all"):
            st.session_state["query_parts"] = []

    # Query Box (editable)
    built_query = " ".join(st.session_state["query_parts"])
    st.markdown("**Query box**")
    full_query = st.text_area("Enter / edit your search query here", value=built_query, height=100)

    # Search options
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
with col2:
    # Use a proper date range picker (start, end)
    date_range = st.date_input(
        "Publication date range (inclusive)",
        value=(date(2010,1,1), date.today()),
        help="Filters both at PubMed (pdat) and locally using robust parsing of 'PubDate'."
    )
    # Ensure tuple of (start, end)
    if isinstance(date_range, tuple):
        user_start, user_end = date_range
    else:
        user_start, user_end = (date_range, date_range)

limit = st.slider(
    "Max records to fetch",
    min_value=50,
    max_value=5000,
    value=500,
    step=50,
    help="Cap total results for display so the UI stays responsive."
)

# ------------------------------
# Run Search (reviews only)
# ------------------------------
if st.button("Search review papers"):
    query_to_use = (full_query or "").strip()
    if not query_to_use:
        st.warning("Build a query first.")
    else:
        # Send PubMed the pdat min/max to reduce server-side results
        mindate_str = _to_pdat_string(user_start)
        maxdate_str = _to_pdat_string(user_end)

        with st.spinner("Searching PubMed for review articles…"):
            info = pubmed_search_reviews(
                query=query_to_use,
                sort=sort,
                mindate=mindate_str,
                maxdate=maxdate_str,
                retmax=200,
                max_records=limit,
            )
            pmids = info.get("pmids", [])
            summaries = pubmed_summaries(pmids)

        # Build a DataFrame of results
        rows = []
        for pid in pmids:
            meta = summaries.get(pid, {})
            pubdate_raw = meta.get("pubdate") or ""
            s, e = _parse_pubdate_interval(pubdate_raw)
            # Local, robust interval filter (inclusive overlap)
            if _overlaps(s, e, user_start, user_end):
                rows.append({
                    "PMID": pid,
                    "Title": meta.get("title") or "",
                    "Journal": meta.get("source") or "",
                    "PubDate": pubdate_raw,
                    "PubMed Link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                })

        st.markdown("#### Search details")
        # Show what PubMed returned before local interval filter
        st.json({k: v for k, v in info.items() if k != "pmids"})

        if not rows:
            st.info("No review articles matched the selected date range after parsing PubDate.")
        else:
            st.markdown("#### Review articles (after robust date filtering)")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download list (.csv)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="pubmed_review_results.csv",
                mime="text/csv"
            )


    st.caption("Next step idea: add checkboxes to select PMIDs from the list and send them to the PMC + Groq pipeline.")

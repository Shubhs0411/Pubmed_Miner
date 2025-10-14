# app.py (simplified UI, fixed selection persistence + full-text view/download)
import os, json, time, io, zipfile
from datetime import date
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
# near your other imports in app.py
from services.pmc import get_last_fetch_source



from services.pubmed import (
    esearch_reviews, esummary, parse_pubdate_interval, overlaps, to_pdat
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts, flatten_to_rows

# ---------------- Helpers ----------------
def _persist(key, value):
    st.session_state[key] = value
    return value

def _bucketize_papers(papers_dict):
    fetched, no_pmc, errors = [], [], []
    for pmid, info in papers_dict.items():
        row = {
            "PMID": str(pmid),
            "PMCID": info.get("pmcid") or "",
            "Title": info.get("title") or "",
            "Status": info.get("status"),
            "Error": info.get("error") or "",
        }
        if info.get("status") == "ok":
            fetched.append(row)
        elif info.get("status") == "no_pmc_fulltext":
            no_pmc.append(row)
        else:
            errors.append(row)
    return fetched, no_pmc, errors

# ---------------- App setup ----------------
load_dotenv()
st.set_page_config(page_title="PubMed → PMC → LLM (Batch Miner)", layout="wide")
st.title("🧪 PubMed Review Miner")
st.caption("Search review articles, fetch PMC full text, run your LLM extractor, and download findings.")

# ---------------- Sidebar: minimal knobs, advanced collapsed ----------------
with st.sidebar:
    st.header("LLM Settings")  # always visible, no expander

    # Open (always-on) controls
    chunk_chars = st.slider("Max chars per chunk", 8000, 24000, 16000, 1000)
    overlap_chars = st.slider("Overlap per chunk", 200, 1500, 500, 50)
    delay_ms = st.slider("Delay between chunk calls (ms)", 0, 1500, 400, 50)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)

    # Permanently require the mutation token in the quote (no checkbox)
    REQUIRE_MUT_QUOTE = True

    st.divider()
    st.caption("Tip: set NCBI_API_KEY and GROQ_API_KEY (or GEMINI_API_KEY) in .env for reliability.")


# ---------------- Step 1: Query ----------------
st.subheader("1) Enter your PubMed query (reviews only)")
st.write("Paste a PubMed query (we’ll restrict to **Review** articles automatically).")
query = st.text_area("Query", height=100, placeholder='e.g., dengue[MeSH Terms] AND mutation[Text Word]')

# ---------------- Step 2: Date range (min 2005) + search ----------------
st.subheader("2) Choose publication date range & search")
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    # Enforce minimum selectable date of Jan 1, 2005
    default_range = (date(2005, 1, 1), date(2025, 12, 31))
    rng = st.date_input(
        "Inclusive range",
        value=default_range,
        min_value=date(2005, 1, 1),
        help="Earliest allowed start date is Jan 1, 2005."
    )
with colB:
    sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
with colC:
    cap = st.slider("Max records", 0, 500, 100, 100)

go = st.button("🔎 Search PubMed (reviews)")

if go:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        start_date, end_date = (rng if isinstance(rng, tuple) else (rng, rng))
        mindate = to_pdat(start_date)
        maxdate = to_pdat(end_date)

        with st.spinner("Searching PubMed (reviews)…"):
            pmids = esearch_reviews(query.strip(), mindate=mindate, maxdate=maxdate, sort=sort, cap=cap)
            sums = esummary(pmids)

        # Robust local date interval filter
        rows = []
        for pid in pmids:
            meta = sums.get(pid, {})
            pubdate_raw = meta.get("pubdate") or ""
            if overlaps(parse_pubdate_interval(pubdate_raw), (start_date, end_date)):
                rows.append({
                    "PMID": str(pid),
                    "Title": meta.get("title") or "",
                    "Journal": meta.get("source") or "",
                    "PubDate": pubdate_raw,
                    "PubMed Link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                })

        # Persist results for stable rendering outside this block
        df_hits = pd.DataFrame(rows)
        if df_hits.empty:
            # Clear old state if no results for this new search
            for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.info("No results in this date range.")
        else:
            df_hits["PMID"] = df_hits["PMID"].astype(str)
            _persist("hits_df", df_hits.to_dict("records"))
            _persist("hits_pmids", df_hits["PMID"].tolist())
            # On a new search, start with no selection
            st.session_state["selected_pmids"] = []
            st.success(f"Found {len(df_hits)} results. See 'Results' below to select papers.")

# --- Persistent Results & Selection (always visible after a search) ---
if st.session_state.get("hits_df"):
    st.markdown("#### Results")
    df_hits = pd.DataFrame(st.session_state["hits_df"])
    pmid_options = [str(x) for x in st.session_state.get("hits_pmids", [])]

    st.dataframe(df_hits, use_container_width=True)
    st.download_button(
        "⬇️ Download matches (.csv)",
        data=df_hits.to_csv(index=False).encode("utf-8"),
        file_name="pubmed_review_matches.csv",
        mime="text/csv"
    )

    st.markdown("##### Select PMIDs to process with LLM")
    # Initialize selection if missing
    if "selected_pmids" not in st.session_state:
        st.session_state["selected_pmids"] = []

    # Separate keys so toggling 'Select all' doesn't reset multiselect by accident
    select_all = st.checkbox("Select all", value=False, key="select_all_hits")
    if select_all:
        st.session_state["selected_pmids"] = pmid_options.copy()

    st.session_state["selected_pmids"] = st.multiselect(
        "Choose PMIDs (multi-select supported)",
        options=pmid_options,
        default=[p for p in st.session_state["selected_pmids"] if p in pmid_options],
        key="pmid_multiselect"
    )
    st.caption(f"Selected: {len(st.session_state['selected_pmids'])} of {len(pmid_options)}")

    if st.session_state["selected_pmids"]:
        st.download_button(
            "⬇️ Download selected PMIDs (.txt)",
            data=("\n".join(st.session_state["selected_pmids"]).encode("utf-8")),
            file_name="selected_pmids.txt",
            mime="text/plain",
        )

# ---------------- Step 3: One-click end-to-end (Fetch PMC → LLM) ----------------
st.subheader("3) Run extraction")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    # Explicit ALL override (default OFF)
    override_all = st.checkbox(
        "Process ALL results (ignore selection)",
        value=False,
        help="When checked, all results from Step 2 will be sent to the LLM."
    )
with colB:
    # Clear previous findings/logs to avoid confusion across runs (default ON)
    clear_previous = st.checkbox(
        "Clear previous results before run",
        value=True,
        help="Prevents older findings from appearing alongside this run."
    )
with colC:
    run_all = st.button(
        "🚀 Fetch PMC & Run LLM",
        disabled=(not override_all and len(st.session_state.get("selected_pmids", [])) == 0)
    )

# Reset button
if st.button("🔁 Reset"):
    for k in ["hits_df", "hits_pmids", "batch_papers", "batch_results", "llm_log", "selected_pmids", "select_all_hits", "pmid_multiselect"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

if run_all:
    # Decide EXACT set of PMIDs for THIS run (no silent fallback)
    hits = [str(x) for x in st.session_state.get("hits_pmids", [])]
    selected = [str(x) for x in st.session_state.get("selected_pmids", [])]
    pmids = hits if override_all else selected

    if not pmids:
        st.warning("No PMIDs selected. Pick at least one in Step 2 or check 'Process ALL results'.")
        st.stop()

    # Optional: clear prior findings/logs so outputs only reflect this run
    if clear_previous:
        st.session_state["batch_results"] = {}
        st.session_state["llm_log"] = []

    if not os.getenv("GROQ_API_KEY"):
        st.info("Note: GROQ_API_KEY isn’t set—set it in .env for best reliability.")

    # A) Fetch PMC only for THIS run's PMIDs
    with st.spinner(f"Fetching PMC full texts for {len(pmids)} PMIDs…"):
        papers = fetch_all_fulltexts(pmids, delay_ms=150)
        _persist("batch_papers", papers)  # persist only this run's set

    fetched, no_pmc, errors = _bucketize_papers(papers)
    n_ok, n_no, n_err = len(fetched), len(no_pmc), len(errors)
    st.success(f"PMC texts: ✅ {n_ok} fetched | ⚠️ {n_no} no PMC | ❌ {n_err} errors")

    if n_ok:
        with st.expander("Show fetched list", expanded=False):
            st.dataframe(pd.DataFrame(fetched), use_container_width=True, height=min(400, 40 + 28 * len(fetched)))
    if n_no:
        with st.expander("Show no-PMC list", expanded=False):
            st.dataframe(pd.DataFrame(no_pmc), use_container_width=True, height=min(400, 40 + 28 * len(no_pmc)))
    if n_err:
        with st.expander("Show fetch errors", expanded=False):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, height=min(400, 40 + 28 * len(errors)))

    if n_ok == 0:
        st.stop()

    # === NEW: Full-text viewer & downloads for this run (BEFORE LLM) ===
    ok_pmids_this_run = [row["PMID"] for row in fetched]  # fetched==ok only
    st.markdown("#### Full texts for this run (selected & fetched)")

    # Build in-memory zip of all full texts
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pmid in ok_pmids_this_run:
            info = papers.get(pmid, {})
            title = info.get("title") or ""
            pmcid = info.get("pmcid") or ""
            text = (
                info.get("text")
                or info.get("fulltext")
                or info.get("content")
                or ""
            )
            # Individual expander with per-paper download
            with st.expander(f"{pmid} — {title[:100]}"):
                if pmcid:
                    st.markdown(f"**PMCID:** {pmcid}  |  [Open PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/)")
                else:
                    st.markdown("**PMCID:** (not available)")

                # PROVENANCE (JATS vs HTML)
                source = (info.get("source") or get_last_fetch_source(pmid) or "unknown")
                badge = {"jats": "🟢 JATS (XML)", "html": "🔵 HTML", "none": "⚪ none"}.get(source, "⚪ unknown")
                st.markdown(f"**Source:** {badge}")

                if text:
                    st.text_area("Full text (preview)", value=text, height=300, key=f"ta_{pmid}")
                    st.download_button(
                        f"⬇️ Download full text ({pmid}).txt",
                        data=text.encode("utf-8"),
                        file_name=f"{pmid}_{(pmcid or 'NO_PMCID')}.txt",
                        mime="text/plain",
                        key=f"dl_txt_{pmid}"
                    )
                    # Optional: if HTML is present, offer it too
                    html = info.get("html") or info.get("raw_html")
                    if html:
                        st.download_button(
                            f"⬇️ Download HTML ({pmid}).html",
                            data=html.encode("utf-8"),
                            file_name=f"{pmid}_{(pmcid or 'NO_PMCID')}.html",
                            mime="text/html",
                            key=f"dl_html_{pmid}"
                        )
                else:
                    st.info("No full text captured for this paper.")

            # Add to zip (even if empty string; optional—skip empties if you prefer)
            safe_name = f"{pmid}_{(pmcid or 'NO_PMCID')}".replace("/", "_")
            zf.writestr(f"{safe_name}.txt", text if text else "")

    st.download_button(
        "⬇️ Download all selected full texts (.zip)",
        data=zip_buf.getvalue(),
        file_name="selected_full_texts.zip",
        mime="application/zip"
    )

    # ---------- B) Run LLM ONLY on PMIDs chosen for THIS run and with ok status ----------
    # Always re-derive from session state (survives Streamlit reruns)
    papers = st.session_state.get("batch_papers", {})
    ok_pmids_this_run = [pid for pid, info in papers.items() if info.get("status") == "ok"]

    if not ok_pmids_this_run:
        st.info("Nothing to analyze: no successfully fetched PMC full texts in this run.")
        st.stop()

    llm_log = st.session_state.get("llm_log", [])
    batch_results = st.session_state.get("batch_results", {})

    prog = st.progress(0, text="Starting LLM…")
    log_box = st.empty()

    # Single findings header + one reusable table container (no duplicate tables)
    st.markdown("#### Findings")
    table_box = st.empty()

    total = len(ok_pmids_this_run)
    for i, pmid in enumerate(ok_pmids_this_run, start=1):
        title = papers[pmid].get("title") or ""
        pmcid = papers[pmid].get("pmcid") or ""
        log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) — {title[:80]}"
        llm_log.append(log_line); _persist("llm_log", llm_log)
        log_box.code("\n".join(llm_log[-20:]), language="text")

        try:
            single_dict = analyze_texts(
                {pmid: papers[pmid]},
                chunk_chars=chunk_chars, overlap_chars=overlap_chars,
                delay_ms=delay_ms, min_confidence=min_conf, require_mut_quote=True,
            )

            batch_results.update(single_dict); _persist("batch_results", batch_results)

            # live (partial) view uses the SAME table container
            out_df_partial = flatten_to_rows(batch_results)
            table_box.dataframe(out_df_partial, use_container_width=True)
        except Exception as e:
            err_line = f"   ↳ ERROR on PMID {pmid}: {e}"
            llm_log.append(err_line); _persist("llm_log", llm_log)
            log_box.code("\n".join(llm_log[-20:]), language="text")

        prog.progress(int(i * 100 / total), text=f"LLM progress: {i}/{total}")

    st.success("LLM extraction complete ✅")

    # Final view: update the same table_box (no second table call)
    out_df = flatten_to_rows(st.session_state.get("batch_results", {}))
    table_box.dataframe(out_df, use_container_width=True)

    # Downloads
    colD, colE = st.columns([1, 1])
    with colD:
        st.download_button(
            "⬇️ Download findings (.csv)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="pubmed_mutation_findings.csv",
            mime="text/csv"
        )
    with colE:
        st.download_button(
            "⬇️ Download raw JSON (.json)",
            data=json.dumps(st.session_state["batch_results"], ensure_ascii=True, indent=2).encode("utf-8"),
            file_name="pubmed_batch_results.json",
            mime="application/json"
        )

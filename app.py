# app.py
import os, json, time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import date

from services.pubmed import (
    esearch_reviews, esummary, parse_pubdate_interval, overlaps, to_pdat
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts, flatten_to_rows

# ---------------- Helpers to persist & bucketize ----------------
def _persist(key, value):
    st.session_state[key] = value
    return value

def _bucketize_papers(papers_dict):
    """Split papers into fetched/no_pmc/error lists for display/download."""
    fetched, no_pmc, errors = [], [], []
    for pmid, info in papers_dict.items():
        row = {
            "PMID": pmid,
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
st.title("🧪 PubMed Review Miner: Query → Full Text → LLM (Batch)")
st.caption("Search review articles by query & date range, fetch PMC full text, run your LLM extractor across all, and download a CSV of protein/mutation findings.")

# ---------------- Sidebar (advanced controls) ----------------
with st.sidebar:
    st.header("LLM & Extraction Settings")
    st.caption("These control chunking, filters, and thresholds used for every paper.")
    virus_filter = st.text_input("Virus filter (optional)", value="Dengue virus")
    protein_filter = st.text_input("Protein filter (optional)", value="protein")
    exhaustive = st.checkbox("Exhaustive mode", value=True)
    with st.expander("Advanced", expanded=False):
        chunk_chars = st.slider("Max chars per chunk", 8000, 24000, 16000, 1000)
        overlap_chars = st.slider("Overlap per chunk", 200, 1500, 500, 50)
        delay_ms = st.slider("Delay between chunk calls (ms)", 0, 1500, 400, 50)
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
        require_mut_quote = st.checkbox("Require mutation token in a quote", value=True)
    st.divider()
    st.caption("Tip: Add NCBI_API_KEY and GROQ_API_KEY to .env for reliability.")

# ---------------- Step 1–2: Query & Date Range ----------------
st.subheader("1) Build your PubMed query (reviews only)")
st.write("Paste a PubMed query (or build it in the PubMed UI and paste here). We’ll automatically restrict to **Review** articles.")
query = st.text_area("Query", height=100, placeholder='e.g., dengue[MeSH Terms] AND mutation[Text Word]')

st.subheader("2) Publication date range")
# Default to 2005–2025 inclusive
rng = st.date_input("Inclusive range", value=(date(2005,1,1), date(2025,12,31)))
if isinstance(rng, tuple):
    start_date, end_date = rng
else:
    start_date, end_date = (rng, rng)

colA, colB, colC = st.columns([1,1,1])
with colA:
    sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
with colB:
    cap = st.slider("Max records to fetch", 50, 5000, 1000, 50)
with colC:
    go = st.button("🔎 Search PubMed (reviews)")

if go:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        mindate = to_pdat(start_date)
        maxdate = to_pdat(end_date)
        with st.spinner("Searching PubMed (reviews)…"):
            pmids = esearch_reviews(query.strip(), mindate=mindate, maxdate=maxdate, sort=sort, cap=cap)
            sums = esummary(pmids)
        st.success(f"Found {len(pmids)} candidate review PMIDs (before local date parsing).")

        # local robust interval filter to match CSV-style PubDate shapes
        rows = []
        for pid in pmids:
            meta = sums.get(pid, {})
            pubdate_raw = meta.get("pubdate") or ""
            if overlaps(parse_pubdate_interval(pubdate_raw), (start_date, end_date)):
                rows.append({
                    "PMID": pid,
                    "Title": meta.get("title") or "",
                    "Journal": meta.get("source") or "",
                    "PubDate": pubdate_raw,
                    "PubMed Link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                })

        st.markdown("#### Search results (after robust date filter)")
        df_hits = pd.DataFrame(rows)
        if df_hits.empty:
            st.info("No results in this date range.")
        else:
            st.dataframe(df_hits, use_container_width=True)
            st.download_button(
                "⬇️ Download matches (.csv)",
                data=df_hits.to_csv(index=False).encode("utf-8"),
                file_name="pubmed_review_matches.csv",
                mime="text/csv"
            )
            _persist("hits_pmids", df_hits["PMID"].tolist())

            # ---- Selection UI: Select All or pick specific PMIDs ----
            st.markdown("##### Select PMIDs to process")
            if "selected_pmids" not in st.session_state:
                st.session_state["selected_pmids"] = []
            col1, col2 = st.columns([1,3])
            with col1:
                select_all = st.checkbox("Select all", value=(len(st.session_state["selected_pmids"]) == len(df_hits)))
            if select_all:
                st.session_state["selected_pmids"] = df_hits["PMID"].tolist()
            with col2:
                st.session_state["selected_pmids"] = st.multiselect(
                    "Choose PMIDs",
                    options=df_hits["PMID"].tolist(),
                    default=st.session_state["selected_pmids"],
                )
            st.caption(f"Selected: {len(st.session_state['selected_pmids'])} of {len(df_hits)}")
            if st.session_state["selected_pmids"]:
                st.download_button(
                    "⬇️ Download selected PMIDs (.txt)",
                    data=("\n".join(st.session_state["selected_pmids"]).encode("utf-8")),
                    file_name="selected_pmids.txt",
                    mime="text/plain",
                )

# ---------------- Step 3: Fetch → LLM with live progress & persistent state ----------------
st.divider()
st.subheader("3) Fetch PMC full texts & run LLM across all")

colX, colY, colZ = st.columns([1,1,2])
with colX:
    run_fetch = st.button("📥 Fetch PMC texts")
with colY:
    run_llm = st.button("🧠 Run LLM on fetched")
with colZ:
    if st.button("🔁 Reset batch"):
        for k in ["hits_pmids", "batch_papers", "batch_results", "llm_log"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

# ---------- A) FETCH: Persist and show buckets ----------
if run_fetch:
    # Prefer user-selected PMIDs if available; else all hits
    pmids = st.session_state.get("selected_pmids") or st.session_state.get("hits_pmids", [])
    if not pmids:
        st.warning("Please run a search first (Step 1 & 2).")
    else:
        if not os.getenv("GROQ_API_KEY"):
            st.info("Note: GROQ_API_KEY isn’t set—LLM will still run if your code handles it, but set it in .env for reliability.")

        with st.spinner(f"Fetching PMC full texts for {len(pmids)} selected PMIDs…"):
            papers = fetch_all_fulltexts(pmids, delay_ms=150)
            _persist("batch_papers", papers)

        fetched, no_pmc, errors = _bucketize_papers(st.session_state["batch_papers"])
        n_ok, n_no, n_err = len(fetched), len(no_pmc), len(errors)
        st.success(f"PMC texts fetched: ✅ {n_ok} | No PMC: ⚠️ {n_no} | Errors: ❌ {n_err}")

        if n_ok:
            st.markdown("##### ✅ Fetched (has PMC full text)")
            df_fetched = pd.DataFrame(fetched)
            st.dataframe(df_fetched, use_container_width=True, height=min(400, 40 + 28 * len(df_fetched)))
            st.download_button(
                "⬇️ Download fetched (.csv)",
                data=df_fetched.to_csv(index=False).encode("utf-8"),
                file_name="fetched_papers.csv",
                mime="text/csv"
            )

        if n_no:
            st.markdown("##### ⚠️ No PMC (skipped)")
            df_no = pd.DataFrame(no_pmc)
            st.dataframe(df_no, use_container_width=True, height=min(400, 40 + 28 * len(df_no)))
            st.download_button(
                "⬇️ Download no_pmc (.csv)",
                data=df_no.to_csv(index=False).encode("utf-8"),
                file_name="no_pmc_papers.csv",
                mime="text/csv"
            )

        if n_err:
            st.markdown("##### ❌ Errors (fetch failed)")
            df_err = pd.DataFrame(errors)
            st.dataframe(df_err, use_container_width=True, height=min(400, 40 + 28 * len(df_err)))
            st.download_button(
                "⬇️ Download fetch_errors (.csv)",
                data=df_err.to_csv(index=False).encode("utf-8"),
                file_name="fetch_errors.csv",
                mime="text/csv"
            )

# ---------- B) LLM: Real-time progress + persistent results ----------
if run_llm:
    papers = st.session_state.get("batch_papers", {})
    if not papers:
        st.warning("Please fetch PMC texts first.")
    else:
        # Which PMIDs to process
        pmids_order = [pid for pid, info in papers.items() if info.get("status") == "ok"]
        if not pmids_order:
            st.info("Nothing to analyze (no fetched papers).")
        else:
            # Initialize / reuse logs and results
            llm_log = st.session_state.get("llm_log", [])
            batch_results = st.session_state.get("batch_results", {})

            # UI containers for live updates
            prog = st.progress(0, text="Starting LLM passes…")
            log_box = st.empty()
            table_box = st.empty()

            total = len(pmids_order)
            for i, pmid in enumerate(pmids_order, start=1):
                title = papers[pmid].get("title") or ""
                pmcid = papers[pmid].get("pmcid") or ""
                log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) — {title[:80]}"
                llm_log.append(log_line)
                _persist("llm_log", llm_log)
                log_box.code("\n".join(llm_log[-20:]), language="text")  # show last ~20 lines

                try:
                    # Run LLM for just this one (reusing your exact pipeline)
                    single_dict = analyze_texts(
                        {pmid: papers[pmid]},
                        virus_filter=virus_filter, protein_filter=protein_filter,
                        exhaustive=exhaustive, chunk_chars=chunk_chars, overlap_chars=overlap_chars,
                        delay_ms=delay_ms, min_confidence=min_conf, require_mut_quote=require_mut_quote,
                    )
                    # Merge into batch_results (persist after each paper)
                    batch_results.update(single_dict)
                    _persist("batch_results", batch_results)

                    # Live partial table of findings so far
                    out_df_partial = flatten_to_rows(batch_results)
                    table_box.dataframe(out_df_partial, use_container_width=True)
                except Exception as e:
                    err_line = f"   ↳ ERROR on PMID {pmid}: {e}"
                    llm_log.append(err_line)
                    _persist("llm_log", llm_log)
                    log_box.code("\n".join(llm_log[-20:]), language="text")

                prog.progress(int(i * 100 / total), text=f"LLM progress: {i}/{total}")

            st.success("LLM extraction complete for fetched papers.")

            # Final tables + downloads
            out_df = flatten_to_rows(st.session_state.get("batch_results", {}))
            st.markdown("#### Findings table (one row per mutation finding)")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "⬇️ Download findings (.csv)",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="pubmed_mutation_findings.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇️ Download raw JSON (.json)",
                data=json.dumps(st.session_state["batch_results"], ensure_ascii=True, indent=2).encode("utf-8"),
                file_name="pubmed_batch_results.json",
                mime="application/json"
            )

            # Summary line (constant & visible)
            fetched, no_pmc, errors = _bucketize_papers(st.session_state["batch_papers"])
            st.info(
                f"Summary — PMC fetched: ✅ {len(fetched)} | "
                f"No PMC: ⚠️ {len(no_pmc)} | Fetch errors: ❌ {len(errors)} | "
                f"LLM processed: 🧠 {len([p for p in fetched if p['PMID'] in st.session_state['batch_results']])}"
            )


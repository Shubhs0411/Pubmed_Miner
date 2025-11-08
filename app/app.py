# Enhanced app.py - Add to your existing code
from __future__ import annotations

import os, json, io, zipfile
from datetime import date
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from services.pmc import get_last_fetch_source
from services.pubmed import (
    esearch_reviews, esummary, parse_pubdate_interval, overlaps, to_pdat
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts, flatten_to_rows

# Import prompts for editing
from llm.prompts import PROMPTS


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


def main():
    load_dotenv()
    st.set_page_config(page_title="PubMed → PMC → LLM (Batch Miner)", layout="wide")
    st.title("🧪 PubMed Review Miner")
    st.caption("Search review articles, fetch PMC full text, run your LLM extractor, and download findings.")

    # ===== NEW: Model & API Key Configuration Section =====
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select LLM Model",
            ["Gemini (Google)", "GPT-4o (OpenAI)", "Claude (Anthropic)", "Llama (Groq)", "Hugging Face (Any Model)"],
            index=0,
            help="Choose which LLM to use for extraction"
        )
        
        # API Key input based on selection
        api_key_env_var = None
        if "Gemini" in model_choice:
            api_key = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
                help="Get from: https://ai.google.dev/"
            )
            api_key_env_var = "GEMINI_API_KEY"
            model_name = st.text_input("Model Name", value="gemini-2.5-flash-lite")
            
        elif "GPT-4o" in model_choice:
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Get from: https://platform.openai.com/api-keys"
            )
            api_key_env_var = "OPENAI_API_KEY"
            model_name = st.text_input("Model Name", value="gpt-4o-2024-11-20")
            
        elif "Claude" in model_choice:
            api_key = st.text_input(
                "Anthropic API Key",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
                help="Get from: https://console.anthropic.com/"
            )
            api_key_env_var = "ANTHROPIC_API_KEY"
            model_name = st.text_input("Model Name", value="claude-sonnet-4-20250514")
            
        elif "Groq" in model_choice or "Llama" in model_choice:  # Llama (Groq)
            api_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                help="Get from: https://console.groq.com/keys"
            )
            api_key_env_var = "GROQ_API_KEY"
            model_name = st.text_input("Model Name", value="llama-3.3-70b-versatile")
            
        elif "Hugging Face" in model_choice:  # Hugging Face
            api_key = st.text_input(
                "Hugging Face API Key",
                value=os.getenv("HF_API_KEY", "") or os.getenv("HUGGINGFACE_API_KEY", ""),
                type="password",
                help="Get from: https://huggingface.co/settings/tokens"
            )
            api_key_env_var = "HF_API_KEY"
            default_model = st.text_input(
                "Model Name (e.g., gpt2)", 
                value="gpt2",
                help="Free tier models: gpt2, distilgpt2, google/flan-t5-base. Note: Most instruction models require gated access or paid tier. For best results, use Groq/OpenAI/Anthropic."
            )
            
            st.warning(
                "⚠️ **Important**: Hugging Face free Inference API has very limited models available. "
                "Most instruction-tuned models (Qwen, Phi, Mistral, etc.) require gated access or paid subscriptions. "
                "For reliable structured JSON extraction, we strongly recommend using Groq, OpenAI, or Anthropic instead.",
                icon="⚠️"
            )
            model_name = default_model
        
        # Strip whitespace from API key
        if api_key:
            api_key = api_key.strip()
            # Update environment variable (but frontend will take priority in backend)
            os.environ[api_key_env_var] = api_key
        
        st.divider()
        
        # Extraction parameters
        st.header("⚙️ Extraction Settings")
        chunk_chars = st.slider("Max chars per chunk", 8000, 24000, 16000, 1000)
        overlap_chars = st.slider("Overlap per chunk", 200, 1500, 500, 50)
        delay_ms = st.slider("Delay between chunk calls (ms)", 0, 1500, 400, 50)
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
        
        st.divider()
        st.caption("💡 Tip: Test with 1-2 papers first to verify API keys work")

    # ===== NEW: Prompt Editor Section =====
    with st.expander("📝 **Advanced: Edit Extraction Prompt**", expanded=False):
        st.markdown("""
        **Expert users only!** Modify the system prompt used for extraction.
        Changes affect how mutations and features are identified.
        """)
        
        # Load current prompt
        current_prompt = PROMPTS.analyst_prompt
        
        # Tabs for different prompt sections
        tab1, tab2, tab3 = st.tabs(["Main Prompt", "Instructions", "Preview"])
        
        with tab1:
            edited_prompt = st.text_area(
                "Analyst Prompt (used for extraction)",
                value=current_prompt,
                height=400,
                help="This prompt guides the LLM's extraction. Use {TEXT} placeholder."
            )
            
            if st.button("💾 Save Prompt Changes"):
                PROMPTS.analyst_prompt = edited_prompt
                st.success("✅ Prompt updated! Will be used for next extraction.")
        
        with tab2:
            st.markdown("""
            ### Prompt Guidelines:
            - **{TEXT}** placeholder is required - gets replaced with paper content
            - Define clear JSON schema for consistent outputs
            - Include few-shot examples for better accuracy
            - Specify fields explicitly (virus, protein, mutation, effect, etc.)
            - Request evidence quotes for validation
            """)
            
        with tab3:
            st.code(edited_prompt[:1000] + "\n\n... (truncated)" if len(edited_prompt) > 1000 else edited_prompt, language="text")
        
        # Reset button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Reset to Default"):
                from llm.prompts import AnalystPrompts
                PROMPTS.analyst_prompt = AnalystPrompts().analyst_prompt
                st.rerun()

    # ===== Search Section (unchanged) =====
    st.subheader("1) Enter your PubMed query (reviews only)")
    st.write("Paste a PubMed query (we'll restrict to **Review** articles automatically).")
    query = st.text_area("Query", height=100, placeholder='e.g., dengue[MeSH Terms] AND mutation[Text Word]')

    st.subheader("2) Choose publication date range & search")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        default_range = (date(2005, 1, 1), date(2025, 12, 31))
        rng = st.date_input("Inclusive range", value=default_range, min_value=date(2005, 1, 1), 
                           help="Earliest allowed start date is Jan 1, 2005.")
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
            
            df_hits = pd.DataFrame(rows)
            if df_hits.empty:
                for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.info("No results in this date range.")
            else:
                df_hits["PMID"] = df_hits["PMID"].astype(str)
                _persist("hits_df", df_hits.to_dict("records"))
                _persist("hits_pmids", df_hits["PMID"].tolist())
                st.session_state["selected_pmids"] = []
                st.success(f"Found {len(df_hits)} results. See 'Results' below to select papers.")

    # ===== Results display (unchanged) =====
    if st.session_state.get("hits_df"):
        st.markdown("#### Results")
        df_hits = pd.DataFrame(st.session_state["hits_df"])
        pmid_options = [str(x) for x in st.session_state.get("hits_pmids", [])]
        
        st.dataframe(df_hits, width='stretch')
        st.download_button(
            "⬇️ Download matches (.csv)", 
            data=df_hits.to_csv(index=False).encode("utf-8"), 
            file_name="pubmed_review_matches.csv", 
            mime="text/csv"
        )
        
        st.markdown("##### Select PMIDs to process with LLM")
        if "selected_pmids" not in st.session_state:
            st.session_state["selected_pmids"] = []
        
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
                mime="text/plain"
            )

    # ===== Extraction section (modified to pass model info) =====
    st.subheader("3) Run extraction")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        override_all = st.checkbox("Process ALL results (ignore selection)", value=False, 
                                   help="When checked, all results from Step 2 will be sent to the LLM.")
    with colB:
        clear_previous = st.checkbox("Clear previous results before run", value=True, 
                                     help="Prevents older findings from appearing alongside this run.")
    with colC:
        run_all = st.button("🚀 Fetch PMC & Run LLM", 
                           disabled=(not override_all and len(st.session_state.get("selected_pmids", [])) == 0))

    if st.button("🗑️ Reset"):
        for k in ["hits_df", "hits_pmids", "batch_papers", "batch_results", "llm_log", 
                  "selected_pmids", "select_all_hits", "pmid_multiselect"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if run_all:
        hits = [str(x) for x in st.session_state.get("hits_pmids", [])]
        selected = [str(x) for x in st.session_state.get("selected_pmids", [])]
        pmids = hits if override_all else selected
        
        if not pmids:
            st.warning("No PMIDs selected. Pick at least one in Step 2 or check 'Process ALL results'.")
            st.stop()
        
        if clear_previous:
            st.session_state["batch_results"] = {}
            st.session_state["llm_log"] = []
        
        # Validate API key (strip whitespace first)
        api_key = api_key.strip() if api_key else ""
        if not api_key:
            st.error(f"⚠️ Please enter your {model_choice} API key in the sidebar!")
            st.stop()
        
        st.info(f"🤖 Using **{model_choice}** (model: `{model_name}`)")
        
        with st.spinner(f"Fetching PMC full texts for {len(pmids)} PMIDs…"):
            papers = fetch_all_fulltexts(pmids, delay_ms=150)
            _persist("batch_papers", papers)
        
        fetched, no_pmc, errors = _bucketize_papers(papers)
        n_ok, n_no, n_err = len(fetched), len(no_pmc), len(errors)
        st.success(f"PMC texts: ✅ {n_ok} fetched | ⚠️ {n_no} no PMC | ❌ {n_err} errors")
        
        if n_ok:
            with st.expander("Show fetched list", expanded=False):
                st.dataframe(pd.DataFrame(fetched), width='stretch', 
                           height=min(400, 40 + 28 * len(fetched)))
        if n_no:
            with st.expander("Show no-PMC list", expanded=False):
                st.dataframe(pd.DataFrame(no_pmc), width='stretch', 
                           height=min(400, 40 + 28 * len(no_pmc)))
        if n_err:
            with st.expander("Show fetch errors", expanded=False):
                st.dataframe(pd.DataFrame(errors), width='stretch', 
                           height=min(400, 40 + 28 * len(errors)))
        
        if n_ok == 0:
            st.stop()
        
        ok_pmids_this_run = [row["PMID"] for row in fetched]
        
        st.markdown("#### Full texts for this run (selected & fetched)")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pmid in ok_pmids_this_run:
                info = papers.get(pmid, {})
                title = info.get("title") or ""
                pmcid = info.get("pmcid") or ""
                text = (info.get("text") or info.get("fulltext") or info.get("content") or "")
                
                with st.expander(f"{pmid} – {title[:100]}"):
                    if pmcid:
                        st.markdown(f"**PMCID:** {pmcid}  |  [Open PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/)")
                    else:
                        st.markdown("**PMCID:** (not available)")
                    
                    source = (info.get("source") or get_last_fetch_source(pmid) or "unknown")
                    badge = {
                        "jats": "🟢 JATS (XML)", 
                        "html": "🔵 HTML", 
                        "none": "⚪ none"
                    }.get(source, "⚪ unknown")
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
                
                safe_name = f"{pmid}_{(pmcid or 'NO_PMCID')}".replace("/", "_")
                zf.writestr(f"{safe_name}.txt", text if text else "")
        
        st.download_button(
            "⬇️ Download all selected full texts (.zip)", 
            data=zip_buf.getvalue(), 
            file_name="selected_full_texts.zip", 
            mime="application/zip"
        )

        # ===== LLM extraction phase (MODIFIED) =====
        papers = st.session_state.get("batch_papers", {})
        ok_pmids_this_run = [pid for pid, info in papers.items() if info.get("status") == "ok"]
        
        if not ok_pmids_this_run:
            st.info("Nothing to analyze: no successfully fetched PMC full texts in this run.")
            st.stop()
        
        llm_log = st.session_state.get("llm_log", [])
        batch_results = st.session_state.get("batch_results", {})
        
        prog = st.progress(0, text="Starting LLM…")
        log_box = st.empty()
        st.markdown("#### Findings")
        table_box = st.empty()
        
        # Pass model selection to analyze_texts (ensure api_key is passed even if env is set)
        # Frontend API key takes priority over env var in backend
        llm_meta = {
            "model_choice": model_choice,
            "model_name": model_name,
            "api_key": api_key,  # This will be used as PRIMARY in backend
        }
        
        total = len(ok_pmids_this_run)
        for i, pmid in enumerate(ok_pmids_this_run, start=1):
            title = papers[pmid].get("title") or ""
            pmcid = papers[pmid].get("pmcid") or ""
            log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) – {title[:80]}"
            llm_log.append(log_line)
            _persist("llm_log", llm_log)
            log_box.code("\n".join(llm_log[-20:]), language="text")
            
            try:
                single_dict = analyze_texts(
                    {pmid: papers[pmid]},
                    chunk_chars=chunk_chars, 
                    overlap_chars=overlap_chars,
                    delay_ms=delay_ms, 
                    min_confidence=min_conf, 
                    require_mut_quote=True,
                    llm_meta=llm_meta,  # NEW: pass model config
                )
                batch_results.update(single_dict)
                _persist("batch_results", batch_results)
                
                out_df_partial = flatten_to_rows(batch_results)
                table_box.dataframe(out_df_partial, width='stretch')
            except Exception as e:
                err_line = f"   ↳ ERROR on PMID {pmid}: {e}"
                llm_log.append(err_line)
                _persist("llm_log", llm_log)
                log_box.code("\n".join(llm_log[-20:]), language="text")
            
            prog.progress(int(i * 100 / total), text=f"LLM progress: {i}/{total}")
        
        st.success("LLM extraction complete ✅")
        
        out_df = flatten_to_rows(st.session_state.get("batch_results", {}))
        table_box.dataframe(out_df, width='stretch')
        
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


if __name__ == "__main__":
    main()
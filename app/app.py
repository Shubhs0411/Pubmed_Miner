# Enhanced app.py - Simplified UI
from __future__ import annotations

import os, json, io, zipfile
from datetime import date
import calendar
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from typing import Dict

from services.pmc import get_last_fetch_source
from services.pubmed import (
    esearch_reviews, esearch_all, esummary, parse_pubdate_interval, overlaps
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts
from pipeline.simple_csv import raw_to_csv

# Import prompts for editing
from llm.prompts import PROMPTS

# Import natural language query converter
from llm.utils import natural_query_to_pubmed_query


def _persist(key, value):
    st.session_state[key] = value
    return value


def main():
    load_dotenv()
    st.set_page_config(page_title="PubMed Miner", layout="wide")
    st.title("🧬 PubMed Miner")
    st.caption("Search PubMed, extract features with LLM, and download results.")

    # ===== Sidebar Configuration =====
    with st.sidebar:
        # NCBI Configuration
        st.header("🔬 NCBI Configuration")
        ncbi_api_key = st.text_input(
            "NCBI API Key",
            value=os.getenv("NCBI_API_KEY", ""),
            type="password",
            help="Get from: https://www.ncbi.nlm.nih.gov/account/settings/"
        )
        if ncbi_api_key:
            ncbi_api_key = ncbi_api_key.strip()
            os.environ["NCBI_API_KEY"] = ncbi_api_key
        
        st.divider()
        
        # LLM Configuration
        st.header("🤖 LLM Configuration")
        
        model_choice = st.selectbox(
            "Select LLM Model",
            [
                "Gemini (Google)",
                "GPT-4o (OpenAI)",
                "Claude (Anthropic)",
                "Llama (Groq)",
                "Custom (Hackathon)",
            ],
            index=0,
        )
        
        # API Key input based on selection
        api_key_env_var = None
        api_key = ""
        model_name = ""
        custom_api_url = os.getenv("CUSTOM_LLM_URL", "")
        custom_timeout = int(os.getenv("CUSTOM_LLM_TIMEOUT", "60"))
        custom_headers_dict: Dict[str, str] = {}
        
        if "Gemini" in model_choice:
            api_key_env_var = "GEMINI_API_KEY"
            api_key = st.text_input(
                "Gemini API Key",
                value=os.getenv("GEMINI_API_KEY", ""),
                type="password",
            )
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                help="e.g., gemini-2.5-flash-lite, gemini-2.5-flash"
            )
        elif "GPT-4o" in model_choice:
            api_key_env_var = "OPENAI_API_KEY"
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
            )
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("OPENAI_MODEL", "gpt-4o"),
            )
        elif "Claude" in model_choice:
            api_key_env_var = "ANTHROPIC_API_KEY"
            api_key = st.text_input(
                "Anthropic API Key",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
            )
            model_name = st.text_input(
                "Model Name",
                value=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )
        elif "Llama" in model_choice or "Groq" in model_choice:
            api_key_env_var = "GROQ_API_KEY"
            api_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
            )
            model_name = st.selectbox(
                "Model Name",
                ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                index=0,
            )
        elif "Custom" in model_choice:
            custom_api_url = st.text_input(
                "API URL",
                value=custom_api_url,
                help="Full endpoint URL (e.g., https://api.example.com/v1/chat/completions)"
            )
            api_key = st.text_input(
                "API Key (optional)",
                value=os.getenv("CUSTOM_LLM_API_KEY", ""),
                type="password",
            )
            api_key_env_var = "CUSTOM_LLM_API_KEY"
            custom_headers_json = st.text_area(
                "Extra HTTP headers (JSON)",
                value=os.getenv("CUSTOM_LLM_HEADERS", ""),
                help="Optional: JSON object of additional headers"
            )
            if custom_headers_json.strip():
                try:
                    custom_headers_dict = json.loads(custom_headers_json.strip())
                except Exception:
                    st.error("Invalid JSON for headers")
                    custom_headers_dict = {}
            custom_timeout = st.number_input(
                "Request timeout (seconds)",
                min_value=5,
                max_value=600,
                value=custom_timeout,
            )
            model_name = st.text_input("Model Name", value="custom")
        
        if api_key:
            api_key = api_key.strip()
            if api_key_env_var:
                os.environ[api_key_env_var] = api_key

    # ===== Section 1: Search PubMed =====
    st.header("Section 1: Search PubMed")
    
    # Query input tabs
    query_tab1, query_tab2 = st.tabs(["✨ Natural Language", "⚙️ PubMed Query"])
    
    with query_tab1:
        nl_query = st.text_area(
            "Enter your research question",
            height=100,
            placeholder="Example: Find papers about dengue virus mutations that affect E protein binding",
            key="nl_query_input"
        )
        
        if st.button("🪄 Convert to PubMed Query", type="primary"):
            if not nl_query.strip():
                st.warning("Please enter a natural language query first.")
            else:
                api_key_check = api_key.strip() if api_key else ""
                if not api_key_check and "Custom" not in model_choice:
                    st.error(f"Please enter your {model_choice} API key in the sidebar first!")
                else:
                    try:
                        with st.spinner("Converting query..."):
                            llm_meta = {
                                "model_choice": model_choice,
                                "model_name": model_name,
                                "api_key": api_key,
                            }
                            if "Custom" in model_choice:
                                if not custom_api_url:
                                    st.error("Please provide the Custom LLM API URL in the sidebar first!")
                                    st.stop()
                                llm_meta["api_url"] = custom_api_url
                                llm_meta["timeout"] = custom_timeout
                                if custom_headers_dict:
                                    llm_meta["extra_headers"] = custom_headers_dict
                            
                            converted_query = natural_query_to_pubmed_query(nl_query.strip(), llm_meta)
                            st.session_state["converted_query"] = converted_query
                            st.session_state["query_to_search"] = converted_query
                            st.success("✅ Query converted successfully!")
                    except Exception as e:
                        st.error(f"Error converting query: {str(e)}")
    
    with query_tab2:
        manual_query = st.text_area(
            "Enter PubMed Boolean Query",
            height=100,
            placeholder='e.g., dengue[MeSH Terms] AND mutation[Text Word]',
            key="manual_query_input"
        )
        if manual_query.strip():
            st.session_state["query_to_search"] = manual_query.strip()
    
    # Display current query
    query = st.session_state.get("query_to_search", "")
    if query:
        st.info(f"**Current Query:** `{query}`")
    
    # Search parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mindate = st.text_input("Start Date (MM/YYYY)", value="01/2005", placeholder="MM/YYYY")
    with col2:
        maxdate = st.text_input("End Date (MM/YYYY)", value="12/2025", placeholder="MM/YYYY")
    with col3:
        sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
    with col4:
        cap = st.slider("Max Papers", 0, 500, 100, 100)
    
    reviews_only = st.checkbox("🔍 Restrict to Review articles only", value=True)
    
    if st.button("🔎 Search PubMed", type="primary", disabled=(not query or not query.strip())):
        if not query.strip():
            st.warning("Please enter a query first.")
        else:
            try:
                min_parts = mindate.strip().split("/")
                max_parts = maxdate.strip().split("/")
                
                if len(min_parts) != 2 or len(max_parts) != 2:
                    st.error("Invalid date format. Please use MM/YYYY")
                    st.stop()
                
                mindate_formatted = f"{min_parts[1]}/{min_parts[0]}"
                maxdate_formatted = f"{max_parts[1]}/{max_parts[0]}"
                
                start_date = date(int(min_parts[1]), int(min_parts[0]), 1)
                end_year, end_month = int(max_parts[1]), int(max_parts[0])
                last_day = calendar.monthrange(end_year, end_month)[1]
                end_date = date(end_year, end_month, last_day)
                
                with st.spinner("Searching PubMed..."):
                    if reviews_only:
                        pmids = esearch_reviews(query.strip(), mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    else:
                        pmids = esearch_all(query.strip(), mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    
                    if not pmids:
                        st.warning("No results found for your query.")
                        for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                            if k in st.session_state:
                                del st.session_state[k]
                    else:
                        st.info(f"Found {len(pmids)} PMIDs. Fetching metadata...")
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
                                })
                        
                        df_hits = pd.DataFrame(rows)
                        if df_hits.empty:
                            st.warning("Found results, but none match your date range.")
                            for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                                if k in st.session_state:
                                    del st.session_state[k]
                        else:
                            df_hits["PMID"] = df_hits["PMID"].astype(str)
                            _persist("hits_df", df_hits.to_dict("records"))
                            _persist("hits_pmids", df_hits["PMID"].tolist())
                            st.session_state["selected_pmids"] = []
                            st.success(f"✅ Found {len(df_hits)} results.")
            
            except Exception as e:
                st.error(f"Error searching PubMed: {str(e)}")
    
    # Display results
    if st.session_state.get("hits_df"):
        st.markdown("### Results")
        df_hits = pd.DataFrame(st.session_state["hits_df"])
        st.dataframe(df_hits, use_container_width=True)
        st.download_button(
            "⬇️ Download Results (.csv)", 
            data=df_hits.to_csv(index=False).encode("utf-8"), 
            file_name="pubmed_results.csv", 
            mime="text/csv"
        )

    # ===== Section 2: Select PMIDs =====
    st.divider()
    st.header("Section 2: Select PMIDs")
    
    if st.session_state.get("hits_pmids"):
        pmid_options = [str(x) for x in st.session_state.get("hits_pmids", [])]
        
        if "selected_pmids" not in st.session_state:
            st.session_state["selected_pmids"] = []
        
        # Select all checkbox
        select_all = st.checkbox("Select All", value=False, key="select_all_hits")
        if select_all:
            st.session_state["selected_pmids"] = pmid_options.copy()
        elif not select_all and len(st.session_state["selected_pmids"]) == len(pmid_options):
            st.session_state["selected_pmids"] = []
        
        # Multi-select
        selected = st.multiselect(
            "Choose PMIDs to extract", 
            options=pmid_options,
            default=[p for p in st.session_state["selected_pmids"] if p in pmid_options], 
            key="pmid_multiselect"
        )
        st.session_state["selected_pmids"] = selected
        
        st.info(f"**Selected:** {len(selected)} of {len(pmid_options)} PMIDs")
        
        if selected:
            st.download_button(
                "⬇️ Download Selected PMIDs (.txt)", 
                data="\n".join(selected).encode("utf-8"), 
                file_name="selected_pmids.txt", 
                mime="text/plain"
            )
    else:
        st.info("Search PubMed first to see available PMIDs.")

    # ===== Section 3: Run LLM Extraction =====
    st.divider()
    st.header("Section 3: Run LLM Extraction")
    
    # Editable prompt
    with st.expander("📝 Edit Extraction Prompt", expanded=False):
        current_prompt = PROMPTS.analyst_prompt
        edited_prompt = st.text_area(
            "Full Prompt",
            value=current_prompt,
            height=400,
            help="Edit the complete prompt. Keep the {TEXT} placeholder at the end.",
            key="prompt_editor"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Prompt"):
                PROMPTS.analyst_prompt = edited_prompt
                st.success("✅ Prompt updated!")
        with col2:
            if st.button("🔄 Reset to Default"):
                from llm.prompts import AnalystPrompts
                default_prompts = AnalystPrompts()
                PROMPTS.analyst_prompt = default_prompts.analyst_prompt
                st.success("✅ Reset to default.")
                st.rerun()
    
    # Run LLM button
    selected_pmids = st.session_state.get("selected_pmids", [])
    run_llm = st.button("🚀 Run LLM", type="primary", disabled=(len(selected_pmids) == 0))
    
    if run_llm:
        pmids = [str(x) for x in selected_pmids]
        
        if not pmids:
            st.warning("No PMIDs selected. Please select PMIDs in Section 2.")
            st.stop()
        
        st.session_state["batch_results"] = {}
        st.session_state["llm_log"] = []
        
        # Validate API key
        api_key = api_key.strip() if api_key else ""
        if not api_key and "Custom" not in model_choice:
            st.error(f"Please enter your {model_choice} API key in the sidebar!")
            st.stop()
        if "Custom" in model_choice and not custom_api_url:
            st.error("Please provide the Custom LLM API URL in the sidebar.")
            st.stop()
        
        st.info(f"Using **{model_choice}** (model: `{model_name}`)")
        
        # Fetch full texts
        with st.spinner(f"Fetching PMC full texts for {len(pmids)} PMIDs…"):
            papers = fetch_all_fulltexts(pmids, delay_ms=150)
            _persist("batch_papers", papers)
        
        # Count results
        fetched = [pid for pid, info in papers.items() if info.get("status") == "ok"]
        no_pmc = [pid for pid, info in papers.items() if info.get("status") == "no_pmc_fulltext"]
        errors = [pid for pid, info in papers.items() if info.get("status") == "error"]
        
        st.success(f"PMC texts: ✅ {len(fetched)} fetched | ⚠️ {len(no_pmc)} no PMC | ❌ {len(errors)} errors")
        
        if len(fetched) == 0:
            st.warning("No papers with full text available. Cannot proceed with extraction.")
            st.stop()
        
        ok_pmids_this_run = fetched
        
        # LLM extraction
        llm_log = st.session_state.get("llm_log", [])
        batch_results = st.session_state.get("batch_results", {})
        
        prog = st.progress(0, text="Starting LLM extraction…")
        log_box = st.empty()
        
        llm_meta = {
            "model_choice": model_choice,
            "model_name": model_name,
            "api_key": api_key,
            "analyst_prompt": PROMPTS.analyst_prompt,
        }
        
        if "Custom" in model_choice:
            llm_meta["api_url"] = custom_api_url
            llm_meta["timeout"] = custom_timeout
            if custom_headers_dict:
                llm_meta["extra_headers"] = custom_headers_dict
            if custom_api_url and ("/v1" in custom_api_url or "/openai" in custom_api_url.lower()):
                llm_meta["openai_compatible"] = True
        
        total = len(ok_pmids_this_run)
        
        for i, pmid in enumerate(ok_pmids_this_run, start=1):
            title = papers[pmid].get("title") or ""
            pmcid = papers[pmid].get("pmcid") or ""
            log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) – {title[:60]}"
            llm_log.append(log_line)
            _persist("llm_log", llm_log)
            log_box.code("\n".join(llm_log[-10:]), language="text")
            
            try:
                single_dict = analyze_texts(
                    {pmid: papers[pmid]},
                    chunk_chars=16000, 
                    overlap_chars=500,
                    delay_ms=400, 
                    min_confidence=0.0, 
                    require_mut_quote=False,  # No filtering
                    llm_meta=llm_meta,
                )
                
                batch_results.update(single_dict)
                _persist("batch_results", batch_results)
                        
            except Exception as e:
                err_line = f"   ↳ ERROR on PMID {pmid}: {e}"
                llm_log.append(err_line)
                _persist("llm_log", llm_log)
                log_box.code("\n".join(llm_log[-10:]), language="text")
            
            prog.progress(int(i * 100 / total), text=f"Progress: {i}/{total}")
        
        st.success("✅ LLM extraction complete!")
    
    # CSV Export section (always visible if results exist, updates when filter toggle changes)
    if st.session_state.get("batch_results"):
        st.markdown("### CSV Export")
        
        apply_filters = st.checkbox(
            "Apply Filters", 
            value=True,
            help="When enabled: Skip rows with feature name but no position, and skip empty rows. When disabled: Show all rows.",
            key="csv_filter_toggle"
        )
        
        # Convert to CSV and display (updates when toggle changes)
        csv_df = raw_to_csv(st.session_state.get("batch_results", {}), apply_filters=apply_filters)
        
        if not csv_df.empty:
            st.dataframe(csv_df, use_container_width=True)
            st.info(f"Showing {len(csv_df)} rows ({'filtered' if apply_filters else 'unfiltered'})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV (.csv)", 
                    data=csv_df.to_csv(index=False).encode("utf-8"), 
                    file_name="raw_llm_output.csv", 
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Raw JSON (.json)", 
                    data=json.dumps(st.session_state["batch_results"], ensure_ascii=True, indent=2).encode("utf-8"), 
                    file_name="raw_llm_output.json", 
                    mime="application/json"
                )
        else:
            st.info("No features extracted.")
            st.download_button(
                "⬇️ Download Raw JSON (.json)", 
                data=json.dumps(st.session_state["batch_results"], ensure_ascii=True, indent=2).encode("utf-8"), 
                file_name="raw_llm_output.json", 
                mime="application/json"
            )
    
    # Section 4: Validation (Optional)
    st.markdown("---")
    st.markdown("### Section 4: Validation (Optional)")
    
    uploaded_file = st.file_uploader(
        "Upload Ground Truth CSV",
        type=["csv"],
        help="Upload a CSV file with ground truth data. Quotes will be matched against LLM output."
    )
    
    if uploaded_file is not None and st.session_state.get("batch_results"):
        try:
            # Read ground truth CSV
            gt_df = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            required_cols = ["pmid", "quote"]
            if not all(col in gt_df.columns for col in required_cols):
                st.error(f"Ground truth CSV must contain columns: {', '.join(required_cols)}")
            else:
                # Extract LLM quotes from batch_results
                llm_quotes = []
                for pmid, entry in st.session_state.get("batch_results", {}).items():
                    if entry.get("status") != "ok":
                        continue
                    result = entry.get("result", {})
                    features = result.get("sequence_features", [])
                    for feat in features:
                        if not isinstance(feat, dict):
                            continue
                        quote = (feat.get("evidence_snippet") or "").strip()
                        if quote:
                            llm_quotes.append({
                                "pmid": pmid,
                                "quote": quote,
                                "virus": (feat.get("virus") or "").strip(),
                                "protein": (feat.get("protein") or "").strip(),
                            })
                
                if not llm_quotes:
                    st.info("No quotes found in LLM output.")
                else:
                    # Simple quote matching using text similarity
                    from difflib import SequenceMatcher
                    
                    matches = []
                    matched_gt_quotes = set()
                    
                    for _, gt_row in gt_df.iterrows():
                        gt_pmid = str(gt_row.get("pmid", "")).strip()
                        gt_quote = str(gt_row.get("quote", "")).strip()
                        
                        if not gt_quote:
                            continue
                        
                        # Find best matching LLM quote for this ground truth quote
                        best_match = None
                        best_score = 0.0
                        
                        for llm_item in llm_quotes:
                            # Only match if PMID matches (if available in GT)
                            if gt_pmid and str(llm_item["pmid"]) != gt_pmid:
                                continue
                            
                            llm_quote = llm_item["quote"]
                            # Calculate similarity
                            similarity = SequenceMatcher(None, gt_quote.lower(), llm_quote.lower()).ratio()
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_match = {
                                    "gt_quote": gt_quote,  # From uploaded CSV (ground truth)
                                    "llm_quote": llm_quote,  # From LLM extraction
                                    "similarity": similarity,
                                    "pmid": llm_item["pmid"],
                                    "virus": llm_item.get("virus", ""),
                                    "protein": llm_item.get("protein", ""),
                                }
                        
                        if best_match and best_score > 0.3:  # Threshold for matching
                            matches.append(best_match)
                            matched_gt_quotes.add(gt_quote)
                    
                    # Find unmatched ground truth quotes
                    unmatched = []
                    for _, gt_row in gt_df.iterrows():
                        gt_quote = str(gt_row.get("quote", "")).strip()
                        if gt_quote and gt_quote not in matched_gt_quotes:
                            unmatched.append({
                                "quote": gt_quote,
                                "pmid": str(gt_row.get("pmid", "")).strip(),
                            })
                    
                    # Display results - compact format
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Matched", len(matches))
                        if matches:
                            for i, match in enumerate(matches[:5], 1):  # Show max 5 matches
                                st.caption(f"**{i}.** {match['gt_quote'][:80]}...")
                                st.caption(f"   → {match['llm_quote'][:80]}... ({match['similarity']:.0%})")
                            if len(matches) > 5:
                                st.caption(f"... and {len(matches) - 5} more")
                    
                    with col2:
                        st.metric("Unmatched", len(unmatched))
                        if unmatched:
                            for i, item in enumerate(unmatched[:5], 1):  # Show max 5 unmatched
                                st.caption(f"**{i}.** {item['quote'][:100]}...")
                            if len(unmatched) > 5:
                                st.caption(f"... and {len(unmatched) - 5} more")
                    
                    if not matches and not unmatched:
                        st.info("No quotes found in ground truth file.")
                        
        except Exception as e:
            st.error(f"Error processing ground truth file: {e}")
    elif uploaded_file is not None:
        st.warning("Please run LLM extraction first to compare with ground truth.")


if __name__ == "__main__":
    main()

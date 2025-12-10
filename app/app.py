# Enhanced app.py - Add to your existing code
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
from pipeline.csv_export import flatten_to_rows

# Import prompts for editing
from llm.prompts import PROMPTS

# Import natural language query converter
from llm.utils import natural_query_to_pubmed_query



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
    st.title("🧪 PubMed Miner")
    st.caption("Search PubMed articles, fetch PMC full text, run your LLM extractor, and download findings.")

    # ===== Sidebar Configuration =====
    with st.sidebar:
        # NCBI Configuration (needed for search)
        st.header("🔬 NCBI Configuration")
        st.caption("Required for PubMed search")
        
        ncbi_api_key = st.text_input(
            "NCBI API Key",
            value=os.getenv("NCBI_API_KEY", ""),
            type="password",
            help="Get from: https://www.ncbi.nlm.nih.gov/account/settings/"
        )
        
        # Strip whitespace and update environment
        if ncbi_api_key:
            ncbi_api_key = ncbi_api_key.strip()
            os.environ["NCBI_API_KEY"] = ncbi_api_key
        
        st.divider()
        
        # LLM Configuration
        st.header("🤖 LLM Configuration")
        
        # Model selection
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
            help="Choose which LLM to use for extraction"
        )
        
        # API Key input based on selection
        api_key_env_var = None
        api_key = ""
        model_name = ""
        custom_api_url = os.getenv("CUSTOM_LLM_URL", "")
        custom_headers_json = os.getenv("CUSTOM_LLM_HEADERS", "")
        custom_headers_dict: Dict[str, str] = {}
        custom_timeout = int(os.getenv("CUSTOM_LLM_TIMEOUT", "120") or 120)
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
        
        elif "Custom" in model_choice:
            st.info("🎯 **Hackathon Models**: Select from the available models provided through the local proxy.")
            
            # Hackathon models from the capability matrix
            HACKATHON_MODELS = [
                "gpt35",
                "gpt35large",
                "gpt4",
                "gpt4large",
                "gpt4turbo",
                "gpt4o",
                "gpto1",
                "gpto1mini",
                "gpto3",
                "gpto3mini",
                "gpto4mini",
                "gpt41",
                "gpt41mini",
                "gpt41nano",
                "gpt5",
                "gpt5mini",
                "gpt5nano",
                "gemini25pro",
                "gemini25flash",
                "claudeopus4",
                "claudeopus41",
                "claudesonnet4",
                "claudesonnet45",
                "claudesonnet37",
                "claudesonnet35v2",
            ]
            
            # Model selector with auto-populate
            selected_model = st.selectbox(
                "Select Hackathon Model",
                options=[""] + HACKATHON_MODELS,
                index=0,
                help="Choose from the 25 models available through the local proxy. All models support text extraction."
            )
            
            # Auto-populate model name if selected, otherwise allow manual entry
            if selected_model:
                model_name = selected_model
                os.environ["CUSTOM_LLM_MODEL"] = model_name
            else:
                model_name = st.text_input(
                    "Model Name (or select from dropdown above)",
                    value=os.getenv("CUSTOM_LLM_MODEL", ""),
                    help="Enter model name manually if not in the dropdown"
                )
            
            custom_api_url = st.text_input(
                "Local Proxy API URL",
                value=custom_api_url,
                help="The proxy endpoint URL provided by hackathon organizers (e.g., http://localhost:8080/v1/completions or https://proxy.hackathon.local/v1/completions)"
            ).strip()
            if custom_api_url:
                os.environ["CUSTOM_LLM_URL"] = custom_api_url
            
            api_key = st.text_input(
                "API Key (optional)",
                value=os.getenv("CUSTOM_LLM_API_KEY", ""),
                type="password",
                help="Leave blank if the proxy does not require authentication."
            )
            api_key_env_var = "CUSTOM_LLM_API_KEY"
            custom_headers_json = st.text_area(
                "Extra HTTP headers (JSON)",
                value=custom_headers_json,
                help="Optional: JSON object of additional headers. Example: {\"X-Org\": \"Team-42\"}"
            )
            parsed_headers: Dict[str, str] = {}
            if custom_headers_json.strip():
                try:
                    parsed_headers = json.loads(custom_headers_json.strip())
                    if not isinstance(parsed_headers, dict):
                        raise ValueError("Headers JSON must be an object")
                except Exception:
                    st.error("Extra HTTP headers must be a valid JSON object of key-value pairs.")
                    parsed_headers = {}
            custom_headers_dict = parsed_headers
            custom_timeout = st.number_input(
                "Request timeout (seconds)",
                min_value=5,
                max_value=600,
                value=custom_timeout,
                help="Adjust if the endpoint is slow."
            )
            os.environ["CUSTOM_LLM_TIMEOUT"] = str(custom_timeout)
            if custom_headers_dict:
                os.environ["CUSTOM_LLM_HEADERS"] = json.dumps(custom_headers_dict)
        
        # Strip whitespace from API key and persist in environment for backend usage
        if api_key:
            api_key = api_key.strip()
            if api_key_env_var:
                os.environ[api_key_env_var] = api_key
        
        st.divider()

    # ===== Prompt Editor Section =====
    with st.expander("📝 **Edit Extraction Prompt**", expanded=False):
        # Load current prompt
        current_prompt = PROMPTS.analyst_prompt
        
        # Single editor for full prompt
        edited_prompt = st.text_area(
            "Full Prompt",
            value=current_prompt,
            height=500,
            help="Edit the complete prompt. Make sure to keep the {TEXT} placeholder at the end."
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("💾 Save Changes"):
                PROMPTS.analyst_prompt = edited_prompt
                st.success("✅ Prompt updated! Will be used for next extraction.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default"):
                from llm.prompts import AnalystPrompts
                default_prompts = AnalystPrompts()
                PROMPTS.analyst_prompt = default_prompts.analyst_prompt
                st.success("✅ Reset to default prompt.")
                st.rerun()
        
        # Debug preview section
        with st.expander("🔍 Preview Full Prompt", expanded=False):
            # Preview with sample text
            sample_text = "This is a sample paper text. It contains information about mutations like A226V in the E1 protein of Chikungunya virus. The mutation affects viral transmission."
            preview_prompt = edited_prompt.replace("{TEXT}", sample_text)
            
            st.code(preview_prompt, language="text")

    # ===== Natural Language Query Section =====
    st.divider()
    st.subheader("🔍 Search PubMed")
    
    # Add tabs for Natural Language vs Manual Query
    query_tab1, query_tab2 = st.tabs(["✨ Natural Language Query", "⚙️ Manual Boolean Query"])
    
    with query_tab1:
        nl_query = st.text_area(
            "Enter your research question in natural language",
            height=120,
            placeholder="Example: Find papers about dengue virus mutations that affect E protein binding and immune response",
            help="Describe what you're looking for in plain English. The AI will convert it to a PubMed Boolean query.",
            key="nl_query_input"
        )
        
        col_convert1, col_convert2, col_convert3 = st.columns([1, 2, 1])
        with col_convert1:
            convert_button = st.button("🪄 Convert to PubMed Query", type="primary", use_container_width=True)
        
        if convert_button:
            if not nl_query.strip():
                st.warning("⚠️ Please enter a natural language query first.")
            else:
                # Check if API key is available
                api_key = api_key.strip() if api_key else ""
                if not api_key and "Custom" not in model_choice:
                    st.error(f"⚠️ Please enter your {model_choice} API key in the sidebar first!")
                else:
                    try:
                        with st.spinner(f"🤖 Converting your query using {model_choice}..."):
                            # Build LLM metadata
                            llm_meta = {
                                "model_choice": model_choice,
                                "model_name": model_name,
                                "api_key": api_key,
                            }
                            
                            if "Custom" in model_choice:
                                if not custom_api_url:
                                    st.error("⚠️ Please provide the Custom LLM API URL in the sidebar first!")
                                    st.stop()
                                llm_meta["api_url"] = custom_api_url
                                llm_meta["timeout"] = custom_timeout
                                if custom_headers_dict:
                                    llm_meta["extra_headers"] = custom_headers_dict
                            
                            # Convert natural language to PubMed query
                            converted_query = natural_query_to_pubmed_query(nl_query.strip(), llm_meta)
                            
                            # Store in session state
                            st.session_state["converted_query"] = converted_query
                            st.session_state["nl_query_used"] = nl_query.strip()
                            
                            st.success("✅ Query converted successfully!")
                            
                    except Exception as e:
                        st.error(f"❌ Error converting query:\n\n```\n{str(e)}\n```\n\n"
                                f"**Troubleshooting:**\n"
                                f"1. Check your API key is correct\n"
                                f"2. Verify your internet connection\n"
                                f"3. Try simplifying your query\n"
                                f"4. Check the LLM service status")
        
        # Display converted query
        if st.session_state.get("converted_query"):
            st.markdown("---")
            st.markdown("### 📋 Converted PubMed Query")
            
            # Show original NL query
            with st.expander("💬 Original Natural Language Query", expanded=False):
                st.info(st.session_state.get("nl_query_used", ""))
            
            # Show converted query in an editable field
            edited_query = st.text_area(
                "PubMed Boolean Query (you can edit before searching)",
                value=st.session_state["converted_query"],
                height=100,
                help="This is the converted query. You can edit it before searching if needed.",
                key="converted_query_display"
            )
            
            st.session_state["query_to_search"] = edited_query
            
            st.success("✅ Ready to search! Scroll down to configure date range and click **Search PubMed**.")
    
    with query_tab2:
        manual_query = st.text_area(
            "Enter PubMed Boolean Query",
            height=100,
            placeholder='e.g., dengue[MeSH Terms] AND mutation[Text Word]',
            help="Enter a PubMed Boolean query directly",
            key="manual_query_input"
        )
        
        if manual_query.strip():
            st.session_state["query_to_search"] = manual_query.strip()
            st.success("✅ Query entered. Scroll down to configure date range and click **Search PubMed**.")
    
    # ===== Search Configuration & Execution =====
    st.markdown("---")
    st.subheader("1) Configure Search Parameters")
    
    # Show NCBI API status in main area
    if not os.getenv("NCBI_API_KEY"):
        st.info("💡 **Tip:** Add your NCBI API key in the sidebar (👈) to increase rate limits from 3 to 10 requests/second.")
    
    # Get the query from session state (either converted or manual)
    query = st.session_state.get("query_to_search", "")
    
    # Display current query if available
    if query:
        st.info(f"**Current Query:** `{query}`")
    
    # Toggle for review papers only
    reviews_only = st.checkbox(
        "🔍 Restrict to Review articles only",
        value=True,
        help="If checked, only search for Review articles. If unchecked, search all article types."
    )

    st.subheader("2) Choose publication date range & search")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        mindate = st.text_input(
            "Start Date (MM/YYYY)", 
            value="01/2005",
            placeholder="MM/YYYY",
            help="Enter start date in MM/YYYY format (e.g., 01/2020)"
        )
    with colB:
        maxdate = st.text_input(
            "End Date (MM/YYYY)", 
            value="12/2025",
            placeholder="MM/YYYY",
            help="Enter end date in MM/YYYY format (e.g., 12/2025)"
        )
    with colC:
        sort = st.selectbox("Sort", ["relevance", "pub+date"], index=0)
    with colD:
        cap = st.slider("Max records", 0, 500, 100, 100)

    search_button_text = "🔎 Search PubMed (reviews)" if reviews_only else "🔎 Search PubMed (all articles)"
    go = st.button(search_button_text, disabled=(not query or not query.strip()))
    
    if not query or not query.strip():
        st.warning("⚠️ Please enter a query above using either Natural Language or Manual Boolean Query tabs.")
    
    if go:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # Validate and convert MM/YYYY to YYYY/MM format for NCBI
            try:
                # Parse MM/YYYY input
                min_parts = mindate.strip().split("/")
                max_parts = maxdate.strip().split("/")
                
                if len(min_parts) != 2 or len(max_parts) != 2:
                    st.error("❌ Invalid date format. Please use MM/YYYY (e.g., 01/2020)")
                    st.stop()
                
                # Convert MM/YYYY to YYYY/MM for NCBI API
                mindate_formatted = f"{min_parts[1]}/{min_parts[0]}"
                maxdate_formatted = f"{max_parts[1]}/{max_parts[0]}"
                
                # Create date objects for overlap checking
                start_date = date(int(min_parts[1]), int(min_parts[0]), 1)
                # Last day of month for end date
                end_year, end_month = int(max_parts[1]), int(max_parts[0])
                last_day = calendar.monthrange(end_year, end_month)[1]
                end_date = date(end_year, end_month, last_day)
                
            except Exception as e:
                st.error(f"❌ Invalid date format: {e}. Please use MM/YYYY (e.g., 01/2020)")
                st.stop()
            
            try:
                search_type = "reviews" if reviews_only else "all articles"
                with st.spinner(f"Searching PubMed ({search_type})…"):
                    if reviews_only:
                        pmids = esearch_reviews(query.strip(), mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    else:
                        pmids = esearch_all(query.strip(), mindate=mindate_formatted, maxdate=maxdate_formatted, sort=sort, cap=cap)
                    
                    if not pmids:
                        st.warning(f"❌ **No results found for your query.**\n\n"
                                  f"**Your query:** `{query.strip()}`\n\n"
                                  f"**Tips:**\n"
                                  f"- Try broader search terms\n"
                                  f"- Check spelling\n"
                                  f"- Expand date range (currently {start_date} to {end_date})\n"
                                  f"- Try a simpler query (e.g., just 'dengue mutation')")
                        for k in ["hits_df", "hits_pmids", "selected_pmids"]:
                            if k in st.session_state:
                                del st.session_state[k]
                    else:
                        st.info(f"✅ Found {len(pmids)} PMIDs from NCBI. Fetching metadata...")
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
                            st.warning(f"⚠️ **Found {len(pmids)} results, but none match your date range.**\n\n"
                                      f"Date range: {start_date} to {end_date}\n\n"
                                      f"Try expanding the date range above.")
                        else:
                            df_hits["PMID"] = df_hits["PMID"].astype(str)
                            _persist("hits_df", df_hits.to_dict("records"))
                            _persist("hits_pmids", df_hits["PMID"].tolist())
                            st.session_state["selected_pmids"] = []
                            st.success(f"✅ Found {len(df_hits)} results. See 'Results' below to select papers.")
            
            except Exception as e:
                st.error(f"❌ **Error searching PubMed:**\n\n"
                        f"```\n{str(e)}\n```\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Check your internet connection\n"
                        f"2. Verify NCBI API key in `.env` file\n"
                        f"3. Try again in a few seconds (rate limiting)\n"
                        f"4. Simplify your query")

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
        
        st.markdown("##### Select PMIDs to LLM extract")
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

    # ===== Extraction section =====
    st.subheader("3) Run LLM")
    
    run_llm = st.button("🚀 Run LLM", 
                       disabled=(len(st.session_state.get("selected_pmids", [])) == 0))

    if st.button("🗑️ Reset"):
        for k in ["hits_df", "hits_pmids", "batch_papers", "batch_results", "llm_log", 
                  "selected_pmids", "select_all_hits", "pmid_multiselect", 
                  "converted_query", "nl_query_used", "query_to_search"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    if run_llm:
        selected = [str(x) for x in st.session_state.get("selected_pmids", [])]
        pmids = selected
        
        if not pmids:
            st.warning("No PMIDs selected. Pick at least one in Step 2.")
            st.stop()
        
        st.session_state["batch_results"] = {}
        st.session_state["llm_log"] = []
        
        # Validate API key (strip whitespace first)
        api_key = api_key.strip() if api_key else ""
        if not api_key and "Custom" not in model_choice:
            st.error(f"⚠️ Please enter your {model_choice} API key in the sidebar!")
            st.stop()
        if "Custom" in model_choice and not custom_api_url:
            st.error("⚠️ Please provide the Custom LLM API URL in the sidebar before running extraction.")
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
        # Also pass current prompt (may be edited by user)
        llm_meta = {
            "model_choice": model_choice,
            "model_name": model_name,
            "api_key": api_key,  # This will be used as PRIMARY in backend
            "analyst_prompt": PROMPTS.analyst_prompt,  # Current prompt (includes user edits)
        }
        
        if "Custom" in model_choice:
            if not custom_api_url:
                st.warning("Please provide the Custom LLM API URL before running extraction.")
            llm_meta["api_url"] = custom_api_url
            llm_meta["timeout"] = custom_timeout
            if custom_headers_dict:
                llm_meta["extra_headers"] = custom_headers_dict
            # Auto-detect OpenAI-compatible endpoints
            if custom_api_url and ("/v1" in custom_api_url or "/openai" in custom_api_url.lower()):
                llm_meta["openai_compatible"] = True

        total = len(ok_pmids_this_run)
        
        # Debug section for first paper only
        debug_expander = None
        if ok_pmids_this_run:
            debug_expander = st.expander("🔍 Debug: Prompt & Text Preview (First Paper)", expanded=False)
        
        for i, pmid in enumerate(ok_pmids_this_run, start=1):
            title = papers[pmid].get("title") or ""
            pmcid = papers[pmid].get("pmcid") or ""
            log_line = f"[{i}/{total}] Analyzing PMID {pmid} ({pmcid}) – {title[:80]}"
            llm_log.append(log_line)
            _persist("llm_log", llm_log)
            log_box.code("\n".join(llm_log[-20:]), language="text")
            
            # Enable raw LLM capture for debug
            if i == 1:
                llm_meta["debug_raw"] = True
            
            try:
                single_dict = analyze_texts(
                    {pmid: papers[pmid]},
                    chunk_chars=16000, 
                    overlap_chars=500,
                    delay_ms=400, 
                    min_confidence=0.0, 
                    require_mut_quote=True,
                    llm_meta=llm_meta,
                )
                
                # Show raw LLM JSON output and filtered result
                if i == 1 and debug_expander and pmid in single_dict:
                    result_data = single_dict[pmid]
                    
                    # Check for raw LLM responses in the result
                    raw_responses = None
                    if "result" in result_data and isinstance(result_data["result"], dict):
                        raw_responses = result_data["result"].get("_raw_llm_responses")
                    
                    # Fallback to old raw_llm format if available
                    if not raw_responses and "raw_llm" in result_data:
                        raw_responses = [{"chunk": 1, "parsed_json": result_data["raw_llm"]}]
                    
                    with debug_expander:
                        # Show raw LLM output
                        if raw_responses:
                            st.markdown("---")
                            st.markdown("### 📊 Raw LLM JSON Output")
                            st.markdown("*Unfiltered, unprocessed response directly from LLM (before normalization/conversion)*")
                            
                            for resp in raw_responses:
                                chunk_num = resp.get("chunk", 1)
                                total_chunks = resp.get("total_chunks", len(raw_responses))
                                parsed_json = resp.get("parsed_json")
                                
                                if parsed_json is not None:
                                    json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                                    
                                    with st.expander(f"📄 Chunk {chunk_num}/{total_chunks} - Raw LLM Output", expanded=(chunk_num == 1)):
                                        st.code(json_str, language="json")
                                        
                                        # Show summary stats
                                        if isinstance(parsed_json, dict):
                                            if "sequence_features" in parsed_json:
                                                features = parsed_json["sequence_features"]
                                                st.markdown(f"**Features in chunk:** {len(features) if isinstance(features, list) else 0}")
                                            elif isinstance(parsed_json, list):
                                                st.markdown(f"**Features in chunk:** {len(parsed_json)}")
                                        
                                        # Also show raw string if available
                                        raw_string = resp.get("raw_response")
                                        if raw_string:
                                            with st.expander("🔤 Raw String Response", expanded=False):
                                                st.code(raw_string[:5000] + ("..." if len(raw_string) > 5000 else ""), language="text")
                                                if len(raw_string) > 5000:
                                                    st.caption(f"Showing first 5000 characters of {len(raw_string):,} total")
                        
                        # Show filtered/cleaned result
                        if "result" in result_data:
                            cleaned_result = result_data["result"]
                            st.markdown("---")
                            st.markdown("### ✅ Filtered & Processed JSON Output")
                            st.markdown("*After normalization, conversion, confidence scoring, and filtering*")
                            
                            filtered_json = json.dumps(cleaned_result, indent=2, ensure_ascii=False)
                            
                            with st.expander("📋 Full Filtered JSON", expanded=True):
                                st.code(filtered_json, language="json")
                            
                            # Show summary stats
                            if isinstance(cleaned_result, dict) and "sequence_features" in cleaned_result:
                                filtered_features = cleaned_result["sequence_features"]
                                feature_count = len(filtered_features) if isinstance(filtered_features, list) else 0
                                st.markdown(f"**Features after filtering:** {feature_count}")
                                
                                # Count by type
                                if isinstance(filtered_features, list) and len(filtered_features) > 0:
                                    type_counts = {}
                                    for feat in filtered_features:
                                        if isinstance(feat, dict):
                                            feat_type = feat.get("target_type") or feat.get("mutation") or "unknown"
                                            type_counts[feat_type] = type_counts.get(feat_type, 0) + 1
                                    
                                    if type_counts:
                                        st.markdown("**Breakdown by type:**")
                                        for ftype, count in sorted(type_counts.items()):
                                            st.markdown(f"  - {ftype}: {count}")
                                
                                # Show sample feature
                                if isinstance(filtered_features, list) and len(filtered_features) > 0:
                                    st.markdown("**Sample filtered feature:**")
                                    sample_feature = filtered_features[0]
                                    sample_json = json.dumps(sample_feature, indent=2, ensure_ascii=False)
                                    st.code(sample_json, language="json")
                
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
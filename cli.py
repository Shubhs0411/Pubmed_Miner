#!/usr/bin/env python3
"""
PubMed Miner CLI - Terminal version of the app
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import date
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from services.pubmed import (
    esearch_reviews, esearch_all, esummary, parse_pubdate_interval, overlaps
)
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts
from pipeline.simple_csv import raw_to_csv
from llm.prompts import PROMPTS
from llm.utils import natural_query_to_pubmed_query


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text: str):
    """Print info message."""
    print(f"ℹ️  {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")


def get_ncbi_api_key() -> str:
    """Get NCBI API key from env or prompt."""
    key = os.getenv("NCBI_API_KEY", "")
    if not key:
        key = input("Enter NCBI API Key (or press Enter to skip): ").strip()
        if key:
            os.environ["NCBI_API_KEY"] = key
    return key


def get_llm_config() -> Dict:
    """Get LLM configuration interactively."""
    print_header("LLM Configuration")
    
    models = [
        "Gemini (Google)",
        "GPT-4o (OpenAI)",
        "Claude (Anthropic)",
        "Llama (Groq)",
        "Custom (Hackathon)",
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    choice = input("\nSelect model (1-5) [1]: ").strip() or "1"
    try:
        idx = int(choice) - 1
        model_choice = models[idx]
    except (ValueError, IndexError):
        model_choice = models[0]
    
    print(f"\nSelected: {model_choice}")
    
    config = {
        "model_choice": model_choice,
        "model_name": "",
        "api_key": "",
    }
    
    if "Gemini" in model_choice:
        config["api_key"] = os.getenv("GEMINI_API_KEY", "") or input("Enter Gemini API Key: ").strip()
        config["model_name"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        model_name = input(f"Model name [{config['model_name']}]: ").strip()
        if model_name:
            config["model_name"] = model_name
    elif "GPT-4o" in model_choice:
        config["api_key"] = os.getenv("OPENAI_API_KEY", "") or input("Enter OpenAI API Key: ").strip()
        config["model_name"] = os.getenv("OPENAI_MODEL", "gpt-4o")
        model_name = input(f"Model name [{config['model_name']}]: ").strip()
        if model_name:
            config["model_name"] = model_name
    elif "Claude" in model_choice:
        config["api_key"] = os.getenv("ANTHROPIC_API_KEY", "") or input("Enter Anthropic API Key: ").strip()
        config["model_name"] = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        model_name = input(f"Model name [{config['model_name']}]: ").strip()
        if model_name:
            config["model_name"] = model_name
    elif "Llama" in model_choice or "Groq" in model_choice:
        config["api_key"] = os.getenv("GROQ_API_KEY", "") or input("Enter Groq API Key: ").strip()
        groq_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        print("\nAvailable Groq models:")
        for i, m in enumerate(groq_models, 1):
            print(f"  {i}. {m}")
        choice = input(f"Select model (1-3) [1]: ").strip() or "1"
        try:
            config["model_name"] = groq_models[int(choice) - 1]
        except (ValueError, IndexError):
            config["model_name"] = groq_models[0]
    elif "Custom" in model_choice:
        config["api_url"] = os.getenv("CUSTOM_LLM_URL", "") or input("Enter Custom API URL: ").strip()
        config["api_key"] = input("Enter API Key (optional): ").strip()
        config["timeout"] = int(os.getenv("CUSTOM_LLM_TIMEOUT", "60"))
        timeout = input(f"Timeout (seconds) [{config['timeout']}]: ").strip()
        if timeout:
            config["timeout"] = int(timeout)
        if config["api_url"] and ("/v1" in config["api_url"] or "/openai" in config["api_url"].lower()):
            config["openai_compatible"] = True
    
    return config


def search_pubmed() -> List[str]:
    """Search PubMed and return PMIDs."""
    print_header("Section 1: PubMed Search")
    
    print("Search options:")
    print("  1. Natural language query")
    print("  2. Boolean query")
    
    search_type = input("\nSelect search type (1-2) [1]: ").strip() or "1"
    
    if search_type == "1":
        query = input("\nEnter natural language query: ").strip()
        if not query:
            print_error("Query cannot be empty")
            return []
        
        # Convert to PubMed query
        pubmed_query = natural_query_to_pubmed_query(query)
        print_info(f"Converted query: {pubmed_query}")
    else:
        pubmed_query = input("\nEnter boolean query: ").strip()
        if not pubmed_query:
            print_error("Query cannot be empty")
            return []
    
    # Date range
    print("\nDate range (optional):")
    mindate = input("Start date (YYYY-MM-DD or YYYY): ").strip() or None
    maxdate = input("End date (YYYY-MM-DD or YYYY): ").strip() or None
    
    # Sort
    sort_options = ["relevance", "pub_date", "first_author", "journal"]
    print("\nSort options:")
    for i, opt in enumerate(sort_options, 1):
        print(f"  {i}. {opt}")
    sort_choice = input("Select sort (1-4) [1]: ").strip() or "1"
    try:
        sort = sort_options[int(sort_choice) - 1]
    except (ValueError, IndexError):
        sort = "relevance"
    
    # Max papers
    max_papers = input("Max papers to fetch [200]: ").strip() or "200"
    try:
        max_papers = int(max_papers)
    except ValueError:
        max_papers = 200
    
    # Review papers only
    review_only = input("Review papers only? (y/n) [n]: ").strip().lower() == "y"
    
    print_info(f"Searching PubMed with query: {pubmed_query}")
    
    try:
        if review_only:
            pmids = esearch_reviews(
                pubmed_query,
                mindate=mindate,
                maxdate=maxdate,
                sort=sort,
                retmax=max_papers
            )
        else:
            pmids = esearch_all(
                pubmed_query,
                mindate=mindate,
                maxdate=maxdate,
                sort=sort,
                retmax=max_papers
            )
        
        print_success(f"Found {len(pmids)} PMIDs")
        return pmids
    except Exception as e:
        print_error(f"Search failed: {e}")
        return []


def display_results(pmids: List[str]) -> List[str]:
    """Display search results and let user select PMIDs."""
    if not pmids:
        print_warning("No PMIDs to display")
        return []
    
    print_header("Section 2: Select PMIDs")
    
    # Get summaries
    print_info("Fetching paper summaries...")
    summaries = esummary(pmids)
    
    # Display results
    print(f"\n{'Index':<8} {'PMID':<12} {'Title':<50}")
    print("-" * 70)
    for i, pmid in enumerate(pmids[:50], 1):  # Show first 50
        info = summaries.get(pmid, {})
        title = (info.get("title") or "")[:47] + "..." if len(info.get("title", "")) > 50 else (info.get("title") or "")
        print(f"{i:<8} {pmid:<12} {title}")
    
    if len(pmids) > 50:
        print(f"\n... and {len(pmids) - 50} more papers")
    
    # Select PMIDs
    print("\nSelect PMIDs to extract:")
    print("  - Enter comma-separated indices (e.g., 1,3,5)")
    print("  - Enter 'all' to select all")
    print("  - Enter range (e.g., 1-10)")
    
    selection = input("\nSelection: ").strip()
    
    selected_pmids = []
    if selection.lower() == "all":
        selected_pmids = pmids
    elif "-" in selection:
        # Range selection
        try:
            start, end = map(int, selection.split("-"))
            selected_pmids = [pmids[i-1] for i in range(start, end+1) if 1 <= i <= len(pmids)]
        except ValueError:
            print_error("Invalid range format")
            return []
    else:
        # Comma-separated indices
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            selected_pmids = [pmids[i-1] for i in indices if 1 <= i <= len(pmids)]
        except ValueError:
            print_error("Invalid selection format")
            return []
    
    print_success(f"Selected {len(selected_pmids)} PMIDs")
    return selected_pmids


def run_llm_extraction(pmids: List[str], llm_config: Dict) -> Dict[str, Dict]:
    """Run LLM extraction on selected PMIDs."""
    print_header("Section 3: LLM Extraction")
    
    if not pmids:
        print_warning("No PMIDs selected")
        return {}
    
    # Fetch full texts
    print_info(f"Fetching PMC full texts for {len(pmids)} PMIDs...")
    papers = fetch_all_fulltexts(pmids, delay_ms=150)
    
    # Count results
    fetched = [pid for pid, info in papers.items() if info.get("status") == "ok"]
    no_pmc = [pid for pid, info in papers.items() if info.get("status") == "no_pmc_fulltext"]
    errors = [pid for pid, info in papers.items() if info.get("status") == "error"]
    
    print_success(f"PMC texts: ✅ {len(fetched)} fetched | ⚠️ {len(no_pmc)} no PMC | ❌ {len(errors)} errors")
    
    if len(fetched) == 0:
        print_error("No papers with full text available. Cannot proceed with extraction.")
        return {}
    
    # LLM extraction
    print_info(f"Running LLM extraction on {len(fetched)} papers...")
    
    llm_meta = {
        "model_choice": llm_config["model_choice"],
        "model_name": llm_config["model_name"],
        "api_key": llm_config["api_key"],
        "analyst_prompt": PROMPTS.analyst_prompt,
    }
    
    if "Custom" in llm_config.get("model_choice", ""):
        llm_meta["api_url"] = llm_config.get("api_url", "")
        llm_meta["timeout"] = llm_config.get("timeout", 60)
        if llm_meta["api_url"] and ("/v1" in llm_meta["api_url"] or "/openai" in llm_meta["api_url"].lower()):
            llm_meta["openai_compatible"] = True
    
    # Run extraction
    batch_results = {}
    for i, pmid in enumerate(fetched, 1):
        print(f"[{i}/{len(fetched)}] Processing PMID {pmid}...", end=" ", flush=True)
        try:
            single_dict = analyze_texts(
                {pmid: papers[pmid]},
                chunk_chars=16000,
                overlap_chars=500,
                delay_ms=400,
                min_confidence=0.0,
                require_mut_quote=False,
                llm_meta=llm_meta,
            )
            batch_results.update(single_dict)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print_success("LLM extraction complete!")
    return batch_results, papers


def export_csv(batch_results: Dict[str, Dict], papers: Dict[str, Dict], output_file: str = "output.csv"):
    """Export results to CSV."""
    print_header("CSV Export")
    
    apply_filters = input("Apply filters? (y/n) [y]: ").strip().lower() != "n"
    
    print_info("Converting to CSV...")
    csv_df = raw_to_csv(batch_results, apply_filters=apply_filters, papers=papers)
    
    if not csv_df.empty:
        print_success(f"Generated CSV with {len(csv_df)} rows ({'filtered' if apply_filters else 'unfiltered'})")
        csv_df.to_csv(output_file, index=False)
        print_success(f"Saved to: {output_file}")
        
        # Show preview
        print("\nPreview (first 5 rows):")
        print(csv_df.head().to_string())
    else:
        print_warning("No features extracted. CSV is empty.")


def validate_with_ground_truth(batch_results: Dict[str, Dict], gt_file: str):
    """Validate results against ground truth."""
    print_header("Section 4: Validation")
    
    if not Path(gt_file).exists():
        print_error(f"Ground truth file not found: {gt_file}")
        return
    
    try:
        gt_df = pd.read_csv(gt_file)
        
        if "quote" not in gt_df.columns:
            print_error("Ground truth CSV must contain 'quote' column")
            return
        
        # Extract LLM quotes
        llm_quotes = []
        for pmid, entry in batch_results.items():
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
            print_warning("No quotes found in LLM output")
            return
        
        # Match quotes
        from difflib import SequenceMatcher
        
        matches = []
        matched_gt_quotes = set()
        
        for _, gt_row in gt_df.iterrows():
            gt_pmid = str(gt_row.get("pmid", "")).strip()
            gt_quote = str(gt_row.get("quote", "")).strip()
            
            if not gt_quote:
                continue
            
            best_match = None
            best_score = 0.0
            
            for llm_item in llm_quotes:
                if gt_pmid and str(llm_item["pmid"]) != gt_pmid:
                    continue
                
                llm_quote = llm_item["quote"]
                similarity = SequenceMatcher(None, gt_quote.lower(), llm_quote.lower()).ratio()
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = {
                        "gt_quote": gt_quote,
                        "llm_quote": llm_quote,
                        "similarity": similarity,
                        "pmid": llm_item["pmid"],
                    }
            
            if best_match and best_score > 0.3:
                matches.append(best_match)
                matched_gt_quotes.add(gt_quote)
        
        # Find unmatched
        unmatched = []
        for _, gt_row in gt_df.iterrows():
            gt_quote = str(gt_row.get("quote", "")).strip()
            if gt_quote and gt_quote not in matched_gt_quotes:
                unmatched.append({"quote": gt_quote, "pmid": str(gt_row.get("pmid", "")).strip()})
        
        # Display results
        print(f"\nMatched: {len(matches)} | Unmatched: {len(unmatched)}")
        
        if matches:
            print("\n--- Matched Quotes ---")
            for i, match in enumerate(matches[:10], 1):
                print(f"\nMatch {i} (Similarity: {match['similarity']:.2%}):")
                print(f"  GT: {match['gt_quote'][:100]}...")
                print(f"  LLM: {match['llm_quote'][:100]}...")
        
        if unmatched:
            print("\n--- Unmatched Ground Truth Quotes ---")
            for i, item in enumerate(unmatched[:10], 1):
                print(f"\n{i}. {item['quote'][:100]}...")
    
    except Exception as e:
        print_error(f"Validation failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="PubMed Miner CLI")
    parser.add_argument("--ncbi-key", help="NCBI API key")
    parser.add_argument("--output", "-o", default="output.csv", help="Output CSV file")
    parser.add_argument("--ground-truth", "-g", help="Ground truth CSV file for validation")
    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode (use env vars)")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    print_header("🧬 PubMed Miner CLI")
    
    # Get API keys
    if args.ncbi_key:
        os.environ["NCBI_API_KEY"] = args.ncbi_key
    else:
        get_ncbi_api_key()
    
    # Get LLM config
    if args.non_interactive:
        # Use defaults from env
        llm_config = {
            "model_choice": os.getenv("LLM_MODEL_CHOICE", "Gemini (Google)"),
            "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            "api_key": os.getenv("GEMINI_API_KEY", ""),
        }
    else:
        llm_config = get_llm_config()
    
    # Search PubMed
    pmids = search_pubmed()
    if not pmids:
        print_error("No PMIDs found. Exiting.")
        return
    
    # Select PMIDs
    selected_pmids = display_results(pmids)
    if not selected_pmids:
        print_error("No PMIDs selected. Exiting.")
        return
    
    # Run LLM extraction
    batch_results, papers = run_llm_extraction(selected_pmids, llm_config)
    if not batch_results:
        print_error("No results from LLM extraction. Exiting.")
        return
    
    # Export CSV
    export_csv(batch_results, papers, args.output)
    
    # Validation (optional)
    if args.ground_truth:
        validate_with_ground_truth(batch_results, args.ground_truth)
    
    print_success("Done!")


if __name__ == "__main__":
    main()


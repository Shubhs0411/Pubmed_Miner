#!/usr/bin/env python3
"""
PubMed Miner CLI - Terminal version of the app
Usage examples:
  python pubmed_cli.py "influenza[Title] AND mutation[Text Word]"
  python pubmed_cli.py --query "influenza[Title] AND mutation[Text Word]" --max-papers 100
  python pubmed_cli.py --query "dengue[Title] AND protein[Text Word]" --output results.csv
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import re
from datetime import date, datetime
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


def is_simple_disease_query(query: str) -> bool:
    """Check if query is just a disease name (no boolean operators)."""
    query_lower = query.lower().strip()
    # Check for common boolean operators
    boolean_indicators = [" and ", " or ", " not ", "[title]", "[text word]", "[mesh]", "(", ")"]
    return not any(indicator in query_lower for indicator in boolean_indicators)


def construct_full_query(disease_name: str) -> str:
    """Construct full PubMed query from disease name."""
    # Standard text word terms
    text_word_terms = [
        "active site",
        "interaction",
        "disease",
        "mutation",
        "virulence",
        "activity",
        "assembly",
        "binding",
        "replication",
        "infectivity",
        "viral synthesis"
    ]
    
    # Build OR clause for text words
    text_word_clause = " OR ".join([f"({term}[Text Word])" for term in text_word_terms])
    
    # Construct full query
    full_query = f"(({disease_name}[Title]) AND (protein)) AND (({text_word_clause}))"
    
    return full_query


def sanitize_filename(name: str) -> str:
    """Sanitize disease name for use in filename."""
    # Convert to lowercase
    name = name.lower()
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove special characters, keep only alphanumeric and underscores
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def ensure_output_dir() -> str:
    """Ensure output directory exists and return its path."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


def generate_output_filename(disease_name: Optional[str] = None) -> str:
    """Generate output filename: output/disease_LLM_findings_timestamp.csv"""
    output_dir = ensure_output_dir()
    
    if disease_name:
        sanitized = sanitize_filename(disease_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sanitized}_LLM_findings_{timestamp}.csv"
    else:
        # Fallback to default
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_LLM_findings_{timestamp}.csv"
    
    return str(Path(output_dir) / filename)


def get_llm_config_from_env() -> Dict:
    """Get LLM configuration from environment variables only."""
    # Determine model choice from env
    model_choice = os.getenv("LLM_MODEL_CHOICE", "Gemini (Google)")
    
    # If not set, try to infer from available API keys
    if model_choice == "Gemini (Google)":
        if not os.getenv("GEMINI_API_KEY"):
            # Try other models
            if os.getenv("OPENAI_API_KEY"):
                model_choice = "GPT-4o (OpenAI)"
            elif os.getenv("ANTHROPIC_API_KEY"):
                model_choice = "Claude (Anthropic)"
            elif os.getenv("GROQ_API_KEY"):
                model_choice = "Llama (Groq)"
            elif os.getenv("CUSTOM_LLM_URL"):
                model_choice = "Custom (Hackathon)"
    
    config = {
        "model_choice": model_choice,
        "model_name": "",
        "api_key": "",
    }
    
    if "Gemini" in model_choice:
        config["api_key"] = os.getenv("GEMINI_API_KEY", "")
        config["model_name"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    elif "GPT-4o" in model_choice or "OpenAI" in model_choice:
        config["api_key"] = os.getenv("OPENAI_API_KEY", "")
        config["model_name"] = os.getenv("OPENAI_MODEL", "gpt-4o")
    elif "Claude" in model_choice or "Anthropic" in model_choice:
        config["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
        config["model_name"] = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    elif "Llama" in model_choice or "Groq" in model_choice:
        config["api_key"] = os.getenv("GROQ_API_KEY", "")
        config["model_name"] = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    elif "Custom" in model_choice:
        config["api_url"] = os.getenv("CUSTOM_LLM_URL", "")
        config["api_key"] = os.getenv("CUSTOM_LLM_API_KEY", "")
        config["model_name"] = os.getenv("CUSTOM_LLM_MODEL", "custom")
        config["timeout"] = int(os.getenv("CUSTOM_LLM_TIMEOUT", "60"))
        if config["api_url"] and ("/v1" in config["api_url"] or "/openai" in config["api_url"].lower()):
            config["openai_compatible"] = True
    
    return config


def search_pubmed(
    query: str,
    mindate: Optional[str] = None,
    maxdate: Optional[str] = None,
    sort: str = "relevance",
    max_papers: int = 100,
    reviews_only: bool = True,
    open_access_only: bool = True
) -> List[str]:
    """Search PubMed and return PMIDs."""
    print_header("Section 1: PubMed Search")
    
    if not query:
        print_error("Query cannot be empty")
        return []
    
    print_info(f"Search query: {query}")
    print_info(f"Max papers: {max_papers}")
    print_info(f"Reviews only: {reviews_only}")
    print_info(f"Open access only: {open_access_only}")
    
    # Convert dates to YYYY/MM format if needed
    mindate_formatted = None
    maxdate_formatted = None
    
    if mindate:
        # Try to parse YYYY-MM-DD or YYYY
        parts = mindate.split("-")
        if len(parts) == 3:  # YYYY-MM-DD
            mindate_formatted = f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1:  # YYYY
            mindate_formatted = f"{parts[0]}/01"
        else:
            mindate_formatted = mindate
    
    if maxdate:
        parts = maxdate.split("-")
        if len(parts) == 3:  # YYYY-MM-DD
            maxdate_formatted = f"{parts[0]}/{parts[1]}"
        elif len(parts) == 1:  # YYYY
            maxdate_formatted = f"{parts[0]}/12"
        else:
            maxdate_formatted = maxdate
    
    try:
        if reviews_only:
            pmids = esearch_reviews(
                query,
                mindate=mindate_formatted,
                maxdate=maxdate_formatted,
                sort=sort,
                retmax=max_papers,
                open_access_only=open_access_only
            )
        else:
            pmids = esearch_all(
                query,
                mindate=mindate_formatted,
                maxdate=maxdate_formatted,
                sort=sort,
                retmax=max_papers,
                open_access_only=open_access_only
            )
        
        print_success(f"Found {len(pmids)} PMIDs")
        return pmids
    except Exception as e:
        print_error(f"Search failed: {e}")
        return []


def run_llm_extraction(
    pmids: List[str],
    llm_config: Dict,
    chunk_chars: int = 16000,
    overlap_chars: int = 500,
    delay_ms: int = 400,
    min_confidence: float = 0.0,
    paper_pause_sec: float = 2.0
) -> tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Run LLM extraction on selected PMIDs."""
    print_header("Section 3: LLM Extraction")
    
    if not pmids:
        print_warning("No PMIDs selected")
        return {}, {}
    
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
        return {}, {}
    
    # LLM extraction
    print_info(f"Running LLM extraction on {len(fetched)} papers...")
    
    llm_meta = {
        "model_choice": llm_config["model_choice"],
        "model_name": llm_config["model_name"],
        "api_key": llm_config["api_key"],
        "analyst_prompt": PROMPTS.analyst_prompt,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "delay_ms": delay_ms,
        "min_confidence": min_confidence,
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
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
                delay_ms=delay_ms,
                min_confidence=min_confidence,
                require_mut_quote=False,
                llm_meta=llm_meta,
                paper_pause_sec=paper_pause_sec,
            )
            batch_results.update(single_dict)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print_success("LLM extraction complete!")
    return batch_results, papers


def export_csv(
    batch_results: Dict[str, Dict],
    papers: Dict[str, Dict],
    output_file: str = "output.csv",
    apply_filters: bool = True
):
    """Export results to CSV."""
    print_header("CSV Export")
    
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
    parser = argparse.ArgumentParser(
        description="PubMed Miner CLI - Extract sequence features from PubMed papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with positional query
  python pubmed_cli.py "influenza[Title] AND mutation[Text Word]"
  
  # With max papers limit
  python pubmed_cli.py --query "dengue[Title] AND protein[Text Word]" --max-papers 50
  
  # With custom output file
  python pubmed_cli.py "influenza[Title]" --output results.csv
  
  # With date range
  python pubmed_cli.py "influenza[Title]" --mindate 2020-01-01 --maxdate 2024-12-31
        """
    )
    
    # Query (positional or --query)
    parser.add_argument(
        "query",
        nargs="?",
        help="PubMed Boolean query (e.g., 'influenza[Title] AND mutation[Text Word]')"
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_arg",
        help="PubMed Boolean query (alternative to positional argument)"
    )
    
    # Search parameters
    parser.add_argument(
        "--mindate",
        help="Start date in YYYY-MM-DD or YYYY format (default: None)"
    )
    parser.add_argument(
        "--maxdate",
        help="End date in YYYY-MM-DD or YYYY format (default: None)"
    )
    parser.add_argument(
        "--sort",
        choices=["relevance", "pub_date", "first_author", "journal"],
        default="relevance",
        help="Sort order (default: relevance)"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Maximum papers to fetch (default: 100)"
    )
    parser.add_argument(
        "--no-reviews-only",
        action="store_true",
        help="Include all article types (default: reviews only)"
    )
    parser.add_argument(
        "--open-access-only",
        action="store_true",
        help="Restrict to open access papers only (default: include all papers)"
    )
    
    # LLM parameters
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=16000,
        help="Text chunk size in characters (default: 16000)"
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=500,
        help="Overlap between chunks in characters (default: 500)"
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=400,
        help="Delay between LLM requests in milliseconds (default: 400)"
    )
    parser.add_argument(
        "--paper-pause-sec",
        type=float,
        default=2.0,
        help="Pause between papers in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold 0.0-1.0 (default: 0.0)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV file (default: auto-generated as disease_LLM_findings_timestamp.csv)"
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Don't apply CSV filters (default: apply filters)"
    )
    
    # Other parameters
    parser.add_argument(
        "--ground-truth", "-g",
        help="Ground truth CSV file for validation"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    print_header("🧬 PubMed Miner CLI")
    
    # Get query (positional or --query)
    raw_query = args.query or args.query_arg
    if not raw_query:
        parser.error("Query is required. Provide it as positional argument or use --query")
    
    # Check if it's a simple disease name and construct full query
    disease_name = None
    if is_simple_disease_query(raw_query):
        disease_name = raw_query.strip()
        query = construct_full_query(disease_name)
        print_info(f"Detected disease name: {disease_name}")
        print_info(f"Constructed query: {query}")
    else:
        query = raw_query
        # Try to extract disease name from query for filename
        # Look for pattern like "Disease Name[Title]"
        match = re.search(r'\(?([^[\(]+)\[Title\]', query, re.IGNORECASE)
        if match:
            disease_name = match.group(1).strip()
    
    # Generate output filename (use custom if provided, otherwise auto-generate)
    output_file = args.output if args.output else generate_output_filename(disease_name)
    print_info(f"Output file: {output_file}")
    
    # Check API keys from env (no prompts)
    ncbi_key = os.getenv("NCBI_API_KEY", "")
    if ncbi_key:
        os.environ["NCBI_API_KEY"] = ncbi_key
    else:
        print_warning("NCBI_API_KEY not set in environment. Rate limits may apply.")
    
    # Get LLM config from env only
    llm_config = get_llm_config_from_env()
    if not llm_config.get("api_key"):
        print_error("No LLM API key found in environment. Please set GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, or CUSTOM_LLM_URL")
        return
    
    print_info(f"Using LLM: {llm_config['model_choice']} ({llm_config['model_name']})")
    
    # Ask for dates if not provided
    mindate = args.mindate
    maxdate = args.maxdate
    
    if not mindate:
        mindate_input = input("\nEnter start date (YYYY-MM-DD or YYYY, or press Enter to skip): ").strip()
        mindate = mindate_input if mindate_input else None
    
    if not maxdate:
        maxdate_input = input("Enter end date (YYYY-MM-DD or YYYY, or press Enter to skip): ").strip()
        maxdate = maxdate_input if maxdate_input else None
    
    # Search PubMed (always reviews only, select all)
    pmids = search_pubmed(
        query=query,
        mindate=mindate,
        maxdate=maxdate,
        sort=args.sort,
        max_papers=args.max_papers,
        reviews_only=not args.no_reviews_only,  # Default: True
        open_access_only=args.open_access_only  # Default: False (include all papers)
    )
    
    if not pmids:
        print_error("No PMIDs found. Exiting.")
        return
    
    # Always select all PMIDs (no interactive selection)
    selected_pmids = pmids
    print_info(f"Processing all {len(selected_pmids)} PMIDs")
    
    # Run LLM extraction
    batch_results, papers = run_llm_extraction(
        selected_pmids,
        llm_config,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        delay_ms=args.delay_ms,
        min_confidence=args.min_confidence,
        paper_pause_sec=args.paper_pause_sec
    )
    
    if not batch_results:
        print_error("No results from LLM extraction. Exiting.")
        return
    
    # Export CSV (always apply filters unless --no-filters)
    export_csv(
        batch_results,
        papers,
        output_file=output_file,
        apply_filters=not args.no_filters  # Default: True
    )
    
    # Validation (optional)
    if args.ground_truth:
        validate_with_ground_truth(batch_results, args.ground_truth)
    
    print_success("Done!")


if __name__ == "__main__":
    main()




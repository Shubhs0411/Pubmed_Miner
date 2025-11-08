# pipeline/batch_analyze.py - FIXED VERSION
from __future__ import annotations

import os
import time
from typing import Dict, List

import pandas as pd

from llm.unified import run_on_paper, clean_and_ground
from services.pmc import get_pmc_fulltext_with_meta, get_last_fetch_source
from pipeline.csv_export import flatten_to_rows


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


SAVE_RAW_LLM = _is_truthy(os.getenv("SAVE_RAW_LLM"))


def fetch_all_fulltexts(pmids: List[str],
                        delay_ms: int = 200,
                        retries: int = 3,
                        backoff: float = 0.8) -> Dict[str, Dict]:
    """
    Fetch PMC full text for each PMID with retries.
    Returns: dict[pmid] = { pmcid, title, text, status, error, source }
      - status in {"ok", "no_pmc_fulltext", "error"}
    """
    out: Dict[str, Dict] = {}
    for pmid in pmids:
        entry = {
            "pmid": pmid, 
            "pmcid": None, 
            "title": None, 
            "text": "", 
            "status": "error", 
            "error": None,
            "source": None
        }
        
        attempt = 0
        while attempt < retries:
            try:
                pmcid, text, title = get_pmc_fulltext_with_meta(pmid)
                entry["pmcid"] = pmcid
                entry["title"] = title
                entry["source"] = get_last_fetch_source(pmid)
                
                if not pmcid or not text:
                    entry["status"] = "no_pmc_fulltext"
                else:
                    entry["text"] = text
                    entry["status"] = "ok"
                break  # success
                
            except Exception as e:
                msg = str(e)
                entry["error"] = msg
                attempt += 1

                # Special handling for 403 (server-side throttle)
                if "403" in msg or "Forbidden" in msg:
                    time.sleep(4.0 * attempt)
                else:
                    time.sleep((backoff ** attempt))

                if attempt >= retries:
                    entry["status"] = "error"
                    break

        out[pmid] = entry
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
            
    return out




def analyze_texts(papers: dict,
                  *,
                  chunk_chars: int = 12000,
                  overlap_chars: int = 500,
                  delay_ms: int = 0,
                  min_confidence: float = 0.6,
                  require_mut_quote: bool = True,
                  llm_meta: dict | None = None,
                  paper_pause_sec: float | None = None) -> Dict[str, Dict]:
    """
    Run LLM extraction on each 'ok' paper, then clean+ground.
    Returns dict[pmid] = { status, pmcid, title, result? }
    """
    if paper_pause_sec is None:
        try:
            paper_pause_sec = float(os.getenv("PAPER_PAUSE_SEC", "2.0"))
        except Exception:
            paper_pause_sec = 2.0

    results: Dict[str, Dict] = {}
    
    for pmid, info in papers.items():
        if info.get("status") != "ok":
            results[pmid] = {
                "status": info.get("status"),
                "pmcid": info.get("pmcid"),
                "title": info.get("title"),
                "error": info.get("error"),
            }
            continue

        text = info["text"]
        pmcid = info.get("pmcid")
        title = info.get("title")

        # Pass meta through to the backend
        meta = {
            "pmid": pmid,
            "pmcid": pmcid,
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "delay_ms": delay_ms,
            "min_confidence": min_confidence,  # Pass through for LLM
        }
        if llm_meta:
            meta.update(llm_meta)

        debug_override = meta.pop("debug_raw", None) if "debug_raw" in meta else None
        capture_raw = SAVE_RAW_LLM or _is_truthy(debug_override)

        # Run LLM extraction
        raw = run_on_paper(text, meta=meta)

        # Clean and ground results - NOW applies min_confidence BEFORE final scoring
        cleaned = clean_and_ground(
            raw,
            text,
            restrict_to_paper=True,
            require_mutation_in_quote=require_mut_quote,
            min_confidence=min_confidence,
        )

        if "paper" in cleaned:
            cleaned["paper"]["title"] = title

        results[pmid] = {
            "status": "ok",
            "pmcid": pmcid,
            "title": title,
            "result": cleaned,
        }
        
        if capture_raw:
            results[pmid]["raw_llm"] = raw

        # Gentle pacing between papers
        if paper_pause_sec and paper_pause_sec > 0:
            time.sleep(paper_pause_sec)

    return results



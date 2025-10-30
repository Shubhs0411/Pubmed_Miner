# pipeline/batch_analyze.py
from __future__ import annotations

import os
import time
import re
from typing import Dict, List

import pandas as pd

from services.pmc import get_pmc_fulltext_with_meta, get_last_fetch_source
from llm.gemini import run_on_paper, clean_and_ground


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


SAVE_RAW_LLM = _is_truthy(os.getenv("SAVE_RAW_LLM"))

# Compile regex once for performance
_STANDARD_MUT_RE = re.compile(r"^[A-Z]\d+[A-Z*]$")


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
        }
        if llm_meta:
            meta.update(llm_meta)

        debug_override = meta.pop("debug_raw", None) if "debug_raw" in meta else None
        capture_raw = SAVE_RAW_LLM or _is_truthy(debug_override)

        # Run LLM extraction
        raw = run_on_paper(text, meta=meta)

        # Clean and ground results
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


def _calculate_confidence(feature: dict, quote: str) -> float:
    """
    Calculate confidence weighted by column completeness.
    Higher scores for more complete data.
    """
    score = 0.0
    
    # Evidence quote (most important indicator)
    if quote:
        score += 0.4
    
    # Position information
    pos = feature.get("position")
    if pos is not None:
        try:
            # Validate it's a real position
            if isinstance(pos, int) or (isinstance(pos, str) and pos.strip().isdigit()):
                score += 0.15
        except Exception:
            pass
    
    # Biological context
    if feature.get("virus"):
        score += 0.1
    if feature.get("protein"):
        score += 0.15
    
    # Mutation format quality (standard format like A226V)
    mutation = (feature.get("mutation") or "").strip()
    if mutation and _STANDARD_MUT_RE.match(mutation):
        score += 0.1
    
    # Experimental context
    ctx = feature.get("experiment_context") or {}
    if any(ctx.get(k) for k in ("system", "assay", "temperature")):
        score += 0.1
    
    return min(score, 1.0)


def flatten_to_rows(batch: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert batch LLM outputs into a row-per-finding table.
    Optimized: essential columns only, improved confidence scoring.
    
    Columns:
      pmid, pmcid, title, virus, protein, mutation, position,
      confidence, quote, target_type
    """
    rows: List[Dict] = []
    
    for pmid, entry in batch.items():
        if entry.get("status") != "ok":
            continue
            
        pmcid = entry.get("pmcid")
        title = entry.get("title")
        seq = (entry.get("result", {}) or {}).get("sequence_features", []) or []
        
        for f in seq:
            # Get first quote only
            quotes = [q for q in (f.get("evidence_quotes") or []) 
                     if isinstance(q, str) and q.strip()]
            quote = quotes[0] if quotes else ""
            
            # Enhanced confidence based on completeness
            conf = _calculate_confidence(f, quote)
            
            # Normalize position (may be str or int)
            pos_val = f.get("position")
            try:
                if isinstance(pos_val, str) and pos_val.strip().isdigit():
                    pos_val = int(pos_val.strip())
            except Exception:
                pass  # leave as-is
            
            # Determine target_type
            target_type = f.get("target_type")
            if not target_type:
                target_type = "mutation" if f.get("mutation") else ""
            
            rows.append({
                "pmid": pmid,
                "pmcid": pmcid or "",
                "title": title or "",
                "virus": f.get("virus") or "",
                "protein": f.get("protein") or "",
                "mutation": f.get("mutation") or "",
                "position": pos_val,
                "confidence": conf,
                "quote": quote,
                "target_type": target_type,
            })

    return pd.DataFrame(rows)
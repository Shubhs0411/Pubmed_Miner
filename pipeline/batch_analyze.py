# pipeline/batch_analyze.py - FIXED VERSION
from __future__ import annotations

import os
import time
import re
from typing import Dict, List, Set, Tuple

import pandas as pd

from llm.unified import run_on_paper, clean_and_ground
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


def _normalize_mutation(mut: str) -> str:
    """Normalize mutation strings for deduplication."""
    if not mut:
        return ""
    # Remove whitespace and convert to uppercase
    return mut.strip().upper().replace(" ", "")


def _create_dedup_key(row: dict) -> Tuple[str, str, str, str]:
    """
    Create deduplication key: (pmid, protein, mutation, position)
    This prevents duplicate rows for same mutation in same protein.
    """
    pmid = str(row.get("pmid", "")).strip()
    protein = str(row.get("protein", "")).strip().lower()
    mutation = _normalize_mutation(row.get("mutation", ""))
    position = str(row.get("position", "")).strip()
    
    return (pmid, protein, mutation, position)


def _merge_quotes(quotes: List[str]) -> str:
    """
    Merge multiple quotes intelligently:
    - Keep unique sentences
    - Prefer shorter, more specific quotes
    - Maximum 3 quotes combined
    """
    if not quotes:
        return ""
    
    # Deduplicate
    unique = []
    seen: Set[str] = set()
    for q in quotes:
        normalized = q.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(q.strip())
    
    # Sort by length (prefer concise quotes)
    unique.sort(key=len)
    
    # Take top 3 most concise
    selected = unique[:3]
    
    # Join with separator
    return " | ".join(selected)


def flatten_to_rows(batch: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert batch LLM outputs into a row-per-finding table with DEDUPLICATION.
    
    Key improvements:
    1. Deduplicates by (pmid, protein, mutation, position)
    2. Merges evidence quotes for same finding
    3. Keeps highest confidence when duplicates found
    4. Filters by min_confidence AFTER scoring
    
    Columns:
      pmid, pmcid, title, virus, protein, mutation, position,
      confidence, quote, target_type
    """
    # Collect all features with their confidence scores
    pre_dedup: Dict[Tuple, Dict] = {}
    
    for pmid, entry in batch.items():
        if entry.get("status") != "ok":
            continue
            
        pmcid = entry.get("pmcid")
        title = entry.get("title")
        seq = (entry.get("result", {}) or {}).get("sequence_features", []) or []
        
        for f in seq:
            # Get all quotes
            quotes = [q for q in (f.get("evidence_quotes") or []) 
                     if isinstance(q, str) and q.strip()]
            
            # Calculate confidence
            first_quote = quotes[0] if quotes else ""
            conf = _calculate_confidence(f, first_quote)
            
            # Normalize position
            pos_val = f.get("position")
            try:
                if isinstance(pos_val, str) and pos_val.strip().isdigit():
                    pos_val = int(pos_val.strip())
            except Exception:
                pass
            
            # Determine target_type
            target_type = f.get("target_type")
            if not target_type:
                target_type = "mutation" if f.get("mutation") else ""
            
            # Extract region information if present
            residue_positions = f.get("residue_positions") or []
            region_range = ""
            if residue_positions and isinstance(residue_positions, list):
                ranges = []
                for rp in residue_positions:
                    if isinstance(rp, dict):
                        start = rp.get("start")
                        end = rp.get("end")
                        if start is not None and end is not None:
                            if start == end:
                                ranges.append(str(start))
                            else:
                                ranges.append(f"{start}-{end}")
                if ranges:
                    region_range = "; ".join(ranges)
            
            # Get feature name for non-mutation features
            feature_name = ""
            if not f.get("mutation"):
                feat = f.get("feature") or {}
                feature_name = feat.get("name_or_label") or ""
            
            # Build row
            row = {
                "pmid": pmid,
                "pmcid": pmcid or "",
                "title": title or "",
                "virus": f.get("virus") or "",
                "protein": f.get("protein") or "",
                "mutation": f.get("mutation") or "",
                "feature_name": feature_name,  # NEW: for domains/regions
                "region_range": region_range,  # NEW: e.g., "1-73; 85-207"
                "position": pos_val,
                "confidence": conf,
                "quote": first_quote,  # Will be merged later
                "target_type": target_type,
                "all_quotes": quotes,  # Temporary for merging
            }
            
            # Create dedup key
            key = _create_dedup_key(row)
            
            # Keep best confidence or merge
            if key in pre_dedup:
                existing = pre_dedup[key]
                # Merge quotes
                existing["all_quotes"].extend(quotes)
                # Keep higher confidence
                if conf > existing["confidence"]:
                    existing["confidence"] = conf
            else:
                pre_dedup[key] = row
    
    # Now merge quotes and finalize
    final_rows: List[Dict] = []
    for row in pre_dedup.values():
        # Merge all quotes intelligently
        row["quote"] = _merge_quotes(row["all_quotes"])
        # Remove temporary field
        del row["all_quotes"]
        final_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(final_rows)
    
    # Sort by confidence (highest first) and then by PMID
    if not df.empty:
        df = df.sort_values(["confidence", "pmid"], ascending=[False, True])
        df = df.reset_index(drop=True)
    
    return df
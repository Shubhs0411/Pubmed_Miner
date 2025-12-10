# pipeline/csv_export.py - CSV formulation logic
# Extracted from batch_analyze.py for separation of concerns

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import pandas as pd

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
            
            # Use confidence from feature dict (already calculated in utils.py)
            conf = float(f.get("confidence", 0.0))
            first_quote = quotes[0] if quotes else ""
            
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


__all__ = ["flatten_to_rows"]


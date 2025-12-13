# pipeline/simple_csv.py - Simple CSV export from raw LLM output
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import re


def _extract_position_from_feature(feature_obj: Dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract start, end, or single position from feature object."""
    start_pos = None
    end_pos = None
    single_pos = None
    
    # Try to get from residue_positions first
    residue_positions = feature_obj.get("residue_positions", [])
    if residue_positions and isinstance(residue_positions, list) and len(residue_positions) > 0:
        first_pos = residue_positions[0]
        if isinstance(first_pos, dict):
            start_pos = first_pos.get("start")
            end_pos = first_pos.get("end")
            if start_pos is not None:
                try:
                    start_pos = int(start_pos)
                except (ValueError, TypeError):
                    start_pos = None
            if end_pos is not None:
                try:
                    end_pos = int(end_pos)
                except (ValueError, TypeError):
                    end_pos = None
    
    # If no range, try specific_residues
    if start_pos is None and end_pos is None:
        specific_residues = feature_obj.get("specific_residues", [])
        if specific_residues and isinstance(specific_residues, list) and len(specific_residues) > 0:
            first_res = specific_residues[0]
            if isinstance(first_res, dict):
                pos = first_res.get("position")
                if pos is not None:
                    try:
                        single_pos = int(pos)
                        start_pos = single_pos
                        end_pos = single_pos
                    except (ValueError, TypeError):
                        pass
    
    return start_pos, end_pos, single_pos


def _format_position(start: Optional[int], end: Optional[int], single: Optional[int]) -> str:
    """Format position as start-end or comma-separated list."""
    if start is not None and end is not None:
        if start == end:
            return f"{start}-{end}"
        else:
            return f"{start}-{end}"
    elif single is not None:
        return f"{single}-{single}"
    return ""


def raw_to_csv(batch: Dict[str, Dict], apply_filters: bool = True) -> pd.DataFrame:
    """
    Convert raw LLM output to CSV format matching ground truth.
    
    Rules:
    1. If sequence feature name exists but NO position (start/end), skip row
    2. If position (start/end) exists but NO sequence feature name, show row
    3. Group multiple features from same quote/virus/protein into one row
    """
    # First pass: collect all features with their data
    feature_data = []
    
    for pmid, entry in batch.items():
        if entry.get("status") != "ok":
            continue
        
        pmcid = entry.get("pmcid", "")
        title = entry.get("title", "")
        result = entry.get("result", {})
        features = result.get("sequence_features", [])
        
        for feat in features:
            if not isinstance(feat, dict):
                continue
            
            feature_obj = feat.get("feature", {})
            confidence_obj = feat.get("confidence", {})
            
            feature_name = (feature_obj.get("name_or_label") or "").strip()
            quote = (feat.get("evidence_snippet") or "").strip()
            virus = (feat.get("virus") or "").strip()
            protein = (feat.get("protein") or "").strip()
            
            # Extract positions
            start_pos, end_pos, single_pos = _extract_position_from_feature(feature_obj)
            
            # Apply filters only if enabled
            if apply_filters:
                # Rule 1: Skip if has feature name but NO position
                if feature_name and start_pos is None and end_pos is None and single_pos is None:
                    continue
                
                # Skip if has NO feature name AND NO position AND NO quote (empty row)
                if not feature_name and start_pos is None and end_pos is None and single_pos is None and not quote:
                    continue
            
            # Rule 2: Keep if has position but no feature name (or has both)
            
            # Determine target_type
            target_type = ""
            feature_type = feature_obj.get("type", "")
            variants = feature_obj.get("variants", [])
            
            if feature_type == "mutation_effect" or variants:
                target_type = "mutation"
            elif feature_type in ("domain", "region", "linker", "disordered_region"):
                target_type = "domain"
            elif feature_type in ("signal", "active_site", "binding_site"):
                target_type = "domain"
            else:
                if feature_name and re.search(r"[A-Z]\d+[A-Z*]", feature_name):
                    target_type = "mutation"
                elif start_pos is not None or single_pos is not None:
                    target_type = "amino_acid"
                else:
                    target_type = "domain"
            
            # Get confidence
            confidence = 0.0
            try:
                confidence = float(confidence_obj.get("score_0_to_1", 0.0))
            except (ValueError, TypeError):
                pass
            
            feature_data.append({
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "virus": virus,
                "protein": protein,
                "feature_name": feature_name,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "single_pos": single_pos,
                "quote": quote,
                "target_type": target_type,
                "confidence": confidence,
            })
    
    # Second pass: Group features by (pmid, virus, protein, quote)
    grouped = {}
    
    for data in feature_data:
        # Create grouping key: same quote, same virus, same protein
        key = (data["pmid"], data["virus"], data["protein"], data["quote"])
        
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(data)
    
    # Third pass: Build rows (group multiple features in same quote)
    rows = []
    
    for key, group in grouped.items():
        pmid, virus, protein, quote = key
        
        # Get common fields (same for all in group)
        pmcid = group[0]["pmcid"]
        title = group[0]["title"]
        target_type = group[0]["target_type"]  # Assume same type for grouped features
        confidence = group[0]["confidence"]  # Use first confidence
        
        # Collect feature names and positions
        feature_names = []
        start_positions = []
        end_positions = []
        single_positions = []
        
        for item in group:
            if item["feature_name"]:
                feature_names.append(item["feature_name"])
            
            if item["start_pos"] is not None:
                start_positions.append(item["start_pos"])
            if item["end_pos"] is not None:
                end_positions.append(item["end_pos"])
            if item["single_pos"] is not None:
                single_positions.append(item["single_pos"])
        
        # Format sequence feature name (comma-separated if multiple)
        sequence_str = ", ".join(feature_names) if feature_names else ""
        
        # Format positions
        # If we have ranges, use start-end format
        # If we have single positions, use comma-separated list
        position_str = ""
        start_pos = None
        end_pos = None
        
        if start_positions and end_positions:
            # Check if all are ranges or all are single
            if len(start_positions) == 1 and len(end_positions) == 1:
                start_pos = start_positions[0]
                end_pos = end_positions[0]
                if start_pos == end_pos:
                    position_str = f"{start_pos}-{end_pos}"
                else:
                    position_str = f"{start_pos}-{end_pos}"
            else:
                # Multiple ranges - format as comma-separated
                pos_parts = []
                for s, e in zip(start_positions, end_positions):
                    if s == e:
                        pos_parts.append(str(s))
                    else:
                        pos_parts.append(f"{s}-{e}")
                position_str = ", ".join(pos_parts)
        elif single_positions:
            # Single positions - comma-separated
            position_str = ", ".join(str(p) for p in sorted(set(single_positions)))
            if len(single_positions) == 1:
                start_pos = single_positions[0]
                end_pos = single_positions[0]
        
        # Build row
        row = {
            "pmid": pmid,
            "pmcid": pmcid or "",
            "title": title or "",
            "virus": virus or "",
            "protein": protein or "",
            "sequence_feature_name": sequence_str,
            "start_position": start_pos,
            "end_position": end_pos,
            "position": position_str,
            "quote": quote,
            "target_type": target_type,
            "confidence": confidence,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


__all__ = ["raw_to_csv"]



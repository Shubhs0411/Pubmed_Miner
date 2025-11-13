# llm/anthropic.py - Anthropic Claude API-specific implementation
# Uses shared utilities from llm.utils

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional

try:
    import anthropic
except Exception as e:
    raise RuntimeError("Missing dependency: pip install anthropic") from e

# Import ALL shared utilities
from llm import utils

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _anthropic_complete(prompt: str, api_key: str, model_name: str, max_output_tokens: int = 8192) -> str:
    """Anthropic Claude API completion call."""
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=max_output_tokens,
            temperature=0.0,
            # Use ONLY the prompt from pass2_prompt (which already contains all system instructions)
            # This ensures consistency with Gemini and all other models
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"[Claude] Error: {e}")
        raise


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Single-call, multi-chunk extraction using the analyst prompt.
    Uses shared utilities from llm.utils for all processing logic.
    """
    meta = meta or {}
    
    # If custom prompt provided from frontend, use it (temporarily override)
    custom_prompt = meta.get("analyst_prompt")
    if custom_prompt:
        from llm.prompts import PROMPTS
        original_prompt = PROMPTS.analyst_prompt
        PROMPTS.analyst_prompt = custom_prompt
    
    # Handle API key from meta (frontend) or env (backup)
    api_key_from_meta = meta.get("api_key")
    api_key_from_env = os.getenv("ANTHROPIC_API_KEY")
    api_key = api_key_from_meta or api_key_from_env
    
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"ANTHROPIC_API_KEY not set. Provide it in meta['api_key'] (from frontend) or set ANTHROPIC_API_KEY environment variable. "
            f"Received: meta['api_key']={repr(api_key_from_meta)}, env={repr(api_key_from_env)}"
        )
    
    api_key = api_key.strip()
    model_name = meta.get("model_name", DEFAULT_MODEL)
    delay_ms = int(meta.get("delay_ms") or 0)
    
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    text_norm = utils.normalize_ws(paper_text or "")

    scan_candidates = utils.scan_text_candidates(text_norm)

    # Chunk parameters
    chunk_chars = int(meta.get("chunk_chars") or 100_000)
    overlap_chars = int(meta.get("overlap_chars") or 2_000)
    max_chunks = int(meta.get("max_chunks") or 4)

    chunks = list(utils.chunk_text(text_norm, max_chars=chunk_chars, overlap=overlap_chars))
    chunks = chunks[:max_chunks] if chunks else [text_norm]

    all_features: List[Any] = []
    
    # Process each chunk
    for idx, ch in enumerate(chunks, 1):
        prompt2 = utils.pass2_prompt(
            ch, target_token="", pmid=pmid, pmcid=pmcid, 
            token_type="paper", scan_candidates=scan_candidates
        )
        
        try:
            raw2 = _anthropic_complete(prompt2, api_key, model_name, max_output_tokens=8192)
            j2 = utils.safe_json_value(raw2)

            if isinstance(j2, dict) and isinstance(j2.get("sequence_features"), list):
                feats = j2["sequence_features"]
            elif isinstance(j2, list):
                feats = j2
            else:
                feats = []

            print(f"[DEBUG] chunk {idx}/{len(chunks)} feature_count:", len(feats))
            
            normalized: List[Any] = []
            for feat in feats:
                if isinstance(feat, dict):
                    norm = utils.normalize_prompt_feature(feat)
                    if norm:
                        normalized.append(norm)
            
            all_features.extend(normalized)
        except Exception as e:
            print(f"[Claude] Error on chunk {idx}: {e}")
            continue
        
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    # DISABLED: Targeted follow-up for missed mutations (regex-based)
    # This section used regex-found tokens to do a second pass extraction.
    # Disabled to let LLM extract purely from the prompt without regex hints.
    
    # ========== COMMENTED OUT - REGEX-BASED FOLLOW-UP EXTRACTION ==========
    # extracted_tokens = utils.collect_extracted_tokens([f for f in all_features if isinstance(f, dict)])
    # mutation_candidates = scan_candidates.get("mutation_tokens", []) if isinstance(scan_candidates, dict) else []
    # hgvs_candidates = scan_candidates.get("hgvs_tokens", []) if isinstance(scan_candidates, dict) else []
    # 
    # followup_tokens: List[str] = []
    # for token in mutation_candidates + hgvs_candidates:
    #     if token and token not in extracted_tokens:
    #         followup_tokens.append(token)
    # 
    # # Deduplicate
    # deduped_follow = []
    # seen_follow: set = set()
    # for token in followup_tokens:
    #     if token not in seen_follow:
    #         seen_follow.add(token)
    #         deduped_follow.append(token)
    # followup_tokens = deduped_follow
    # 
    # # Process missed mutations
    # for token in followup_tokens[:15]:
    #     context = utils.token_context_windows(
    #         text_norm, token, left=900, right=900, max_windows=4
    #     )
    #     hints = {"mutation_tokens": [token]}
    #     prompt_focus = utils.pass2_prompt(
    #         context, target_token=token, pmid=pmid, pmcid=pmcid, 
    #         token_type="mutation", scan_candidates=hints
    #     )
    #     
    #     try:
    #         raw_focus = _anthropic_complete(prompt_focus, api_key, model_name, max_output_tokens=2048)
    #         parsed_focus = utils.safe_json_value(raw_focus)
    #         
    #         focus_feats: List[Dict[str, Any]] = []
    #         if isinstance(parsed_focus, dict) and isinstance(parsed_focus.get("sequence_features"), list):
    #             focus_feats = parsed_focus["sequence_features"]
    #         elif isinstance(parsed_focus, list):
    #             focus_feats = parsed_focus
    #         
    #         normalized_focus: List[Dict[str, Any]] = []
    #         for feat in focus_feats:
    #             if isinstance(feat, dict):
    #                 norm_feat = utils.normalize_prompt_feature(feat)
    #                 if norm_feat:
    #                     normalized_focus.append(norm_feat)
    #         
    #         if normalized_focus:
    #             all_features.extend(normalized_focus)
    #             extracted_tokens.update(utils.collect_extracted_tokens(normalized_focus))
    #     except Exception as e:
    #         print(f"[Claude] Error on followup token {token}: {e}")
    #         continue
    #     
    #     if delay_ms:
    #         time.sleep(delay_ms / 1000.0)

    # Deduplicate at JSON-schema level
    def _k(f):
        if not isinstance(f, dict):
            return ("", "", "", "", "")
        feat = f.get("feature") or {}
        positions = feat.get("residue_positions") or []
        pos0 = positions[0] if positions else {}
        return (
            (f.get("virus") or "").lower(),
            (f.get("protein") or "").lower(),
            (feat.get("name_or_label") or "").lower(),
            (feat.get("type") or "").lower(),
            f"{pos0.get('start')}-{pos0.get('end')}",
        )
    
    seen = set()
    uniq = []
    for f in all_features:
        k = _k(f)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)

    raw = {
        "paper": {
            "pmid": pmid, 
            "pmcid": pmcid, 
            "title": meta.get("title"),
            "virus_candidates": [], 
            "protein_candidates": []
        },
        "sequence_features": uniq,
        "scan_candidates": scan_candidates,
    }

    cleaned = utils.clean_and_ground(
        raw, text_norm,
        restrict_to_paper=True,
        require_mutation_in_quote=False,
        min_confidence=float(meta.get("min_confidence") or 0.0),
    )
    
    return cleaned


# Re-export clean_and_ground from utils for backward compatibility
clean_and_ground = utils.clean_and_ground


__all__ = [
    "run_on_paper",
    "clean_and_ground",
    "DEFAULT_MODEL",
]


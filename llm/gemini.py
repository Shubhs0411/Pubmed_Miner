# llm/gemini.py - Gemini API-specific implementation
# Uses shared utilities from llm.utils

from __future__ import annotations

import os
import time
import threading
from time import monotonic as _mono
from typing import List, Dict, Any, Optional

# Import ALL shared utilities
from llm import utils

# Config
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

GEMINI_RPM = int(os.getenv("GEMINI_RPM", "15"))
GEMINI_TPM = int(os.getenv("GEMINI_TPM", "250000"))

try:
    import google.generativeai as genai
except Exception as e:
    raise RuntimeError("Missing dependency: pip install google-generativeai") from e

# Initialize with API key if available (can be updated via reload)
# Always reconfigure to ensure we use the latest API key (important for reloads)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel(DEFAULT_MODEL)
else:
    # Model will be initialized lazily in _gemini_complete when API key is available
    _model = None

_RATE_LOCK = threading.Lock()
_MIN_INTERVAL = 60.0 / GEMINI_RPM if GEMINI_RPM > 0 else 0.0
_WINDOW_START = _mono()
_TOKENS_USED = 0


# ========== Gemini-specific Rate limiting ==========
def _approx_tokens(s: str) -> int:
    """Approximate token count for Gemini."""
    return int(len(s) / 4) if s else 0


def _rpm_gate():
    """Rate limit by requests per minute."""
    if _MIN_INTERVAL <= 0:
        return
    with _RATE_LOCK:
        last = getattr(_rpm_gate, "_last", 0.0)
        now = _mono()
        wait = (last + _MIN_INTERVAL) - now
        if wait > 0:
            time.sleep(wait)
            now = _mono()
        _rpm_gate._last = now


def _tpm_gate(estimated_tokens: int):
    """Rate limit by tokens per minute."""
    global _WINDOW_START, _TOKENS_USED
    if GEMINI_TPM <= 0:
        return
    now = _mono()
    elapsed = now - _WINDOW_START
    if elapsed >= 60.0:
        _WINDOW_START = now
        _TOKENS_USED = 0
    if _TOKENS_USED + estimated_tokens > GEMINI_TPM:
        sleep_for = max(0.0, 60.0 - elapsed)
        if sleep_for > 0:
            print(f"[gemini] TPM cap reached; sleeping {sleep_for:.1f}s…", flush=True)
            time.sleep(sleep_for)
        _WINDOW_START = _mono()
        _TOKENS_USED = 0
    _TOKENS_USED += max(0, estimated_tokens)


# ========== Gemini API wrapper ==========
if "_gemini_complete" not in globals():
    def _gemini_complete(prompt: str, max_output_tokens: int = 8192) -> str:
        """Gemini API completion call with rate limiting."""
        # Ensure model is initialized (in case API key was set after import)
        global _model
        if _model is None:
            current_key = os.getenv("GEMINI_API_KEY")
            if not current_key:
                raise RuntimeError("GEMINI_API_KEY not set. Put it in your environment or .env, or provide via frontend.")
            genai.configure(api_key=current_key)
            _model = genai.GenerativeModel(DEFAULT_MODEL)
        
        try:
            _rpm_gate()
        except Exception:
            pass
        try:
            _tpm_gate(_approx_tokens(prompt))
        except Exception:
            pass

        resp = _model.generate_content(
            [prompt],
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": 0.0,
            },
        )
        text = getattr(resp, "text", None)
        if not text:
            cand = getattr(resp, "candidates", None)
            if cand:
                try:
                    text = cand[0].content.parts[0].text
                except Exception:
                    text = ""
        return text or ""


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
    
    # Handle API key from meta (frontend) or env (backup) - ensure it's set before use
    api_key_from_meta = meta.get("api_key")
    api_key_from_env = os.getenv("GEMINI_API_KEY")
    api_key = api_key_from_meta or api_key_from_env
    
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"GEMINI_API_KEY not set. Provide it in meta['api_key'] (from frontend) or set GEMINI_API_KEY environment variable. "
            f"Received: meta['api_key']={repr(api_key_from_meta)}, env={repr(api_key_from_env)}"
        )
    
    # Strip whitespace
    api_key = api_key.strip()
    
    # Update environment and reconfigure with the API key (frontend takes priority)
    os.environ["GEMINI_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    # Reinitialize model with current API key and model name
    global _model
    model_name = meta.get("model_name") or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    if model_name != DEFAULT_MODEL:
        os.environ["GEMINI_MODEL"] = model_name
    # Recreate model with current API key and model name
    _model = genai.GenerativeModel(model_name)
    
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
        raw2 = _gemini_complete(prompt2, max_output_tokens=8192)
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
    #     raw_focus = _gemini_complete(prompt_focus, max_output_tokens=2048)
    #     parsed_focus = utils.safe_json_value(raw_focus)
    #     
    #     focus_feats: List[Dict[str, Any]] = []
    #     if isinstance(parsed_focus, dict) and isinstance(parsed_focus.get("sequence_features"), list):
    #         focus_feats = parsed_focus["sequence_features"]
    #     elif isinstance(parsed_focus, list):
    #         focus_feats = parsed_focus
    #     
    #     normalized_focus: List[Dict[str, Any]] = []
    #     for feat in focus_feats:
    #         if isinstance(feat, dict):
    #             norm_feat = utils.normalize_prompt_feature(feat)
    #             if norm_feat:
    #                 normalized_focus.append(norm_feat)
    #     
    #     if normalized_focus:
    #         all_features.extend(normalized_focus)
    #         extracted_tokens.update(utils.collect_extracted_tokens(normalized_focus))

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

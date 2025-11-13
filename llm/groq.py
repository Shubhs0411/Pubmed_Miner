# llm/groq.py - Groq/Llama API-specific implementation
# Uses shared utilities from llm.utils

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Import ALL shared utilities
from llm import utils

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _require_key(api_key: Optional[str] = None) -> str:
    """Get API key from parameter or environment."""
    if api_key:
        return api_key
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Export it or put it in a .env file.")
    return key


def _post_chat(payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Post to Groq API with retries and rate limiting."""
    # Validate API key before making request
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"GROQ_API_KEY is empty or invalid. "
            f"Received key (first 10 chars): {repr(api_key[:10]) if api_key else 'None'}"
        )
    
    api_key = api_key.strip()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    max_retries = 3  # Reduced from 6 to 3 for faster failure
    backoff = 0.5  # Reduced initial backoff from 1.0 to 0.5
    timeout = 30  # Reduced from 90 to 30 seconds for faster timeout
    for attempt in range(1, max_retries + 1):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 401:
            # 401 Unauthorized - don't retry, provide helpful error
            error_msg = f"Groq API 401 Unauthorized. "
            try:
                error_body = resp.json()
                if "error" in error_body:
                    error_msg += f"Error: {error_body.get('error', {}).get('message', 'Invalid API key')}. "
            except Exception:
                error_msg += "Invalid or missing API key. "
            error_msg += (
                f"Please check your API key at https://console.groq.com/keys. "
                f"API key (first 10 chars): {repr(api_key[:10])}"
            )
            raise RuntimeError(error_msg)
        if resp.status_code == 404 and "unknown_url" in (resp.text or ""):
            raise RuntimeError(
                "Groq API 404 unknown_url. Use POST https://api.groq.com/openai/v1/chat/completions"
            )
        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = backoff
            else:
                try:
                    import os as _os
                    sleep_s = backoff + (0.5 * _os.urandom(1)[0] / 255.0)
                except Exception:
                    sleep_s = backoff
            if attempt == max_retries:
                resp.raise_for_status()
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, 16.0)
            continue
        if resp.status_code == 413:
            raise requests.HTTPError("413 Payload Too Large", response=resp)
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded retry attempts contacting Groq API.")


def _payload_size(messages: List[Dict[str, str]]) -> int:
    """Calculate total payload size."""
    return sum(len(m.get("content", "")) for m in messages)


def _shrink_user_content(messages: List[Dict[str, str]], keep_bytes: int) -> List[Dict[str, str]]:
    """Shrink user content to fit within byte limit."""
    if not messages:
        return messages
    total = _payload_size(messages)
    if total <= keep_bytes:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            usr = messages[i].get("content", "")
            overshoot = total - keep_bytes
            if overshoot <= 0:
                return messages
            new_len = max(0, len(usr) - overshoot - 512)
            messages = messages.copy()
            messages[i] = {"role": "user", "content": usr[:new_len]}
            return messages
    return messages


def _groq_complete(prompt: str, api_key: str, model_name: str, max_output_tokens: int = 8192) -> str:
    """Groq API completion call with payload size handling."""
    # Use ONLY the prompt from pass2_prompt (which already contains all system instructions)
    # This ensures consistency with Gemini and all other models
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }
    
    try:
        data = _post_chat(payload, api_key)
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 413:
            # Try shrinking payload
            shrink_targets = [20000, 14000, 10000, 7000, 5000]
            for budget in shrink_targets:
                shrunk = _shrink_user_content(messages, keep_bytes=budget)
                payload = {
                    "model": model_name,
                    "messages": shrunk,
                    "temperature": 0.0,
                    "max_tokens": max_output_tokens,
                }
                try:
                    data = _post_chat(payload, api_key)
                    return data["choices"][0]["message"]["content"]
                except requests.HTTPError as e2:
                    if getattr(e2, "response", None) is not None and e2.response.status_code == 413:
                        continue
                    raise
            raise
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
    api_key_from_env = os.getenv("GROQ_API_KEY")
    api_key = api_key_from_meta or api_key_from_env
    
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"GROQ_API_KEY not set. Provide it in meta['api_key'] (from frontend) or set GROQ_API_KEY environment variable. "
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
            raw2 = _groq_complete(prompt2, api_key, model_name, max_output_tokens=8192)
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
        except requests.HTTPError as e:
            # Handle HTTP errors (including 401)
            if e.response and e.response.status_code == 401:
                error_msg = (
                    f"Groq API 401 Unauthorized. Invalid or missing API key. "
                    f"Please check your API key at https://console.groq.com/keys. "
                    f"API key provided: {'Yes' if api_key else 'No'} "
                    f"(first 10 chars: {repr(api_key[:10]) if api_key else 'N/A'})"
                )
                raise RuntimeError(error_msg) from e
            print(f"[Groq] HTTP Error on chunk {idx}: {e}")
            continue
        except RuntimeError as e:
            # Don't continue on auth errors - fail fast
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "API key" in error_msg:
                raise RuntimeError(f"[Groq] Authentication failed on chunk {idx}: {error_msg}") from e
            print(f"[Groq] Error on chunk {idx}: {e}")
            continue
        except Exception as e:
            print(f"[Groq] Error on chunk {idx}: {e}")
            import traceback
            print(f"[Groq] Traceback: {traceback.format_exc()}")
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
    #         raw_focus = _groq_complete(prompt_focus, api_key, model_name, max_output_tokens=2048)
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
    #         print(f"[Groq] Error on followup token {token}: {e}")
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


def chat_complete(messages: List[Dict[str, str]],
                  api_key: Optional[str] = None,
                  model: str = DEFAULT_MODEL,
                  temperature: float = 0.2,
                  max_tokens: int = 1024) -> str:
    """Legacy chat_complete function for backward compatibility."""
    api_key = _require_key(api_key)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _post_chat(payload, api_key)
    return data["choices"][0]["message"]["content"]


__all__ = [
    "chat_complete",
    "run_on_paper",
    "clean_and_ground",
    "DEFAULT_MODEL",
    "GROQ_API_URL",
]

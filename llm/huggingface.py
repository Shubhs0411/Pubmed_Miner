# llm/huggingface.py - Hugging Face Inference API implementation
# Uses shared utilities from llm.utils

from __future__ import annotations

import os
import time
import requests
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Import ALL shared utilities
from llm import utils

# Hugging Face Inference API endpoint
# Official endpoint: https://api-inference.huggingface.co/models/{model_id}
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

# Default model - Models confirmed to work on free Inference API:
# Note: Many newer models require gated access or aren't available on free tier
# Using basic, widely available models that are guaranteed to work:
# 1. gpt2 - Basic but always available, free tier friendly
# 2. distilgpt2 - Faster version of GPT-2
# 3. google/flan-t5-base - Smaller T5 model, more likely to be available
# 4. facebook/bart-large-cnn - Good for summarization
DEFAULT_MODEL = "gpt2"  # Most basic, always available on free tier


def _hf_complete(prompt: str, api_key: str, model_name: str, max_output_tokens: int = 8192) -> str:
    """Hugging Face Inference API completion call.
    
    Official API documentation: https://huggingface.co/docs/api-inference/index
    Endpoint: https://api-inference.huggingface.co/models/{model_id}
    """
    # Validate API key
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"HF_API_KEY is empty or invalid. "
            f"Received key (first 10 chars): {repr(api_key[:10]) if api_key else 'None'}"
        )
    
    api_key = api_key.strip()
    
    # Official Hugging Face Inference API headers
    # Only Authorization header is needed - requests library sets Content-Type automatically
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    # Use ONLY the prompt from pass2_prompt (which already contains all system instructions)
    # This ensures consistency with Gemini and all other models
    # The analyst_prompt already starts with "Respond only in JSON" and has system instructions
    formatted_prompt = prompt
    
    # Official Inference API endpoint format
    url = HF_API_URL.format(model=model_name)
    
    # Official payload format per Hugging Face docs
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "temperature": 0.0,
            "max_new_tokens": min(max_output_tokens, 1024),  # Many free models have lower limits
            "return_full_text": False,
        }
    }
    
    max_retries = 3
    backoff = 0.5
    timeout = 90  # HF can be slower, especially for first request (model loading)
    
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            # Handle model loading (HF returns 503 while loading)
            if resp.status_code == 503:
                error_data = resp.json() if resp.content else {}
                wait_time = error_data.get("estimated_time", 15)
                if attempt < max_retries:
                    print(f"[HF] Model loading, waiting {wait_time}s (attempt {attempt}/{max_retries})...")
                    time.sleep(min(wait_time, 45))  # Cap at 45 seconds
                    continue
                else:
                    raise RuntimeError(
                        f"HF model {model_name} is still loading after {max_retries} attempts. "
                        f"Try again in a few minutes or use a different model."
                    )
            
            # Handle rate limiting
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "10")
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = backoff
                if attempt < max_retries:
                    print(f"[HF] Rate limited, waiting {sleep_s}s (attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2.0, 16.0)
                    continue
            
            # Handle model no longer available (410 Gone) or not accessible
            if resp.status_code == 410:
                error_msg = (
                    f"Hugging Face model '{model_name}' is not available (410 Gone).\n\n"
                    f"⚠️ Many models are NOT available on the free Inference API.\n\n"
                    f"Models that usually work on free tier:\n"
                    f"• gpt2 (always available, basic)\n"
                    f"• distilgpt2 (faster GPT-2)\n"
                    f"• google/flan-t5-base (if available)\n"
                    f"• facebook/bart-large-cnn (for summarization)\n\n"
                    f"Note: Most instruction-tuned models require:\n"
                    f"1. Gated access (accept terms on model page)\n"
                    f"2. Paid Inference API subscription\n"
                    f"3. Or are simply not available on free tier\n\n"
                    f"💡 Recommendation: For reliable structured JSON output, use:\n"
                    f"• Groq (free tier, fast, reliable)\n"
                    f"• OpenAI GPT-4o (paid but excellent)\n"
                    f"• Anthropic Claude (paid but excellent)\n\n"
                    f"Hugging Face free tier is best for simple text generation tasks, "
                    f"not complex structured extraction."
                )
                raise RuntimeError(error_msg)
            
            # Handle authentication errors
            if resp.status_code == 401:
                error_msg = "Hugging Face API 401 Unauthorized. Invalid or missing API key. "
                try:
                    error_body = resp.json()
                    if "error" in error_body:
                        error_msg += f"Error: {error_body.get('error', 'Invalid API key')}. "
                except Exception:
                    pass
                error_msg += (
                    f"Please check your API key at https://huggingface.co/settings/tokens. "
                    f"API key (first 10 chars): {repr(api_key[:10])}"
                )
                raise RuntimeError(error_msg)
            
            resp.raise_for_status()
            data = resp.json()
            
            # Check for errors in response body (sometimes API returns errors with 200 status)
            if isinstance(data, dict) and "error" in data:
                error_msg = data.get("error", "Unknown error")
                # Handle authentication/login errors
                if "username" in str(error_msg).lower() or "password" in str(error_msg).lower():
                    raise RuntimeError(
                        f"Hugging Face API error: {error_msg}\n\n"
                        f"This suggests an authentication issue. Make sure you're using a valid API token (not username/password). "
                        f"Get your token from: https://huggingface.co/settings/tokens\n"
                        f"Token should start with 'hf_' and be a long string."
                    )
                raise RuntimeError(f"Hugging Face API error: {error_msg}")
            
            # Extract response from HF inference API format
            # Response is usually a list with one dict containing "generated_text"
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                # Some models return string directly in list
                if isinstance(data[0], str):
                    return data[0]
            
            # Fallback: check if it's a dict directly
            if isinstance(data, dict):
                if "generated_text" in data:
                    return data["generated_text"]
                if "text" in data:
                    if isinstance(data["text"], list) and len(data["text"]) > 0:
                        return data["text"][0]
                    return str(data["text"])
            
            raise RuntimeError(f"Unexpected response format from HF API: {data}")
            
        except requests.Timeout:
            if attempt < max_retries:
                print(f"[HF] Request timeout, retrying (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 16.0)
                continue
            raise RuntimeError(f"HF API request timed out after {max_retries} attempts")
        
        except requests.RequestException as e:
            if attempt < max_retries:
                print(f"[HF] Request error: {e}, retrying (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 16.0)
                continue
            raise RuntimeError(f"HF API request failed: {e}")
    
    raise RuntimeError(f"HF API request failed after {max_retries} attempts")


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Single-call, multi-chunk extraction using the analyst prompt.
    Uses shared utilities from llm.utils for all processing logic.
    """
    meta = meta or {}
    
    # Handle API key from meta (frontend) or env (backup)
    api_key_from_meta = meta.get("api_key")
    api_key_from_env = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    api_key = api_key_from_meta or api_key_from_env
    
    if not api_key or not api_key.strip():
        raise RuntimeError(
            f"HF_API_KEY not set. Provide it in meta['api_key'] (from frontend) or set HF_API_KEY or HUGGINGFACE_API_KEY environment variable. "
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
            raw2 = _hf_complete(prompt2, api_key, model_name, max_output_tokens=8192)
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
            # Handle HTTP errors
            if e.response:
                status_code = e.response.status_code
                if status_code == 401:
                    error_msg = (
                        f"Hugging Face API 401 Unauthorized. Invalid or missing API key. "
                        f"Please check your API key at https://huggingface.co/settings/tokens. "
                        f"API key provided: {'Yes' if api_key else 'No'} "
                        f"(first 10 chars: {repr(api_key[:10]) if api_key else 'N/A'})"
                    )
                    raise RuntimeError(error_msg) from e
                elif status_code == 410:
                    # Model no longer available - don't retry
                    error_msg = (
                        f"Hugging Face model '{model_name}' is not available (410 Gone).\n\n"
                        f"This model may require gated access or isn't on the free Inference API.\n"
                        f"Visit https://huggingface.co/{model_name} to check access requirements.\n"
                        f"Try a publicly available model like google/flan-t5-large"
                    )
                    raise RuntimeError(error_msg) from e
            print(f"[HF] HTTP Error on chunk {idx}: {e}")
            continue
        except RuntimeError as e:
            # Don't continue on auth or model unavailable errors - fail fast
            error_msg = str(e)
            if any(keyword in error_msg for keyword in ["401", "410", "Unauthorized", "API key", "no longer available", "Gone"]):
                raise RuntimeError(f"[HF] Error on chunk {idx}: {error_msg}") from e
            print(f"[HF] Error on chunk {idx}: {e}")
            continue
        except Exception as e:
            print(f"[HF] Error on chunk {idx}: {e}")
            import traceback
            print(f"[HF] Traceback: {traceback.format_exc()}")
            continue
        
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    # Targeted follow-up for missed mutations
    extracted_tokens = utils.collect_extracted_tokens([f for f in all_features if isinstance(f, dict)])
    mutation_candidates = scan_candidates.get("mutation_tokens", []) if isinstance(scan_candidates, dict) else []
    hgvs_candidates = scan_candidates.get("hgvs_tokens", []) if isinstance(scan_candidates, dict) else []
    
    followup_tokens: List[str] = []
    for token in mutation_candidates + hgvs_candidates:
        if token and token not in extracted_tokens:
            followup_tokens.append(token)
    
    # Deduplicate
    deduped_follow = []
    seen_follow: set = set()
    for token in followup_tokens:
        if token not in seen_follow:
            seen_follow.add(token)
            deduped_follow.append(token)
    followup_tokens = deduped_follow

    # Process missed mutations
    for token in followup_tokens[:15]:
        context = utils.token_context_windows(
            text_norm, token, left=900, right=900, max_windows=4
        )
        hints = {"mutation_tokens": [token]}
        prompt_focus = utils.pass2_prompt(
            context, target_token=token, pmid=pmid, pmcid=pmcid, 
            token_type="mutation", scan_candidates=hints
        )
        
        try:
            raw_focus = _hf_complete(prompt_focus, api_key, model_name, max_output_tokens=2048)
            parsed_focus = utils.safe_json_value(raw_focus)
            
            focus_feats: List[Dict[str, Any]] = []
            if isinstance(parsed_focus, dict) and isinstance(parsed_focus.get("sequence_features"), list):
                focus_feats = parsed_focus["sequence_features"]
            elif isinstance(parsed_focus, list):
                focus_feats = parsed_focus
            
            normalized_focus: List[Dict[str, Any]] = []
            for feat in focus_feats:
                if isinstance(feat, dict):
                    norm_feat = utils.normalize_prompt_feature(feat)
                    if norm_feat:
                        normalized_focus.append(norm_feat)
            
            if normalized_focus:
                all_features.extend(normalized_focus)
                extracted_tokens.update(utils.collect_extracted_tokens(normalized_focus))
        except Exception as e:
            print(f"[HF] Error on followup token {token}: {e}")
            continue
        
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

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
    }

    return raw


__all__ = ["run_on_paper"]


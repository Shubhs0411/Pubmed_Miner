# llm/custom.py - Custom/Hackathon LLM backend scaffold
# Uses shared utilities from llm.utils and mirrors the behavior of other backends.

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, Optional, List

import requests

from llm import utils

DEFAULT_MODEL = os.getenv("CUSTOM_LLM_MODEL", "custom-hackathon-model")
DEFAULT_URL = os.getenv("CUSTOM_LLM_URL", "")
DEFAULT_TIMEOUT = int(os.getenv("CUSTOM_LLM_TIMEOUT", "120"))
DEFAULT_MAX_TOKENS = int(os.getenv("CUSTOM_LLM_MAX_TOKENS", "8192"))
DEFAULT_TEMPERATURE = float(os.getenv("CUSTOM_LLM_TEMPERATURE", "0.0"))
DEFAULT_OPENAI_COMPATIBLE = os.getenv(
    "CUSTOM_LLM_OPENAI_COMPATIBLE", ""
).strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_OPENAI_CHAT_MODE = os.getenv(
    "CUSTOM_LLM_CHAT_MODE", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _custom_complete(
    prompt: str,
    *,
    api_url: str,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    openai_compatible: bool = False,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    chat_mode: bool = False,
) -> str:
    """Generic HTTP handler for custom LLM endpoints.

    Expects a JSON response that includes either:
      - "completion"
      - "text"
      - "output"
      - OpenAI-style {"choices": [{"text"|"message": {"content"}}]}
    """
    if not api_url:
        raise RuntimeError(
            "CUSTOM_LLM_URL (or meta['api_url']) is required for the custom backend."
        )

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    if extra_headers:
        headers.update(extra_headers)

    request_url = api_url.rstrip("/")

    def _maybe_apply_params(payload: Dict[str, Any]) -> None:
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

    if openai_compatible:
        base_url = request_url
        last_error: Optional[Exception] = None

        # Ordering: try /completions (prompt) first, then /chat/completions if needed or requested
        attempt_chat = chat_mode
        endpoints: List[Dict[str, Any]] = []

        if not attempt_chat:
            # prefer completions first
            if base_url.endswith("/v1"):
                endpoints.append({"url": f"{base_url}/completions", "mode": "completion"})
            elif base_url.endswith("/completions"):
                endpoints.append({"url": base_url, "mode": "completion"})
            else:
                endpoints.append({"url": base_url, "mode": "completion"})
                endpoints.append({"url": f"{base_url}/completions", "mode": "completion"})
            # allow chat as fallback
            endpoints.append({"url": f"{base_url}/chat/completions", "mode": "chat"})
        else:
            # Chat first if chat_mode requested
            if base_url.endswith("/v1"):
                endpoints.append({"url": f"{base_url}/chat/completions", "mode": "chat"})
            elif base_url.endswith("/chat/completions"):
                endpoints.append({"url": base_url, "mode": "chat"})
            else:
                endpoints.append({"url": base_url, "mode": "chat"})
                endpoints.append({"url": f"{base_url}/chat/completions", "mode": "chat"})
            # completions fallback
            endpoints.append({"url": f"{base_url}/completions", "mode": "completion"})

        for candidate in endpoints:
            mode = candidate["mode"]
            url = candidate["url"].rstrip("/")
            try:
                if mode == "completion":
                    payload: Dict[str, Any] = {
                        "model": model_name or DEFAULT_MODEL,
                        "prompt": prompt,
                    }
                    _maybe_apply_params(payload)
                else:
                    payload = {
                        "model": model_name or DEFAULT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    _maybe_apply_params(payload)

                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                request_url = url  # track final used URL
                break
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in {404, 405}:
                    continue  # try next endpoint
                raise
            except Exception as exc:
                last_error = exc
                raise
        else:
            if last_error:
                raise last_error
            raise RuntimeError("Custom LLM: no valid OpenAI-compatible endpoint could be reached.")
    else:
        payload = {"prompt": prompt}
        if model_name:
            payload["model"] = model_name
        _maybe_apply_params(payload)
        resp = requests.post(request_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()

    data = resp.json()
    
    # Debug: log response structure
    print(f"[CUSTOM LLM] API response keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

    # Flexible extraction of text from various response shapes
    text = (
        data.get("completion")
        or data.get("text")
        or data.get("output")
    )

    # Handle OpenAI-style responses
    if not text and isinstance(data.get("choices"), list):
        first_choice = data["choices"][0]
        if isinstance(first_choice, dict):
            if "text" in first_choice:
                text = first_choice["text"]
            elif isinstance(first_choice.get("message"), dict):
                text = first_choice["message"].get("content")
    
    # Debug: log what we extracted
    if text:
        print(f"[CUSTOM LLM] Extracted text length: {len(text)} chars, first 200: {text[:200]}")
    else:
        print(f"[CUSTOM LLM] WARNING: No text extracted. Full response structure: {str(data)[:500]}")

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"Custom LLM response is missing a usable completion field. Response keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}"
        )

    return text.strip()


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the custom LLM on the provided paper text using shared utilities."""
    meta = meta or {}
    
    # If custom prompt provided from frontend, use it (temporarily override)
    custom_prompt = meta.get("analyst_prompt")
    original_prompt = None
    if custom_prompt:
        from llm.prompts import PROMPTS
        original_prompt = PROMPTS.analyst_prompt
        PROMPTS.analyst_prompt = custom_prompt

    api_url = meta.get("api_url") or DEFAULT_URL
    api_key = meta.get("api_key") or os.getenv("CUSTOM_LLM_API_KEY", "")
    model_name = meta.get("model_name") or DEFAULT_MODEL
    timeout = int(meta.get("timeout") or DEFAULT_TIMEOUT)
    raw_openai_flag = meta.get("openai_compatible")
    if raw_openai_flag is None:
        openai_compatible = DEFAULT_OPENAI_COMPATIBLE
    elif isinstance(raw_openai_flag, str):
        openai_compatible = raw_openai_flag.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        openai_compatible = bool(raw_openai_flag)

    # For custom LLM, don't limit tokens - let the API handle it
    raw_max_tokens = meta.get("max_tokens")
    if raw_max_tokens is None:
        max_tokens = None  # No limit - send full paper
    else:
        try:
            max_tokens = int(raw_max_tokens)
            if max_tokens <= 0:
                max_tokens = None
        except (TypeError, ValueError):
            max_tokens = None  # Default to no limit

    temperature = meta.get("temperature", DEFAULT_TEMPERATURE)
    try:
        temperature = float(temperature) if temperature is not None else DEFAULT_TEMPERATURE
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE

    extra_headers: Dict[str, str] = {}
    raw_headers = meta.get("extra_headers") or os.getenv("CUSTOM_LLM_HEADERS", "")
    if isinstance(raw_headers, dict):
        extra_headers = raw_headers  # Already parsed
    elif isinstance(raw_headers, str) and raw_headers.strip():
        try:
            extra_headers = json.loads(raw_headers)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "CUSTOM_LLM_HEADERS is not valid JSON."
            ) from exc

    chat_mode_flag = meta.get("chat_mode")
    if chat_mode_flag is None:
        chat_mode = DEFAULT_OPENAI_CHAT_MODE
    elif isinstance(chat_mode_flag, str):
        chat_mode = chat_mode_flag.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        chat_mode = bool(chat_mode_flag)

    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")

    text_norm = utils.normalize_ws(paper_text or "")
    scan_candidates = utils.scan_text_candidates(text_norm)

    # For custom LLM: send the FULL paper text in one request (no chunking)
    # This ensures the LLM sees the complete context
    all_features: List[Any] = []

    try:
        print(f"[CUSTOM LLM] Processing full paper text ({len(text_norm)} chars) in single request")
        
        prompt2 = utils.pass2_prompt(
            text_norm,  # Full paper text, no chunking
            target_token="",
            pmid=pmid,
            pmcid=pmcid,
            token_type="paper",
            scan_candidates=scan_candidates,
        )

        print(f"[CUSTOM LLM] Prompt length: {len(prompt2)} chars")

        raw = _custom_complete(
            prompt2,
            api_url=api_url,
            api_key=api_key,
            extra_headers=extra_headers,
            model_name=model_name,
            timeout=timeout,
            openai_compatible=openai_compatible,
            max_tokens=max_tokens,  # None = no limit
            temperature=temperature,
            chat_mode=chat_mode,
        )

        parsed = utils.safe_json_value(raw)

        # Debug: log response structure
        if parsed is None:
            print(
                "[CUSTOM LLM] WARNING: Failed to parse JSON response. "
                f"Raw response (first 500 chars): {raw[:500]}"
            )
        elif not isinstance(parsed, (dict, list)):
            print(f"[CUSTOM LLM] WARNING: Parsed response is not dict/list: {type(parsed)}")

        if isinstance(parsed, dict) and isinstance(parsed.get("sequence_features"), list):
            feats = parsed["sequence_features"]
        elif isinstance(parsed, list):
            feats = parsed
        else:
            feats = []
            if parsed is not None:
                print(
                    "[CUSTOM LLM] WARNING: Unexpected response structure. "
                    f"Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}"
                )

        print(f"[CUSTOM LLM] Extracted {len(feats)} raw features from full paper")

        normalized = [
            utils.normalize_prompt_feature(f)
            for f in feats
            if isinstance(f, dict)
        ]

        print(f"[CUSTOM LLM] Normalized to {len(normalized)} features")

        all_features.extend([f for f in normalized if f])
    finally:
        if original_prompt is not None:
            from llm.prompts import PROMPTS
            PROMPTS.analyst_prompt = original_prompt

    # DISABLED: Targeted follow-up for missed mutations (regex-based)
    # This section used regex-found tokens to do a second pass extraction.
    # Disabled to let LLM extract purely from the prompt without regex hints.
    
    # ========== COMMENTED OUT - REGEX-BASED FOLLOW-UP EXTRACTION ==========
    # extracted_tokens = utils.collect_extracted_tokens(
    #     [f for f in all_features if isinstance(f, dict)]
    # )
    # mutation_candidates = scan_candidates.get("mutation_tokens", []) if isinstance(scan_candidates, dict) else []
    # hgvs_candidates = scan_candidates.get("hgvs_tokens", []) if isinstance(scan_candidates, dict) else []
    # 
    # followup_tokens: List[str] = []
    # for token in mutation_candidates + hgvs_candidates:
    #     if token and token not in extracted_tokens:
    #         followup_tokens.append(token)
    # 
    # seen_follow: set = set()
    # deduped_follow: List[str] = []
    # for token in followup_tokens:
    #     if token not in seen_follow:
    #         seen_follow.add(token)
    #         deduped_follow.append(token)
    # followup_tokens = deduped_follow
    # 
    # for token in followup_tokens[:15]:
    #     context = utils.token_context_windows(
    #         text_norm, token, left=900, right=900, max_windows=4
    #     )
    #     hints = {"mutation_tokens": [token]}
    #     prompt_focus = utils.pass2_prompt(
    #         context,
    #         target_token=token,
    #         pmid=pmid,
    #         pmcid=pmcid,
    #         token_type="mutation",
    #         scan_candidates=hints,
    #     )
    # 
    #     raw_focus = _custom_complete(
    #         prompt_focus,
    #         api_url=api_url,
    #         api_key=api_key,
    #         extra_headers=extra_headers,
    #         model_name=model_name,
    #         timeout=timeout,
    #     )
    #     parsed_focus = utils.safe_json_value(raw_focus)
    # 
    #     if isinstance(parsed_focus, dict) and isinstance(parsed_focus.get("sequence_features"), list):
    #         focus_feats = parsed_focus["sequence_features"]
    #     elif isinstance(parsed_focus, list):
    #         focus_feats = parsed_focus
    #     else:
    #         focus_feats = []
    # 
    #     normalized_focus = [
    #         utils.normalize_prompt_feature(f)
    #         for f in focus_feats
    #         if isinstance(f, dict)
    #     ]
    # 
    #     if normalized_focus:
    #         all_features.extend(normalized_focus)
    #         extracted_tokens.update(utils.collect_extracted_tokens(normalized_focus))
    # 
    #     if delay_ms:
    #         time.sleep(delay_ms / 1000.0)

    # Deduplicate at JSON-schema level (same logic as Gemini)
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

    raw_output = {
        "paper": {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": meta.get("title"),
            "virus_candidates": [],
            "protein_candidates": [],
        },
        "sequence_features": uniq,
        "scan_candidates": scan_candidates,
    }

    cleaned = utils.clean_and_ground(
        raw_output,
        text_norm,
        restrict_to_paper=True,
        require_mutation_in_quote=False,
        min_confidence=float(meta.get("min_confidence") or 0.0),
    )

    return cleaned


# Re-export clean_and_ground for parity with other backends
clean_and_ground = utils.clean_and_ground


__all__ = ["run_on_paper", "clean_and_ground", "DEFAULT_MODEL", "DEFAULT_URL"]

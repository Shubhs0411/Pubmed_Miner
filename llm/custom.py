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


def _custom_complete(
    prompt: str,
    *,
    api_url: str,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
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

    payload: Dict[str, Any] = {"prompt": prompt}
    if model_name:
        payload["model"] = model_name

    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()

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

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            "Custom LLM response is missing a usable completion field."
        )

    return text.strip()


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the custom LLM on the provided paper text using shared utilities."""
    meta = meta or {}

    api_url = meta.get("api_url") or DEFAULT_URL
    api_key = meta.get("api_key") or os.getenv("CUSTOM_LLM_API_KEY", "")
    model_name = meta.get("model_name") or DEFAULT_MODEL
    timeout = int(meta.get("timeout") or DEFAULT_TIMEOUT)

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

    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")

    text_norm = utils.normalize_ws(paper_text or "")
    scan_candidates = utils.scan_text_candidates(text_norm)

    chunk_chars = int(meta.get("chunk_chars") or 100_000)
    overlap_chars = int(meta.get("overlap_chars") or 2_000)
    max_chunks = int(meta.get("max_chunks") or 4)
    delay_ms = int(meta.get("delay_ms") or 0)

    chunks = list(utils.chunk_text(text_norm, max_chars=chunk_chars, overlap=overlap_chars))
    chunks = chunks[:max_chunks] if chunks else [text_norm]

    all_features: List[Any] = []

    for idx, chunk in enumerate(chunks, 1):
        prompt2 = utils.pass2_prompt(
            chunk,
            target_token="",
            pmid=pmid,
            pmcid=pmcid,
            token_type="paper",
            scan_candidates=scan_candidates,
        )

        raw = _custom_complete(
            prompt2,
            api_url=api_url,
            api_key=api_key,
            extra_headers=extra_headers,
            model_name=model_name,
            timeout=timeout,
        )

        parsed = utils.safe_json_value(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("sequence_features"), list):
            feats = parsed["sequence_features"]
        elif isinstance(parsed, list):
            feats = parsed
        else:
            feats = []

        normalized = [
            utils.normalize_prompt_feature(f)
            for f in feats
            if isinstance(f, dict)
        ]
        all_features.extend([f for f in normalized if f])

        if delay_ms:
            time.sleep(delay_ms / 1000.0)

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

    seen = set()
    uniq = []
    for feat in all_features:
        if not isinstance(feat, dict):
            continue
        key = (
            (feat.get("virus") or "").lower(),
            (feat.get("protein") or "").lower(),
            (feat.get("target_token") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(feat)

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

# llm_groq.py
# Two-pass, LLM-only mutation miner for biomedical papers using Groq Chat Completions.
# Pass 1: enumerate all mutation/segment tokens present in the paper (no regex discovery).
# Pass 2: for each token, attribute virus/strain, protein, effect, and provide grounded quotes.
#
# This version avoids 413 errors by:
#  - Chunking the paper for Pass-1 using (chunk_chars, overlap_chars) from meta
#  - Using compact, mutation-local windows for Pass-2 (instead of the whole paper)
#  - Retrying with progressively trimmed user content on 413 responses
#
# Public API (used by pipeline/batch_analyze.py):
#   - run_on_paper(paper_text: str, meta: Optional[dict]) -> dict
#   - clean_and_ground(raw: dict, full_text: str, *, restrict_to_paper=True,
#                      require_mutation_in_quote=True, min_confidence=0.0) -> dict
#
# Env:
#   - GROQ_API_KEY (required)

from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any, Optional

import requests

# Optional .env loading
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ===== Groq API configuration =====
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"  # change if needed

# ===== Low-level client with 413-safe retry =====

def _require_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Export it or put it in a .env file.")
    return api_key

def _post_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_require_key()}",
    }
    max_retries = 6
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
        # Groq compatibility hint
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
        # Explicitly bubble up 413 for outer trimming logic
        if resp.status_code == 413:
            raise requests.HTTPError("413 Payload Too Large", response=resp)
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded retry attempts contacting Groq API.")

def _payload_size(messages: List[Dict[str, str]]) -> int:
    # crude size proxy in bytes
    return sum(len(m.get("content", "")) for m in messages)

def _shrink_user_content(messages: List[Dict[str, str]], keep_bytes: int) -> List[Dict[str, str]]:
    """Trim only the *last* user message content to keep ~keep_bytes total."""
    if not messages:
        return messages
    total = _payload_size(messages)
    if total <= keep_bytes:
        return messages
    # Find last user message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            usr = messages[i].get("content", "")
            overshoot = total - keep_bytes
            if overshoot <= 0:
                return messages
            new_len = max(0, len(usr) - overshoot - 512)  # leave a small buffer
            messages = messages.copy()
            messages[i] = {"role": "user", "content": usr[:new_len]}
            return messages
    return messages

def _chat_retry_shrinking(messages: List[Dict[str, str]],
                          model: str,
                          temperature: float,
                          max_tokens: int,
                          *,
                          shrink_targets: List[int] = [20000, 14000, 10000, 7000, 5000]) -> str:
    """
    Call chat API; if 413 occurs, progressively shrink user content and retry.
    shrink_targets are total *characters* budget for messages (approx).
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        data = _post_chat(payload)
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 413:
            # Try shrinking budgets
            for budget in shrink_targets:
                shrunk = _shrink_user_content(messages, keep_bytes=budget)
                payload = {
                    "model": model,
                    "messages": shrunk,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                try:
                    data = _post_chat(payload)
                    return data["choices"][0]["message"]["content"]
                except requests.HTTPError as e2:
                    if getattr(e2, "response", None) is not None and e2.response.status_code == 413:
                        continue
                    raise
            # final failure if still too large
            raise
        raise

def chat_complete(messages: List[Dict[str, str]],
                  model: str = DEFAULT_MODEL,
                  temperature: float = 0.2,
                  max_tokens: int = 1024) -> str:
    """Normal client (kept for compatibility)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _post_chat(payload)
    return data["choices"][0]["message"]["content"]

# ===== Utilities =====

def _safe_json_loads(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                return None
    return None

def _unique_strs(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for s in seq:
        if not isinstance(s, str): continue
        t = s.strip()
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def _normalize_ws(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 500) -> List[str]:
    """Simple char-based chunking with overlap (sufficient for 413 avoidance)."""
    text = _normalize_ws(text)
    n = len(text)
    if n <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def _find_all_ci(hay: str, needle: str) -> List[int]:
    """All case-insensitive start indices of needle in hay."""
    if not hay or not needle:
        return []
    h = hay.lower(); n = needle.lower()
    out = []
    i = 0
    while True:
        j = h.find(n, i)
        if j == -1:
            break
        out.append(j)
        i = j + max(1, len(n))
    return out

def _windows_for_mutation(full_text: str,
                          mutation: str,
                          *,
                          left: int = 900,
                          right: int = 900,
                          max_windows: int = 6,
                          max_total_chars: int = 12000) -> str:
    """
    Build compact local contexts around each occurrence of `mutation`, concatenated, capped by count and total size.
    Ensures Pass-2 prompts stay small.
    """
    t = _normalize_ws(full_text)
    idxs = _find_all_ci(t, mutation)
    if not idxs:
        # fallback: take a head slice
        return t[:max_total_chars]

    pieces: List[str] = []
    used = 0
    for i in idxs[:max_windows]:
        start = max(0, i - left)
        end = min(len(t), i + len(mutation) + right)
        seg = t[start:end]
        if used + len(seg) + 2 > max_total_chars:
            break
        pieces.append(seg)
        used += len(seg) + 2
    return ("\n\n...\n\n".join(pieces))[:max_total_chars]

# ===== Pass 1: enumerate mutation tokens (LLM-only; no regex discovery) =====

def _pass1_messages(paper_text: str) -> List[Dict[str, str]]:
    sys = (
        "You extract mutation tokens from biomedical papers.\n"
        "Return ONLY normalized mutation/segment tokens that occur in the text, no commentary.\n"
        "Normalize to concise forms such as N501Y, D125A, K70*, del69-70, aa1396-1435, p.Asp125Ala, insA/insAla.\n"
        "Output STRICT JSON only:\n"
        "{ \"mutations\": [\"<token1>\", \"<token2>\", ...] }"
    )
    usr = (
        "Extract all mutation/segment tokens present in the PAPER_TEXT. "
        "Keep unique items, preserve canonical form, and ignore background organisms unless the token is explicitly stated.\n\n"
        "PAPER_TEXT:\n" + paper_text
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

# ===== Pass 2: attribute per mutation with grounded quotes =====

_ALLOWED_CATEGORIES = {
    "RNA_synthesis", "virion_assembly", "binding", "replication",
    "infectivity", "virulence", "immune_evasion", "drug_interaction",
    "temperature_sensitivity", "activity_change", "modification", "other"
}

def _pass2_messages(local_context: str,
                    mutation: str,
                    pmid: Optional[str],
                    pmcid: Optional[str]) -> List[Dict[str, str]]:
    sys = (
        "You attribute a given mutation token to virus, strain/lineage, protein, and effect using ONLY the provided context from this paper.\n"
        "Be concise and specific. Provide 1–2 short quotes (≤ ~20 words each) that appear verbatim, and include "
        "the exact mutation token in at least one quote. Use the controlled set of effect categories."
    )
    usr = (
        "Given the LOCAL_CONTEXT (excerpts from the paper) and the target MUTATION, extract one structured item.\n"
        "STRICT JSON only:\n"
        "{\n"
        "  \"paper\": {\"pmid\": " + json.dumps(pmid) + ", \"pmcid\": " + json.dumps(pmcid) + ", \"title\": null,\n"
        "             \"virus_candidates\": [], \"protein_candidates\": []},\n"
        "  \"sequence_features\": [\n"
        "    {\n"
        "      \"virus\": \"<organism or null>\",\n"
        "      \"source_strain\": \"<serotype/lineage/strain or null>\",\n"
        "      \"protein\": \"<protein or gene symbol>\",\n"
        "      \"mutation\": " + json.dumps(mutation) + ",\n"
        "      \"position\": \"<integer or null>\",\n"
        "      \"effect_category\": \"<RNA_synthesis|virion_assembly|binding|replication|infectivity|virulence|immune_evasion|drug_interaction|temperature_sensitivity|activity_change|modification|other>\",\n"
        "      \"effect_summary\": \"<1-2 sentences specific to this paper>\",\n"
        "      \"mechanism_hypothesis\": \"<brief or null>\",\n"
        "      \"experiment_context\": {\"system\": \"<cell/animal/in vitro/in silico>\", \"assay\": \"<method>\", \"temperature\": \"<value or null>\"},\n"
        "      \"evidence_quotes\": [\"<short quote 1>\", \"<short quote 2>\"],\n"
        "      \"cross_refs\": [{\"pmid\": null, \"note\": null}]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If you cannot support the mutation with a grounded quote in the LOCAL_CONTEXT, return an empty sequence_features list.\n\n"
        "MUTATION: " + mutation + "\n\n"
        "LOCAL_CONTEXT:\n" + local_context
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

# ===== Public: two-pass LLM pipeline with chunking and windows =====

def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Two-pass, LLM-only pipeline that avoids 413 payloads.

    Pass-1 (mutations):
      - Chunk paper using meta['chunk_chars'] (default 12000) and meta['overlap_chars'] (default 500).
      - Call LLM on each chunk and union the mutations.

    Pass-2 (attribution):
      - For each mutation, build compact LOCAL_CONTEXT windows around each occurrence in the full text.
      - Call LLM with LOCAL_CONTEXT (not the full paper).

    Notes:
      - Virus/protein filters are intentionally ignored/removed (see app/pipeline callers) :contentReference[oaicite:2]{index=2}.
      - meta can carry {pmid, pmcid, exhaustive, chunk_chars, overlap_chars, delay_ms}; only those are used here.
    """
    meta = meta or {}
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    delay_ms = int(meta.get("delay_ms") or 0)
    chunk_chars = int(meta.get("chunk_chars") or 12000)
    overlap_chars = int(meta.get("overlap_chars") or 500)

    paper_text = _normalize_ws(paper_text)

    # ----- Pass 1: enumerate mutations over chunks -----
    mutations: List[str] = []
    seen = set()
    for chunk in _chunk_text(paper_text, max_chars=chunk_chars, overlap=overlap_chars):
        msgs = _pass1_messages(chunk)
        try:
            raw1 = _chat_retry_shrinking(msgs, DEFAULT_MODEL, 0.2, 700, shrink_targets=[18000, 12000, 8000, 6000, 4500])
        except Exception as e:
            # If a single chunk still fails, skip that chunk but continue
            # (this is very unlikely with the budgets above)
            # You could log e if you surface logs in the UI.
            continue
        j1 = _safe_json_loads(raw1) or {"mutations": []}
        for m in j1.get("mutations") or []:
            if isinstance(m, str):
                t = m.strip()
                if t and t not in seen:
                    seen.add(t); mutations.append(t)
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    # ----- Pass 2: per-mutation attribution with local contexts -----
    seq_feats: List[Dict[str, Any]] = []
    for m in mutations:
        local_ctx = _windows_for_mutation(
            paper_text, m,
            left=900, right=900, max_windows=6, max_total_chars=min(12000, chunk_chars)
        )
        msgs2 = _pass2_messages(local_ctx, m, pmid, pmcid)
        try:
            raw2 = _chat_retry_shrinking(msgs2, DEFAULT_MODEL, 0.2, 1200, shrink_targets=[18000, 12000, 9000, 7000, 5000])
        except Exception:
            # If even local context trips 413 (shouldn't), try a harsher window
            local_ctx = _windows_for_mutation(
                paper_text, m, left=600, right=600, max_windows=4, max_total_chars=7000
            )
            msgs2 = _pass2_messages(local_ctx, m, pmid, pmcid)
            raw2 = _chat_retry_shrinking(msgs2, DEFAULT_MODEL, 0.2, 900, shrink_targets=[9000, 7000, 5000, 3500])

        j2 = _safe_json_loads(raw2) or {}
        feats = (j2.get("sequence_features") or []) if isinstance(j2, dict) else []
        for f in feats:
            if isinstance(f, dict):
                seq_feats.append(f)
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    return {
        "paper": {"pmid": pmid, "pmcid": pmcid, "title": None, "virus_candidates": [], "protein_candidates": []},
        "sequence_features": seq_feats
    }

# ===== Minimal cleaner/grounder (hallucination guard + confidence heuristic) =====

_ALLOWED_CATEGORIES = _ALLOWED_CATEGORIES  # reuse

def clean_and_ground(raw: Dict[str, Any],
                     full_text: str,
                     *,
                     restrict_to_paper: bool = True,
                     require_mutation_in_quote: bool = True,
                     min_confidence: float = 0.0) -> Dict[str, Any]:
    """
    Keep only items that (a) have a mutation token and (b) provide at least one quote that
    appears verbatim in the text (case-insensitive). If require_mutation_in_quote=True,
    the accepted quote must also contain the exact mutation token. Confidence is a simple
    heuristic: quotes present (0.6) + has position (0.1) + has context (0.1) + has category (0.1).
    """
    paper = (raw or {}).get("paper") or {
        "pmid": None, "pmcid": None, "title": None, "virus_candidates": [], "protein_candidates": []
    }
    feats = (raw or {}).get("sequence_features") or []
    kept = []

    norm_text = _normalize_ws(full_text).lower() if isinstance(full_text, str) else ""
    for f in feats:
        if not isinstance(f, dict):
            continue
        mut = (f.get("mutation") or "").strip()
        if not mut:
            continue

        quotes = [q for q in (f.get("evidence_quotes") or []) if isinstance(q, str) and q.strip()]
        quote_ok = False
        if quotes and isinstance(full_text, str):
            for q in quotes:
                ql = _normalize_ws(q).lower()
                if ql in norm_text and (not require_mutation_in_quote or (mut.lower() in ql)):
                    quote_ok = True
                    break
        if not quote_ok:
            continue

        # normalize/validate category
        cat = (f.get("effect_category") or "").strip()
        if cat not in _ALLOWED_CATEGORIES:
            f["effect_category"] = "other"

        # simple confidence
        conf = 0.0
        if quotes: conf += 0.6
        if isinstance(f.get("position"), int): conf += 0.1
        ctx = f.get("experiment_context") or {}
        if any(ctx.get(k) for k in ("system", "assay", "temperature")): conf += 0.1
        if f.get("effect_category"): conf += 0.1
        conf = min(conf, 1.0)

        if conf < float(min_confidence or 0.0):
            continue

        f["confidence"] = conf
        kept.append(f)

    return {"paper": paper, "sequence_features": kept}

__all__ = [
    "chat_complete",
    "run_on_paper",
    "clean_and_ground",
    "DEFAULT_MODEL",
    "GROQ_API_URL",
]

# llm_gemini.py
# Two-pass, LLM-only mutation miner using Google Gemini (drop-in replacement for llm_groq.py)
# Public API kept identical: run_on_paper(paper_text, meta) and clean_and_ground(raw, full_text, ...)
from __future__ import annotations

import os
import re
import json
import time
import threading
from time import monotonic as _mono
from typing import List, Dict, Any, Optional
from typing import Tuple

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ===== Gemini config =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Soft rate limits (defaults sized for free tier; tune via .env)
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "12"))       # requests/minute (free tier ~15)
GEMINI_TPM = int(os.getenv("GEMINI_TPM", "200000"))   # tokens/minute (free tier ~250k)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in your environment or .env")

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception as e:
    raise RuntimeError("Missing dependency: pip install google-generativeai") from e

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(DEFAULT_MODEL)

# ===== Soft RPM/TPM gates =====
_RATE_LOCK = threading.Lock()
_MIN_INTERVAL = 60.0 / GEMINI_RPM if GEMINI_RPM > 0 else 0.0
_WINDOW_START = _mono()
_TOKENS_USED = 0

def _approx_tokens(s: str) -> int:
    # crude: ~4 chars per token
    return int(len(s) / 4) if s else 0

def _rpm_gate():
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

# ===== Utilities (identical shapes to llm_groq.py) =====
_ALLOWED_CATEGORIES = {
    "RNA_synthesis", "virion_assembly", "binding", "replication",
    "infectivity", "virulence", "immune_evasion", "drug_interaction",
    "temperature_sensitivity", "activity_change", "modification", "other"
}

def _normalize_ws(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 500) -> List[str]:
    t = _normalize_ws(text)
    if len(t) <= max_chars:
        return [t]
    out, i, n = [], 0, len(t)
    while i < n:
        j = min(n, i + max_chars)
        out.append(t[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)
    return out

def _find_all_ci(hay: str, needle: str) -> List[int]:
    if not hay or not needle:
        return []
    h = hay.lower(); n = needle.lower()
    i = 0; out = []
    while True:
        j = h.find(n, i)
        if j == -1: break
        out.append(j); i = j + max(1, len(n))
    return out

def _windows_for_mutation(full_text: str,
                          mutation: str,
                          *,
                          left: int = 900,
                          right: int = 900,
                          max_windows: int = 6,
                          max_total_chars: int = 12000) -> str:
    t = _normalize_ws(full_text)
    idxs = _find_all_ci(t, mutation)
    if not idxs:
        return t[:max_total_chars]
    pieces: List[str] = []
    used = 0
    for i in idxs[:max_windows]:
        s = max(0, i - left); e = min(len(t), i + len(mutation) + right)
        seg = t[s:e]
        if used + len(seg) + 2 > max_total_chars: break
        pieces.append(seg); used += len(seg) + 2
    return ("\n\n...\n\n".join(pieces))[:max_total_chars]

def _safe_json_loads(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{"); end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except Exception:
                return None
    return None

# ===== Prompts (kept aligned with your llm_groq.py) =====
def _pass1_prompt(paper_text: str) -> str:
    sys = (
        "You extract sequence-related tokens from biomedical papers.\n"
        "Return ONLY normalized tokens that occur in the text, no commentary.\n"
        "Token types and normalization rules:\n"
        "  • mutations: concise forms such as N501Y, D125A, K70*, del69-70, aa1396-1435, p.Asp125Ala, insA/insAla.\n"
        "  • proteins: gene/protein symbols or names (e.g., BRCA1, NS1, E, Spike glycoprotein, RdRp).\n"
        "  • amino_acids: amino-acid mentions as residue tokens when possible (e.g., K417, Lys417, p.Lys417,\n"
        "    Ser315, S315) or bare names (Lysine, Ser) if no position is given.\n"
        "Include items even if they are not associated with any gene/disease/effect in the paper.\n"
        "Output STRICT JSON only:\n"
        "{\n"
        "  \"mutations\":   [\"<token>\", ...],\n"
        "  \"proteins\":    [\"<token>\", ...],\n"
        "  \"amino_acids\": [\"<token>\", ...]\n"
        "}"
    )
    usr = (
        "Extract all mutation, protein, and amino-acid tokens present in the PAPER_TEXT.\n"
        "Keep unique items, preserve canonical form, and ignore background organisms unless the token is explicitly stated.\n\n"
        "PAPER_TEXT:\n" + paper_text
    )
    return sys + "\\n\\n" + usr


def _pass2_prompt(local_context: str, target_token: str, pmid: Optional[str], pmcid: Optional[str]) -> str:
    sys = (
        "You attribute a given biomedical TOKEN using ONLY the provided context from this paper.\n"
        "The TOKEN may be a mutation, a protein/gene, or an amino-acid residue mention.\n"
        "Be concise and specific. Provide 1–2 short quotes (≤ ~20 words each) that appear verbatim; "
        "at least one quote must include the exact TOKEN string. Use the controlled effect categories when applicable."
    )
    usr = (
        "Given the LOCAL_CONTEXT (excerpts from the paper) and the target TOKEN, extract one structured item.\n"
        "If the paper does not state an association (e.g., no disease/effect/protein linkage), still return an item with nulls.\n"
        "STRICT JSON only:\n"
        "{\n"
        f"  \"paper\": {{\"pmid\": {json.dumps(pmid)}, \"pmcid\": {json.dumps(pmcid)}, \"title\": null,\n"
        "             \"virus_candidates\": [], \"protein_candidates\": []},\n"
        "  \"sequence_features\": [\n"
        "    {\n"
        "      \"target_token\": " + json.dumps(target_token) + ",\n"
        "      \"target_type\": \"<mutation|protein|amino_acid>\",\n"
        "      \"virus\": \"<organism or null>\",\n"
        "      \"source_strain\": \"<serotype/lineage/strain or null>\",\n"
        "      \"protein\": \"<protein or gene symbol or null>\",\n"
        "      \"mutation\": \"<mutation token or null>\",\n"
        "      \"residue\": \"<amino-acid residue token or null>\",\n"
        "      \"position\": \"<integer or null>\",\n"
        "      \"effect_category\": \"<RNA_synthesis|virion_assembly|binding|replication|infectivity|virulence|immune_evasion|drug_interaction|temperature_sensitivity|activity_change|modification|other|null>\",\n"
        "      \"effect_summary\": \"<1-2 sentences specific to this paper or null>\",\n"
        "      \"mechanism_hypothesis\": \"<brief or null>\",\n"
        "      \"experiment_context\": {\"system\": \"<cell/animal/in vitro/in silico or null>\", \"assay\": \"<method or null>\", \"temperature\": \"<value or null>\"},\n"
        "      \"evidence_quotes\": [\"<short quote 1>\", \"<short quote 2>\"],\n"
        "      \"cross_refs\": [{\"pmid\": null, \"note\": null}]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"TOKEN: {target_token}\n\n"
        "If you cannot support the TOKEN with a grounded quote in LOCAL_CONTEXT, return an empty sequence_features list.\n\n"
        "LOCAL_CONTEXT:\n" + local_context
    )
    return sys + "\\n\\n" + usr


# ===== Gemini wrapper (with 429-aware retries + RPM/TPM gates) =====
def _parse_retry_delay_seconds(err_text: str) -> int:
    # Try to find explicit “retry in 44.53s” or retry_delay { seconds: 44 }
    m = re.search(r"retry(?:\s+in)?\s*([0-9]+(?:\.[0-9]+)?)\s*s", err_text, re.I)
    if m:
        try:
            return max(1, int(float(m.group(1))))
        except Exception:
            pass
    m2 = re.search(r"retry_delay\s*\{\s*seconds\s*:\s*([0-9]+)", err_text, re.I)
    if m2:
        try:
            return max(1, int(m2.group(1)))
        except Exception:
            pass
    # Fallback for per-minute limits
    return 60

def _gemini_complete(prompt_text: str, max_output_tokens: int) -> str:
    # Soft gates to avoid hitting 429 in the first place
    est = _approx_tokens(prompt_text) + int(max_output_tokens or 0)
    _rpm_gate()
    _tpm_gate(est)

    max_attempts = 5
    backoff = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _model.generate_content(
                contents=prompt_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="text/plain",
                )
            )
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e) or ""
            # Handle free-tier 429 with server-suggested delay
            if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
                delay = _parse_retry_delay_seconds(msg)
                print(f"[gemini] 429/quota on attempt {attempt}/{max_attempts}. Sleeping {delay}s…", flush=True)
                time.sleep(delay)
                continue
            # transient 5xx -> progressive backoff
            if any(x in msg for x in ("502", "503", "504")):
                print(f"[gemini] transient error {msg}. Backoff {backoff}s…", flush=True)
                time.sleep(backoff); backoff = min(backoff * 2, 60)
                continue
            # non-retryable
            raise
    # final failure
    raise RuntimeError("Gemini call failed after retries.")

# ===== Public: two-pass LLM pipeline (same signature as llm_groq.run_on_paper) =====
def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    delay_ms = int(meta.get("delay_ms") or 0)
    chunk_chars = int(meta.get("chunk_chars") or 12000)
    overlap_chars = int(meta.get("overlap_chars") or 500)

    paper_text = _normalize_ws(paper_text)

    # ---------- Pass 1: enumerate tokens over chunks ----------
    # We collect three categories: mutations, proteins, amino_acids.
    # 'seen' dedupes across all categories by literal token string.
    tokens: List[Tuple[str, str]] = []  # (token_type, token)
    seen: set[str] = set()

    for chunk in _chunk_text(paper_text, max_chars=chunk_chars, overlap=overlap_chars):
        prompt = _pass1_prompt(chunk)
        try:
            raw1 = _gemini_complete(prompt, max_output_tokens=700)
        except Exception:
            # Skip this chunk if model hiccups
            continue

        j1 = _safe_json_loads(raw1) or {"mutations": [], "proteins": [], "amino_acids": []}

        # Mutations
        for m in (j1.get("mutations") or []):
            if isinstance(m, str):
                t = m.strip()
                if t and t not in seen:
                    seen.add(t)  
                    tokens.append(("mutation", t))

        # Proteins
        for p in (j1.get("proteins") or []):
            if isinstance(p, str):
                t = p.strip()
                if t and t not in seen:
                    seen.add(t)
                    tokens.append(("protein", t))

        # Amino acids
        for aa in (j1.get("amino_acids") or []):
            if isinstance(aa, str):
                t = aa.strip()
                if t and t not in seen:
                    seen.add(t)
                    tokens.append(("amino_acid", t))

        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    # ---------- Pass 2: attribution per token (token-agnostic) ----------
    seq_feats: List[Dict[str, Any]] = []

    for tok_type, tok in tokens:
        # Reuse the same local-windowing helper (name mentions 'mutation' but it’s just string-based).
        local_ctx = _windows_for_mutation(
            paper_text, tok,
            left=900, right=900, max_windows=6, max_total_chars=min(12000, chunk_chars)
        )
        prompt2 = _pass2_prompt(local_ctx, tok, pmid, pmcid)
        try:
            raw2 = _gemini_complete(prompt2, max_output_tokens=1200)
        except Exception:
            # Try smaller window on failure
            local_ctx = _windows_for_mutation(
                paper_text, tok, left=600, right=600, max_windows=4, max_total_chars=7000
            )
            prompt2 = _pass2_prompt(local_ctx, tok, pmid, pmcid)
            raw2 = _gemini_complete(prompt2, max_output_tokens=900)

        j2 = _safe_json_loads(raw2) or {}
        feats = (j2.get("sequence_features") or []) if isinstance(j2, dict) else []
        for f in feats:
            if isinstance(f, dict):
                # Optionally annotate the detected type for downstream filters
                f.setdefault("target_type", tok_type)
                seq_feats.append(f)

        if delay_ms:
            time.sleep(delay_ms / 1000.0)

    return {
        "paper": {"pmid": pmid, "pmcid": pmcid, "title": None, "virus_candidates": [], "protein_candidates": []},
        "sequence_features": seq_feats
    }


# ===== Cleaner (same math as llm_groq.clean_and_ground) =====
def clean_and_ground(raw: Dict[str, Any],
                     full_text: str,
                     *,
                     restrict_to_paper: bool = True,
                     require_mutation_in_quote: bool = True,
                     min_confidence: float = 0.0) -> Dict[str, Any]:
    paper = (raw or {}).get("paper") or {
        "pmid": None, "pmcid": None, "title": None, "virus_candidates": [], "protein_candidates": []
    }
    feats = (raw or {}).get("sequence_features") or []
    kept = []

    norm_text = _normalize_ws(full_text).lower() if isinstance(full_text, str) else ""
    for f in feats:
        if not isinstance(f, dict):
            continue

        # Prefer the explicit target_token; otherwise fall back to mutation
        token = (f.get("target_token") or f.get("mutation") or "").strip()
        if not token:
            continue

        # Quote grounding: at least one short quote appears verbatim in the paper
        quotes = [q for q in (f.get("evidence_quotes") or []) if isinstance(q, str) and q.strip()]
        quote_ok = False
        if quotes and isinstance(full_text, str):
            for q in quotes:
                ql = _normalize_ws(q).lower()
                # If require_mutation_in_quote is set, interpret it as "the token must be present"
                if ql in norm_text and (not require_mutation_in_quote or (token.lower() in ql)):
                    quote_ok = True
                    break
        if not quote_ok:
            continue

        # Normalize effect category
        cat = (f.get("effect_category") or "").strip()
        if cat and cat not in _ALLOWED_CATEGORIES:
            f["effect_category"] = "other"

        # Simple confidence heuristic (unchanged)
        conf = 0.0
        if quotes: conf += 0.6
        # position may be str or int; count it if it’s an integer-like
        pos = f.get("position")
        try:
            if isinstance(pos, int) or (isinstance(pos, str) and pos.strip().isdigit()):
                conf += 0.1
        except Exception:
            pass
        ctx = f.get("experiment_context") or {}
        if any(ctx.get(k) for k in ("system", "assay", "temperature")): conf += 0.1
        if f.get("effect_category"): conf += 0.1
        f["confidence"] = min(conf, 1.0)

        if f["confidence"] < float(min_confidence or 0.0):
            continue

        # Back-compat: if this is a non-mutation token, keep fields consistent
        # so old consumers that expect "mutation" don’t break.
        if not f.get("mutation") and f.get("target_token"):
            if f.get("target_type") == "mutation":
                f["mutation"] = f["target_token"]

        kept.append(f)

    return {"paper": paper, "sequence_features": kept}


__all__ = [
    "run_on_paper",
    "clean_and_ground",
    "DEFAULT_MODEL",
]

# llm/gemini.py - Optimized version
# Key changes: compiled regexes, reduced redundant calls, cached lookups

from __future__ import annotations

import os
import re
import json
import time
from llm.prompts import PROMPTS
import threading
from time import monotonic as _mono
from typing import List, Dict, Any, Optional, Tuple, Set

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

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in your environment or .env")

try:
    import google.generativeai as genai
except Exception as e:
    raise RuntimeError("Missing dependency: pip install google-generativeai") from e

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(DEFAULT_MODEL)

_RATE_LOCK = threading.Lock()
_MIN_INTERVAL = 60.0 / GEMINI_RPM if GEMINI_RPM > 0 else 0.0
_WINDOW_START = _mono()
_TOKENS_USED = 0


# ========== Rate limiting ==========
def _approx_tokens(s: str) -> int:
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


# ========== Gemini API wrapper ==========
if "_gemini_complete" not in globals():
    def _gemini_complete(prompt: str, max_output_tokens: int = 8192) -> str:
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


# ========== Constants & Compiled Regexes (OPTIMIZATION) ==========
_ALLOWED_CATEGORIES = {
    "RNA_synthesis", "virion_assembly", "binding", "replication",
    "infectivity", "virulence", "immune_evasion", "drug_interaction",
    "temperature_sensitivity", "activity_change", "modification", "other"
}

_ALLOWED_CONTINUITY = {"continuous", "discontinuous", "point", "unknown"}

# Compile once for performance
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_MUT_TOKEN_RE = re.compile(r"\b(?:[A-Z][0-9]{1,5}[A-Z*]|\d{1,5}[A-Z])\b")
_HGVS_RE = re.compile(r"\bp\.[A-Z][a-z]{2}\d{1,5}[A-Z][a-z]{2}\b")
_PROTEIN_RE = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9\-\/]{1,30})\s+(protein|glycoprotein|polyprotein|capsid|envelope|domain|subunit|chain|peptide|complex)\b",
    re.IGNORECASE
)
_RESIDUE_RE = re.compile(r"\b(?:[A-Z][a-z]{2}|[A-Z])\s*(\d{1,5})\b")
_RANGE_RE = re.compile(r"\b(\d{1,5})\s*[-–—]\s*(\d{1,5})\s*(?:aa|amino acids?|residues?)\b", re.IGNORECASE)
_RANGE_WORD_RE = re.compile(r"\bresidues?\s+(\d{1,5})\s*(?:to|through|–|-)\s*(\d{1,5})\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"\b(\d{1,5})\s*(?:aa|amino acids?)\b", re.IGNORECASE)
_MOTIF_RE = re.compile(r"\b[A-Z][A-ZxX\-]{2,9}\b")
_AA_SINGLE_RE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])[-/]?(?:residue|position)?[-/]?(?P<pos>\d{1,5})\b", re.IGNORECASE)

# Amino acid lookup table (cached)
_AA_NAME_TO_CODES = {
    "ala": ("A", "Ala"), "alanine": ("A", "Ala"),
    "arg": ("R", "Arg"), "arginine": ("R", "Arg"),
    "asn": ("N", "Asn"), "asparagine": ("N", "Asn"),
    "asp": ("D", "Asp"), "aspartate": ("D", "Asp"), "aspartic acid": ("D", "Asp"),
    "cys": ("C", "Cys"), "cysteine": ("C", "Cys"),
    "gln": ("Q", "Gln"), "glutamine": ("Q", "Gln"),
    "glu": ("E", "Glu"), "glutamate": ("E", "Glu"), "glutamic acid": ("E", "Glu"),
    "gly": ("G", "Gly"), "glycine": ("G", "Gly"),
    "his": ("H", "His"), "histidine": ("H", "His"),
    "ile": ("I", "Ile"), "isoleucine": ("I", "Ile"),
    "leu": ("L", "Leu"), "leucine": ("L", "Leu"),
    "lys": ("K", "Lys"), "lysine": ("K", "Lys"),
    "met": ("M", "Met"), "methionine": ("M", "Met"),
    "phe": ("F", "Phe"), "phenylalanine": ("F", "Phe"),
    "pro": ("P", "Pro"), "proline": ("P", "Pro"),
    "ser": ("S", "Ser"), "serine": ("S", "Ser"),
    "thr": ("T", "Thr"), "threonine": ("T", "Thr"),
    "trp": ("W", "Trp"), "tryptophan": ("W", "Trp"),
    "tyr": ("Y", "Tyr"), "tyrosine": ("Y", "Tyr"),
    "val": ("V", "Val"), "valine": ("V", "Val"),
    "sec": ("U", "Sec"), "selenocysteine": ("U", "Sec"),
    "pyl": ("O", "Pyl"), "pyrrolysine": ("O", "Pyl"),
    "stop": ("*", "Ter"), "ochre": ("*", "Ter"), "amber": ("*", "Ter"), "opal": ("*", "Ter"),
}

_AA_NAME_PATTERN = "|".join(
    sorted((re.escape(k) for k in _AA_NAME_TO_CODES.keys()), key=len, reverse=True)
)

_SPELLED_MUT_RE = re.compile(
    rf"(?P<from>{_AA_NAME_PATTERN})\s*(?:residue\s*)?(?P<pos>\d{{1,5}})"
    r"(?:\s*(?:to|into|->|→|for|with|by|in|changed\s+to|replaced\s+by|replaced\s+with|substituted\s+with|substituted\s+by))"
    r"\s*(?:a\s+|an\s+)?(?P<to>{_AA_NAME_PATTERN})(?:\s+residue|\s+residues)?",
    re.IGNORECASE
)

_AA_WORD_POS_RE = re.compile(
    rf"\b(?P<name>{_AA_NAME_PATTERN})(?:[\s\-]*(?:residue|position|site))?\s*(?P<pos>\d{{1,5}})\b",
    re.IGNORECASE
)


# ========== Helper functions ==========
def _normalize_ws(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def _normalize_for_match(text: str) -> str:
    """Downcase, strip spacing/punctuation, fold accents."""
    if not isinstance(text, str) or not text:
        return ""
    import unicodedata
    try:
        nfkd = unicodedata.normalize("NFKD", text)
    except Exception:
        nfkd = text
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii", "ignore")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def _expand_to_sentence(full_text: str, fragment: str) -> Optional[str]:
    """Locate the sentence containing the fragment."""
    if not isinstance(full_text, str) or not isinstance(fragment, str):
        return None
    frag_norm = _normalize_ws(fragment).lower()
    if not frag_norm:
        return None

    corpus = full_text.replace("\n", " ").replace("\r", " ")
    sentences = _SENTENCE_SPLIT_RE.split(corpus)
    for sentence in sentences:
        sent_norm = _normalize_ws(sentence).lower()
        if frag_norm in sent_norm:
            return sentence.strip()
    return None


def _lookup_aa_codes(name: str) -> Optional[Tuple[str, str]]:
    return _AA_NAME_TO_CODES.get(name.lower())


def _extract_spelled_mutation(text: str) -> Optional[Tuple[str, str]]:
    if not isinstance(text, str):
        return None
    normalized = text.replace("–", "-").replace("—", "-").replace("→", "->")
    for match in _SPELLED_MUT_RE.finditer(normalized):
        pos = match.group("pos")
        aa_from = _lookup_aa_codes(match.group("from"))
        aa_to = _lookup_aa_codes(match.group("to"))
        if not aa_from or not aa_to:
            continue
        one_from, three_from = aa_from
        one_to, three_to = aa_to
        if not pos or not one_from or not one_to:
            continue
        short = f"{one_from}{pos}{one_to}"
        hgvs = f"p.{three_from}{pos}{three_to}"
        return hgvs, short
    return None


def _scan_text_candidates(text: str) -> Dict[str, List[str]]:
    """Pre-scan for candidate mentions to guide LLM."""
    if not isinstance(text, str):
        return {}
    
    candidates: Dict[str, Set[str]] = {
        "mutation_tokens": set(),
        "hgvs_tokens": set(),
        "protein_terms": set(),
        "residue_numbers": set(),
        "motif_candidates": set(),
        "amino_acid_mentions": set(),
    }
    
    for m in _MUT_TOKEN_RE.finditer(text):
        candidates["mutation_tokens"].add(m.group(0))
    
    for m in _HGVS_RE.finditer(text):
        candidates["hgvs_tokens"].add(m.group(0))
    
    for m in _PROTEIN_RE.finditer(text):
        term = f"{m.group(1)} {m.group(2)}"
        candidates["protein_terms"].add(term)
    
    for m in _RESIDUE_RE.finditer(text):
        candidates["residue_numbers"].add(m.group(1))
    
    for m in _RANGE_RE.finditer(text):
        candidates["residue_numbers"].add(f"{m.group(1)}-{m.group(2)}")
    
    for m in _RANGE_WORD_RE.finditer(text):
        candidates["residue_numbers"].add(f"{m.group(1)}-{m.group(2)}")
    
    for m in _COUNT_RE.finditer(text):
        candidates["residue_numbers"].add(m.group(1))
    
    for m in _MOTIF_RE.finditer(text):
        token = m.group(0)
        if len(token) >= 3:
            candidates["motif_candidates"].add(token)
    
    for m in _AA_SINGLE_RE.finditer(text):
        aa = m.group(1).upper()
        pos = m.group("pos")
        candidates["amino_acid_mentions"].add(f"{aa}{pos}")
    
    for m in _AA_WORD_POS_RE.finditer(text):
        name = m.group("name")
        pos = m.group("pos")
        codes = _lookup_aa_codes(name)
        if codes:
            one_letter, three_letter = codes
            candidates["amino_acid_mentions"].add(f"{three_letter}{pos}")
            candidates["amino_acid_mentions"].add(f"{one_letter}{pos}")

    def _trim(seq: Set[str], limit: int = 200) -> List[str]:
        return sorted(seq)[:limit] if seq else []

    return {
        "mutation_tokens": _trim(candidates["mutation_tokens"]),
        "hgvs_tokens": _trim(candidates["hgvs_tokens"]),
        "protein_terms": _trim(candidates["protein_terms"]),
        "residue_numbers": _trim(candidates["residue_numbers"]),
        "motif_candidates": _trim(candidates["motif_candidates"]),
        "amino_acid_mentions": _trim(candidates["amino_acid_mentions"]),
    }


def _token_context_windows(full_text: str,
                           token: str,
                           *,
                           left: int = 900,
                           right: int = 900,
                           max_windows: int = 4,
                           max_total_chars: int = 12000) -> str:
    """Extract context windows around a token."""
    corpus = _normalize_ws(full_text)
    if not corpus or not token:
        return corpus
    
    lower = corpus.lower()
    needle = token.lower()
    indexes = []
    start = 0
    
    while True:
        idx = lower.find(needle, start)
        if idx == -1:
            break
        indexes.append(idx)
        start = idx + max(1, len(needle))
    
    if not indexes:
        return corpus[:max_total_chars]
    
    segments: List[str] = []
    used = 0
    
    for idx in indexes[:max_windows]:
        seg_start = max(0, idx - left)
        seg_end = min(len(corpus), idx + len(needle) + right)
        snippet = corpus[seg_start:seg_end]
        addition = len(snippet) + 2
        if used + addition > max_total_chars:
            break
        segments.append(snippet)
        used += addition
    
    return ("\n\n...\n\n".join(segments))[:max_total_chars]


def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
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


def _safe_json_value(raw: str):
    """Parse JSON value (dict or array), tolerating truncation."""
    if not isinstance(raw, str):
        return None

    s = raw.strip()

    # Strip code fences
    if s.startswith("```"):
        s = s[3:]
        if "\n" in s:
            s = s.split("\n", 1)[1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract balanced structures
    def _extract_balanced(text, open_ch, close_ch):
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == close_ch:
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1:
                            return text[start:i+1]
        return None

    # Try array
    arr = _extract_balanced(s, "[", "]")
    if arr:
        try:
            return json.loads(arr)
        except Exception:
            pass

    # Salvage objects from truncated array
    objs = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{":
            depth = 0
            in_str = False
            esc = False
            start = i
            j = i
            while j < n:
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = s[start:j+1]
                            try:
                                obj = json.loads(candidate)
                                objs.append(obj)
                            except Exception:
                                pass
                            i = j
                            break
                j += 1
        i += 1

    if objs:
        return objs

    # Try dict
    dic = _extract_balanced(s, "{", "}")
    if dic:
        try:
            return json.loads(dic)
        except Exception:
            pass

    return None


def _normalize_prompt_feature(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize feature object to expected schema."""
    if not isinstance(obj, dict):
        return None

    obj.setdefault("virus", None)
    obj.setdefault("protein", None)

    feat = obj.get("feature")
    if not isinstance(feat, dict):
        feat = {}
        obj["feature"] = feat

    feat.setdefault("name_or_label", None)
    feat.setdefault("type", "other")

    continuity = feat.get("continuity") or "unknown"
    if continuity not in _ALLOWED_CONTINUITY:
        feat["continuity"] = "unknown"
    else:
        feat["continuity"] = continuity

    for key in ("residue_positions", "specific_residues", "variants"):
        value = feat.get(key)
        if not isinstance(value, list):
            feat[key] = []
    
    if "motif_pattern" not in feat:
        feat["motif_pattern"] = None

    eff = obj.get("effect_or_function")
    if not isinstance(eff, dict):
        eff = {}
        obj["effect_or_function"] = eff
    eff.setdefault("description", None)
    eff.setdefault("category", "unknown")
    eff.setdefault("direction", "unknown")
    eff.setdefault("evidence_level", "unknown")

    if not isinstance(obj.get("evidence_snippet"), str):
        obj["evidence_snippet"] = ""

    conf = obj.get("confidence")
    if not isinstance(conf, dict):
        conf = {}
        obj["confidence"] = conf
    conf.setdefault("score_0_to_1", 0.0)
    conf.setdefault("rationale", "")

    interactions = obj.get("interactions")
    if not isinstance(interactions, dict):
        interactions = {}
        obj["interactions"] = interactions
    interactions.setdefault("partner_protein", None)
    interactions.setdefault("interaction_type", None)
    interactions.setdefault("context", None)

    return obj


def _collect_extracted_tokens(features: List[Dict[str, Any]]) -> Set[str]:
    """Collect all tokens already extracted."""
    tokens: Set[str] = set()
    for f in features:
        if not isinstance(f, dict):
            continue
        
        mutation = f.get("mutation")
        if isinstance(mutation, str):
            tokens.add(mutation.strip())
        
        target_token = f.get("target_token")
        if isinstance(target_token, str):
            tokens.add(target_token.strip())
        
        feat = f.get("feature") or {}
        name = feat.get("name_or_label")
        if isinstance(name, str):
            tokens.add(name.strip())
        
        variants = feat.get("variants") or []
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, str):
                    tokens.add(v.strip())
        
        residues = feat.get("specific_residues") or []
        if isinstance(residues, list):
            for r in residues:
                if isinstance(r, dict):
                    pos_val = r.get("position")
                    aa_val = r.get("aa")
                    if isinstance(aa_val, str) and pos_val is not None:
                        tokens.add(f"{aa_val}{pos_val}")
        
        residue_positions = feat.get("residue_positions") or []
        if isinstance(residue_positions, list):
            for rp in residue_positions:
                if isinstance(rp, dict):
                    start = rp.get("start")
                    end = rp.get("end")
                    if start is not None and end is not None:
                        if start == end:
                            tokens.add(str(start))
                        else:
                            tokens.add(f"{start}-{end}")
    
    return {t for t in tokens if t}


def _pass2_prompt(full_text: str,
                  target_token: str,
                  pmid: Optional[str],
                  pmcid: Optional[str],
                  token_type: str,
                  scan_candidates: Optional[Dict[str, List[str]]] = None) -> str:
    """Build prompt using the analyst template (UNCHANGED)."""
    wrapper = (PROMPTS.chunking_wrapper or "").strip()
    payload = (full_text or "").strip()

    meta = []
    if pmid:
        meta.append(f"PMID: {pmid}")
    if pmcid:
        meta.append(f"PMCID: {pmcid}")
    meta_block = "\n".join(meta)
    
    if meta_block:
        payload = f"{payload}\n\n{meta_block}" if payload else meta_block

    template = PROMPTS.analyst_prompt.strip()
    if "{TEXT}" in template:
        body = template.replace("{TEXT}", payload)
    else:
        body = template + ("\n\nTEXT\n" + payload if payload else "")

    if scan_candidates:
        hints = {k: v for k, v in scan_candidates.items() if v}
        if hints:
            hints_json = json.dumps(hints, ensure_ascii=False)
            body += ("\n\nKNOWN_MENTIONS (use internally to ensure coverage; do not list separately):\n"
                     f"{hints_json}")

    if wrapper:
        return f"{wrapper}\n\n{body}"
    return body


def _convert_bio_schema_feature(f: Dict[str, Any]) -> Dict[str, Any]:
    """Convert bioinformatician schema to legacy schema."""
    out: Dict[str, Any] = {}
    out["pmid_or_doi"] = f.get("pmid_or_doi")
    virus = f.get("virus") or ""
    protein = f.get("protein") or ""
    out["virus"] = virus
    out["protein"] = protein

    feat = f.get("feature") or {}
    name = feat.get("name_or_label") or ""
    ftype = feat.get("type") or ""
    variants = feat.get("variants") or []
    motif = feat.get("motif_pattern") or None

    mutation = None
    target_type = None
    target_token = None

    def _apply_mutation(hgvs_value: str, short_value: Optional[str] = None):
        nonlocal mutation, target_token, target_type
        mutation = hgvs_value
        target_token = short_value or hgvs_value
        target_type = "mutation"

    if variants and isinstance(variants, list):
        first_variant = variants[0]
        if isinstance(first_variant, str):
            spelled = _extract_spelled_mutation(first_variant)
            if spelled:
                hgvs_val, short_val = spelled
                _apply_mutation(hgvs_val, short_val)
            elif re.search(r"[A-Z][0-9]{1,5}[A-Z*]", first_variant):
                _apply_mutation(first_variant, first_variant)
            else:
                _apply_mutation(first_variant, first_variant)
    else:
        if isinstance(name, str) and re.search(r"[A-Z][0-9]{1,5}[A-Z*]", name):
            _apply_mutation(name, name)
        else:
            target_type = "protein" if protein else (ftype or "other")
            target_token = name or protein or ftype

    pos = None
    if isinstance(feat.get("specific_residues"), list) and feat["specific_residues"]:
        try:
            pos = int(feat["specific_residues"][0].get("position"))
        except Exception:
            pos = None
    
    if pos is None and isinstance(feat.get("residue_positions"), list) and feat["residue_positions"]:
        try:
            start = int(feat["residue_positions"][0].get("start"))
            pos = start
        except Exception:
            pos = None

    if not mutation:
        residues = feat.get("specific_residues") or []
        residue_positions = feat.get("residue_positions") or []
        residue_label = None
        
        if residues and isinstance(residues, list):
            first = residues[0] or {}
            pos_val = first.get("position")
            aa_val = first.get("aa")
            if pos_val is not None:
                residue_label = f"{aa_val}{pos_val}" if isinstance(aa_val, str) and aa_val else str(pos_val)
        elif residue_positions and isinstance(residue_positions, list):
            first = residue_positions[0] or {}
            start = first.get("start")
            end = first.get("end")
            if start is not None and end is not None:
                residue_label = f"{start}-{end}" if start != end else str(start)
        
        if residue_label:
            if not target_token:
                target_token = residue_label
            if target_type in (None, "", "protein", "other"):
                target_type = "amino_acid"

    eff = f.get("effect_or_function") or {}
    out["effect_summary"] = eff.get("description") or ""
    out["effect_category"] = eff.get("category") or ""
    
    direction = eff.get("direction")
    if direction and isinstance(direction, str) and direction.lower() not in ("unknown", "none"):
        out["effect_summary"] = (out["effect_summary"] + f" (direction: {direction})").strip()

    snippet = f.get("evidence_snippet") or ""
    if snippet:
        out["evidence_quotes"] = [snippet]

    conf = f.get("confidence") or {}
    try:
        out["confidence"] = float(conf.get("score_0_to_1"))
    except Exception:
        pass

    if not mutation:
        search_texts: List[str] = []
        if isinstance(name, str):
            search_texts.append(name)
        if isinstance(variants, list):
            search_texts.extend([v for v in variants if isinstance(v, str)])
        if out.get("effect_summary"):
            search_texts.append(out["effect_summary"])
        if snippet:
            search_texts.append(snippet)
        for extra in f.get("evidence_quotes") or []:
            if isinstance(extra, str):
                search_texts.append(extra)
        
        for text in search_texts:
            spelled = _extract_spelled_mutation(text)
            if spelled:
                hgvs_val, short_val = spelled
                _apply_mutation(hgvs_val, short_val)
                break

    out["mutation"] = mutation
    out["target_token"] = target_token
    out["target_type"] = target_type
    out["position"] = pos
    if motif:
        out["motif"] = motif

    return out


def clean_and_ground(raw: Dict[str, Any],
                     full_text: str,
                     *,
                     restrict_to_paper: bool = True,
                     require_mutation_in_quote: bool = False,
                     require_target_in_quote: Optional[bool] = None,
                     min_confidence: float = 0.0) -> Dict[str, Any]:
    """Clean and validate extracted features (UNCHANGED LOGIC)."""
    if require_target_in_quote is None:
        require_target_in_quote = require_mutation_in_quote
    
    paper = (raw or {}).get("paper") or {
        "pmid": None, "pmcid": None, "title": None, 
        "virus_candidates": [], "protein_candidates": []
    }
    
    feats = (raw or {}).get("sequence_features") or []
    
    # Convert bio schema if needed
    if feats and isinstance(feats, list) and isinstance(feats[0], dict) and 'feature' in feats[0]:
        feats = [_convert_bio_schema_feature(x) for x in feats]
    
    kept = []
    norm_text = _normalize_ws(full_text).lower() if isinstance(full_text, str) else ""
    norm_text_ascii = _normalize_for_match(full_text) if isinstance(full_text, str) else ""
    
    for f in feats:
        if not isinstance(f, dict):
            continue
        
        token = (f.get("target_token") or f.get("mutation") or "").strip()
        if not token:
            continue

        # Evidence validation
        quotes = [q for q in (f.get("evidence_quotes") or []) 
                 if isinstance(q, str) and q.strip()]
        expanded_quotes = []
        
        for q in quotes:
            expanded = _expand_to_sentence(full_text, q)
            expanded_quotes.append(expanded if expanded else q.strip())

        quote_ok = False
        if expanded_quotes and isinstance(full_text, str):
            for usable in expanded_quotes:
                ql = _normalize_ws(usable).lower()
                if ql in norm_text:
                    quote_ok = True
                    break
                ql_ascii = _normalize_for_match(usable)
                if ql_ascii and ql_ascii in norm_text_ascii:
                    quote_ok = True
                    break
        
        if not quote_ok and not require_mutation_in_quote:
            quote_ok = True
        
        if not quote_ok:
            continue
        
        if expanded_quotes:
            f["evidence_quotes"] = expanded_quotes

        # Normalize category
        cat = (f.get("effect_category") or "").strip()
        if cat and cat not in _ALLOWED_CATEGORIES:
            f["effect_category"] = "other"

        # Confidence scoring (simple, permissive)
        conf = 0.0
        if quotes:
            conf += 0.6
        
        pos = f.get("position")
        try:
            if isinstance(pos, int) or (isinstance(pos, str) and pos.strip().isdigit()):
                conf += 0.1
        except Exception:
            pass
        
        ctx = f.get("experiment_context") or {}
        if any(ctx.get(k) for k in ("system", "assay", "temperature")):
            conf += 0.1
        
        if f.get("effect_category"):
            conf += 0.1
        
        if f.get("mutation") in (None, "") and f.get("target_type") == "protein":
            conf = min(1.0, conf + 0.1)
        
        f["confidence"] = min(conf, 1.0)

        if f["confidence"] < float(min_confidence or 0.0):
            continue

        if not f.get("mutation") and f.get("target_token"):
            if f.get("target_type") == "mutation":
                f["mutation"] = f["target_token"]
        
        kept.append(f)
    
    return {"paper": paper, "sequence_features": kept}


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Single-call, multi-chunk extraction using the analyst prompt.
    UNCHANGED LOGIC - only performance optimizations.
    """
    meta = meta or {}
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    text_norm = _normalize_ws(paper_text or "")

    scan_candidates = _scan_text_candidates(text_norm)

    # Chunk parameters
    chunk_chars = int(meta.get("chunk_chars") or 100_000)
    overlap_chars = int(meta.get("overlap_chars") or 2_000)
    max_chunks = int(meta.get("max_chunks") or 4)

    chunks = list(_chunk_text(text_norm, max_chars=chunk_chars, overlap=overlap_chars))
    chunks = chunks[:max_chunks] if chunks else [text_norm]

    all_features: List[Any] = []
    
    # Process each chunk
    for idx, ch in enumerate(chunks, 1):
        prompt2 = _pass2_prompt(
            ch, target_token="", pmid=pmid, pmcid=pmcid, 
            token_type="paper", scan_candidates=scan_candidates
        )
        raw2 = _gemini_complete(prompt2, max_output_tokens=8192)
        j2 = _safe_json_value(raw2)

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
                norm = _normalize_prompt_feature(feat)
                if norm:
                    normalized.append(norm)
        
        all_features.extend(normalized)

    # Targeted follow-up for missed mutations
    extracted_tokens = _collect_extracted_tokens([f for f in all_features if isinstance(f, dict)])
    mutation_candidates = scan_candidates.get("mutation_tokens", []) if isinstance(scan_candidates, dict) else []
    hgvs_candidates = scan_candidates.get("hgvs_tokens", []) if isinstance(scan_candidates, dict) else []
    
    followup_tokens: List[str] = []
    for token in mutation_candidates + hgvs_candidates:
        if token and token not in extracted_tokens:
            followup_tokens.append(token)
    
    # Deduplicate
    deduped_follow = []
    seen_follow: Set[str] = set()
    for token in followup_tokens:
        if token not in seen_follow:
            seen_follow.add(token)
            deduped_follow.append(token)
    followup_tokens = deduped_follow

    # Process missed mutations
    for token in followup_tokens[:15]:
        context = _token_context_windows(
            text_norm, token, left=900, right=900, max_windows=4
        )
        hints = {"mutation_tokens": [token]}
        prompt_focus = _pass2_prompt(
            context, target_token=token, pmid=pmid, pmcid=pmcid, 
            token_type="mutation", scan_candidates=hints
        )
        raw_focus = _gemini_complete(prompt_focus, max_output_tokens=2048)
        parsed_focus = _safe_json_value(raw_focus)
        
        focus_feats: List[Dict[str, Any]] = []
        if isinstance(parsed_focus, dict) and isinstance(parsed_focus.get("sequence_features"), list):
            focus_feats = parsed_focus["sequence_features"]
        elif isinstance(parsed_focus, list):
            focus_feats = parsed_focus
        
        normalized_focus: List[Dict[str, Any]] = []
        for feat in focus_feats:
            if isinstance(feat, dict):
                norm_feat = _normalize_prompt_feature(feat)
                if norm_feat:
                    normalized_focus.append(norm_feat)
        
        if normalized_focus:
            all_features.extend(normalized_focus)
            extracted_tokens.update(_collect_extracted_tokens(normalized_focus))

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

    cleaned = clean_and_ground(
        raw, text_norm,
        restrict_to_paper=True,
        require_mutation_in_quote=False,
        min_confidence=float(meta.get("min_confidence") or 0.0),
    )
    
    return cleaned


__all__ = [
    "run_on_paper",
    "clean_and_ground",
    "DEFAULT_MODEL",
]
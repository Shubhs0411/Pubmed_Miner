import os
import re
import json
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

# Optional .env loading
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ===== Groq API configuration =====
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"  # adjust to a model you have access to

# ===== Utilities =====

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
        if resp.status_code == 404 and "unknown_url" in resp.text:
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
                sleep_s = backoff + (0.5 * os.urandom(1)[0] / 255.0)
            if attempt == max_retries:
                resp.raise_for_status()
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, 16.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exceeded retry attempts contacting Groq API.")

def chat_complete(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL,
                  temperature: float = 0.2, max_tokens: int = 1024) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _post_chat(payload)
    return data["choices"][0]["message"]["content"]

# ===== Canonical helpers (AA mapping) =====

AA_ONE = {
    "alanine": "A", "ala": "A",
    "arginine": "R", "arg": "R",
    "asparagine": "N", "asn": "N",
    "aspartic acid": "D", "aspartate": "D", "asp": "D",
    "cysteine": "C", "cys": "C",
    "glutamine": "Q", "gln": "Q",
    "glutamic acid": "E", "glutamate": "E", "glu": "E",
    "glycine": "G", "gly": "G",
    "histidine": "H", "his": "H",
    "isoleucine": "I", "ile": "I",
    "leucine": "L", "leu": "L",
    "lysine": "K", "lys": "K",
    "methionine": "M", "met": "M",
    "phenylalanine": "F", "phe": "F",
    "proline": "P", "pro": "P",
    "serine": "S", "ser": "S",
    "threonine": "T", "thr": "T",
    "tryptophan": "W", "trp": "W",
    "tyrosine": "Y", "tyr": "Y",
    "valine": "V", "val": "V",
    "*": "*",
}
AA_WORD_RE = r"(alanine|arginine|asparagine|aspartic acid|aspartate|cysteine|glutamine|glutamic acid|glutamate|glycine|histidine|isoleucine|leucine|lysine|methionine|phenylalanine|proline|serine|threonine|tryptophan|tyrosine|valine|ala|arg|asn|asp|cys|gln|glu|gly|his|ile|leu|lys|met|phe|pro|ser|thr|trp|tyr|val|\*)"

def _aa1(x: str) -> Optional[str]:
    return AA_ONE.get(x.strip().lower())

# ====== TEXT AUGMENTATION: add canonical tokens directly into PAPER_TEXT ======

def _augment_text_with_canonical_tokens(text: str) -> str:
    """
    Insert canonical mutation/segment tokens into natural-language phrases so that
    at least one verbatim quote can include the exact token (our sanitizer requires it).
    Examples inserted: (H39R), (A54E), (T280Y), (del42-59), (aa1396-1435), (S114A, W115A, D180A, T301A)
    """
    if not text:
        return text
    t = text

    # 1) "... histidine-39 with arginine" / "serine-112 to alanine"
    pat1 = re.compile(
        rf"(?:a\s+substitution\s+of\s+)?(?P<orig>{AA_WORD_RE})\s*[- ]?\s*(?P<pos>\d+)\s*(?:to|with|by|->|→)\s*(?P<dest>{AA_WORD_RE})",
        flags=re.IGNORECASE
    )
    def sub1(m):
        o = _aa1(m.group("orig")); d = _aa1(m.group("dest")); pos = m.group("pos")
        token = f"({o}{pos}{d})" if o and d and pos else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat1.sub(sub1, t)

    # 2) "threonine at position 280 ... substituted with tyrosine"
    pat2 = re.compile(
        rf"(?P<orig>{AA_WORD_RE})\s+at\s+position\s+(?P<pos>\d+)[^\.]{{0,120}}?(?:substitution|substituted|replaced)[^\.]{{0,40}}?\s+(?:with|by|to)\s+(?P<dest>{AA_WORD_RE})",
        flags=re.IGNORECASE
    )
    def sub2(m):
        o = _aa1(m.group("orig")); d = _aa1(m.group("dest")); pos = m.group("pos")
        token = f"({o}{pos}{d})" if o and d and pos else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat2.sub(sub2, t)

    # 3) "substitution at position 54 from alanine to glutamic acid"
    pat3 = re.compile(
        rf"(?:substitution|mutation)\s+at\s+position\s+(?P<pos>\d+)\s+from\s+(?P<orig>{AA_WORD_RE})\s+(?:to|with|by)\s+(?P<dest>{AA_WORD_RE})",
        flags=re.IGNORECASE
    )
    def sub3(m):
        o = _aa1(m.group("orig")); d = _aa1(m.group("dest")); pos = m.group("pos")
        token = f"({o}{pos}{d})" if o and d and pos else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat3.sub(sub3, t)

    # 4) Region deletion: "positions 42-59 ... deletion/removal"
    pat4 = re.compile(
        r"(positions?\s+(?P<start>\d+)\s*(?:-|to|–|—)\s*(?P<end>\d+)[^\.]{0,120}?(?:removal|deletion|deleted|remove))",
        flags=re.IGNORECASE
    )
    def sub4(m):
        a = m.group("start"); b = m.group("end")
        token = f"(del{a}-{b})" if a and b else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat4.sub(sub4, t)

    # 5) Segment mention "amino acids 1396 to 1435" -> (aa1396-1435)
    pat5 = re.compile(r"(amino acids?\s+(?P<start>\d+)\s*(?:to|-|–|—)\s*(?P<end>\d+))", flags=re.IGNORECASE)
    def sub5(m):
        a = m.group("start"); b = m.group("end")
        token = f"(aa{a}-{b})" if a and b else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat5.sub(sub5, t)

    # 6) "alanine substitutions at positions S114, W115, D180 or T301" -> (S114A, W115A, D180A, T301A)
    pat6 = re.compile(
        r"(alanine substitutions?\s+at\s+positions?\s+)(?P<list>(?:[A-Z]\d+(?:\s*,\s*)?)+(?:\s*(?:and|or)\s*[A-Z]\d+)?)",
        flags=re.IGNORECASE
    )
    def sub6(m):
        sites_str = m.group("list")
        sites = re.findall(r"[A-Z]\d+", sites_str)
        tokens = [f"{s}A" for s in sites]
        token = f" ({', '.join(tokens)})" if tokens else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s}{token}"
    t = pat6.sub(sub6, t)

    # 7) "proline to leucine ... amino acid 101" -> (P101L)
    pat7 = re.compile(
        rf"(?P<orig>{AA_WORD_RE})\s+(?:to|->|→)\s+(?P<dest>{AA_WORD_RE})[^\.]{{0,80}}?amino acid\s+(?P<pos>\d+)",
        flags=re.IGNORECASE
    )
    def sub7(m):
        o = _aa1(m.group("orig")); d = _aa1(m.group("dest")); pos = m.group("pos")
        token = f"({o}{pos}{d})" if o and d and pos else ""
        s = m.group(0)
        return s if not token or token.lower() in s.lower() else f"{s} {token}"
    t = pat7.sub(sub7, t)

    return t

# ===== Generalized Prompt (updated) =====

SCHEMA_EXPLANATION = """
You are an information extraction model for biological literature. Extract mutation/segment-level functional findings
from dengue virus ONLY (any serotype/strain). The PAPER_TEXT may contain helper canonical tokens we added in parentheses,
e.g. "(H39R)", "(A54E)", "(T280Y)", "(del42-59)", "(aa1396-1435)". Use those EXACT tokens in one of your evidence quotes.

SPLITTING & NORMALIZATION
- If a sentence describes multiple mutations together (e.g., "S114, W115, D180 or T301 alanine substitutions"),
  output a separate item for each, e.g., S114A, W115A, D180A, T301A.
- Region/segment edits are valid mutations, e.g., "del42-59" (deletion) or "aa1396-1435" (segment).
- Prefer precise direction/magnitude ("abolished", "reduced ~10-fold", "no effect"), and include system/assay if stated.

OUTPUT FORMAT (STRICT JSON ONLY)
{
  "paper": {
    "pmid": "<string or null>",
    "pmcid": "<string or null>",
    "title": "<string or null>",
    "virus_candidates": ["<organism name>", ...],
    "protein_candidates": ["<protein or gene>", ...]
  },
  "sequence_features": [
    {
      "virus": "<organism name or null>",
      "source_strain": "<strain/serotype/lineage if stated, else null>",
      "protein": "<protein or gene symbol>",
      "mutation": "<e.g., N501Y, D125A, del69-70, p.Asp125Ala, K70*, aa1396-1435>",
      "position": "<integer or null>",
      "effect_category": "<one of: RNA_synthesis | virion_assembly | binding | replication | infectivity | virulence | immune_evasion | drug_interaction | temperature_sensitivity | activity_change | modification | other>",
      "effect_summary": "<1-2 sentences>",
      "mechanism_hypothesis": "<brief mechanistic explanation if present, else null>",
      "experiment_context": { "system": "<cell/animal/in vitro/in silico>", "assay": "<method>", "temperature": "<value or null>" },
      "evidence_quotes": ["<short quote 1>", "<short quote 2>"],
      "cross_refs": [ { "pmid": "<numeric PMID mentioned in the text, else null>", "note": "<why>" } ]
    }
  ]
}

RULES
- Only report findings explicitly supported by the PAPER_TEXT; avoid generic background.
- Provide quotes that appear verbatim in PAPER_TEXT (case-insensitive), each up to ~20 words.
- At least one evidence quote per item MUST contain the exact mutation/segment token (e.g., "H39R", "A54E", "del42-59", "aa1396-1435").
- If no such quote exists, do not output that item.
- If a statement lists several mutations and a shared effect, create one item per mutation and reuse the shared quote.
- Keep position numeric where possible; else null. Use dengue-only entities for virus names.
- Output MUST be valid JSON (no extra commentary).
"""

def _build_messages(paper_text: str,
                    pmid: Optional[str] = None,
                    pmcid: Optional[str] = None,
                    virus_filter: Optional[str] = None,
                    protein_filter: Optional[str] = None) -> List[Dict[str, str]]:
    user_instr = {
        "pmid": pmid,
        "pmcid": pmcid,
        "virus_filter": virus_filter,
        "protein_filter": protein_filter,
    }
    user_msg = (
        "INSTRUCTIONS:\n"
        + json.dumps(user_instr, ensure_ascii=True)
        + "\n\nPAPER_TEXT:\n"
        + paper_text
    )
    return [
        {"role": "system", "content": SCHEMA_EXPLANATION.strip()},
        {"role": "user", "content": user_msg},
    ]

def _parse_json(raw: str, pmid: Optional[str], pmcid: Optional[str]) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {
        "paper": {"pmid": pmid, "pmcid": pmcid, "title": None, "virus_candidates": [], "protein_candidates": []},
        "sequence_features": []
    }

# ========= STRICT sanitizer (requires mutation token & grounded quote) ========

_ALLOWED_CATEGORIES = {
    "RNA_synthesis", "virion_assembly", "binding", "replication",
    "infectivity", "virulence", "immune_evasion", "drug_interaction",
    "temperature_sensitivity", "activity_change", "modification", "other"
}

# Expanded: accept aa-segment tokens too
_MUTATION_TOKEN_RE = re.compile(
    r"(?:"  # classic + arrows + deletions/insertions + p. + aa segments
    r"(?:[A-Z][a-z]{3}|[A-Z])\s*\d+\s*(?:[A-Z][a-z]{3}|[A-Z]|\*)"             # Ala123Thr | A123T | K70*
    r"|(?:[A-Z][a-z]{3}|[A-Z])\s*\d+\s*(?:→|->)\s*(?:[A-Z][a-z]{3}|[A-Z]|\*)" # T123→A / A123 -> T
    r"|(?:del|Δ)\s*\d+(?:\s*[-–—]\s*\d+)?"                                    # del69-70 / Δ69
    r"|aa\d+\s*[-–—]\s*\d+"                                                   # aa1396-1435
    r"|ins\s*[A-Za-z\*]+(?:\d+)?"                                             # insA / insAla / insAla123
    r"|p\.[A-Za-z]{3}\d+[A-Za-z]{3}"                                          # p.Asp125Ala
    r")",
    flags=re.IGNORECASE
)

def _normalize_for_match(s: str) -> str:
    MAP = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00A0": " ", "\u2009": " ", "\u202F": " ", "\u200A": " ", "\u200B": "",
        "\u2192": "->", "\u27F6": "->",
    }
    return re.sub(r"\s+", " ", s.translate(str.maketrans(MAP))).strip()

def _has_mutation_token(s: str) -> bool:
    return bool(_MUTATION_TOKEN_RE.search(s or ""))

def _category_or_other(cat: Optional[str]) -> str:
    cat = (cat or "").strip()
    return cat if cat in _ALLOWED_CATEGORIES else "other"

def sanitize_and_filter_strict(result: dict, full_text: str) -> dict:
    """Drop any item that lacks a strict mutation/segment token and a verbatim quote containing it."""
    if not isinstance(result, dict):
        return {"paper": {"pmid": None, "pmcid": None, "title": None,
                          "virus_candidates": [], "protein_candidates": []},
                "sequence_features": []}

    norm_text = _normalize_for_match(full_text or "")
    paper = result.get("paper") or {}
    feats = result.get("sequence_features") or []
    cleaned = []

    for f in feats:
        if not isinstance(f, dict):
            continue
        mut = (f.get("mutation") or "").strip()
        if not mut or not _has_mutation_token(mut):
            continue
        quotes = f.get("evidence_quotes") or []
        ok_quote = None
        for q in quotes:
            qs = _normalize_for_match(str(q))
            if mut.lower() in qs.lower() and qs and qs.lower() in norm_text.lower():
                ok_quote = q
                break
        if not ok_quote:
            continue
        f["effect_category"] = _category_or_other(f.get("effect_category"))
        cleaned.append(f)

    return {"paper": paper, "sequence_features": cleaned}

# ===== Chunking =====

def _chunk_text(text: str, max_chars: int = 16000, overlap: int = 500) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ===== Merge & misc helpers =====

def _merge_lists_unique(a: List[str], b: List[str], limit: int) -> List[str]:
    out = []
    seen = set()
    for item in (a or []) + (b or []):
        if item and item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= limit:
            break
    return out

def _merge_features(all_feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    merged = []
    for f in all_feats:
        key = (
            (f.get("virus") or "").strip().lower(),
            (f.get("protein") or "").strip().lower(),
            (f.get("mutation") or "").strip().lower(),
            (f.get("effect_category") or "").strip().lower(),
            (f.get("effect_summary") or "").strip().lower(),
        )
        if key not in seen:
            seen.add(key); merged.append(f)
    return merged

_CANON_PROTEIN = {
    "ns1 protein": "NS1", "ns1": "NS1",
    "ns2a protein": "NS2A", "ns2a": "NS2A",
    "ns2b protein": "NS2B", "ns2b": "NS2B",
    "ns3 protein": "NS3", "ns3": "NS3",
    "ns4a protein": "NS4A", "ns4a": "NS4A",
    "ns4b protein": "NS4B", "ns4b": "NS4B",
    "ns5 protein": "NS5", "ns5": "NS5",
    "e protein": "E", "e": "E", "envelope": "E",
    "prm protein": "prM", "prm": "prM", "premembrane": "prM", "pre-m": "prM",
    "m protein": "M", "m": "M", "membrane": "M",
    "capsid protein": "Capsid", "capsid": "Capsid", "c": "Capsid",
}

def _canon_protein(name: Optional[str]) -> Optional[str]:
    if not name: return None
    key = str(name).strip().lower()
    return _CANON_PROTEIN.get(key, name)

def _is_valid_pmid(x):
    return isinstance(x, str) and x.isdigit() and 5 <= len(x) <= 9

def _coerce_int(x):
    try: return int(x)
    except Exception: return None

def _trim_quote(q, max_words: int = 20):
    if not isinstance(q, str): return None
    words = q.split()
    return " ".join(words[:max_words])

def _quote_in_text(q: str, text: str) -> bool:
    if not q or not text: return False
    return q.lower() in text.lower()

def _contains_hint(text: str) -> bool:
    _TARGET_HINTS = {"mutation","virion","assembly","rna synthesis","replication","infectivity","binding",
                     "cleavage","protease","virulence","temperature","activity","interaction","modification"}
    tl = text.lower()
    return any(h in tl for h in _TARGET_HINTS)

def _score_feature(f: Dict[str, Any], full_text: str) -> float:
    score = 0.0
    quotes = f.get("evidence_quotes") or []
    if quotes: score += 0.5
    mut = f.get("mutation")
    if isinstance(mut, str):
        m = mut.lower()
        if any(isinstance(q, str) and m in q.lower() for q in quotes): score += 0.25
        elif isinstance(f.get("effect_summary"), str) and m in f["effect_summary"].lower(): score += 0.15
    if isinstance(f.get("position"), int): score += 0.05
    if isinstance(f.get("effect_category"), str) and f["effect_category"]: score += 0.05
    ctx = f.get("experiment_context") or {}
    if any(ctx.get(k) for k in ("system","assay","temperature")): score += 0.05
    for xr in (f.get("cross_refs") or []):
        pmid = (xr or {}).get("pmid")
        if isinstance(pmid, str) and pmid.isdigit(): score += 0.05; break
    if any(isinstance(q, str) and _contains_hint(q) for q in quotes): score += 0.05
    return min(score, 1.0)

def _merge_unique_strs(a, b):
    out, seen = [], set()
    for s in (a or []) + (b or []):
        if not isinstance(s, str): continue
        if s not in seen: seen.add(s); out.append(s)
    return out

def _merge_xrefs(a, b):
    out, seen = [], set()
    for xr in (a or []) + (b or []):
        if not isinstance(xr, dict): continue
        pmid = xr.get("pmid"); note = xr.get("note")
        key = (pmid, note)
        if key not in seen:
            seen.add(key)
            out.append({"pmid": pmid if pmid is None or isinstance(pmid, str) else str(pmid), "note": note})
    return out

def _choose_primary_summary(category: Optional[str], summaries):
    if not summaries: return None, []
    cat = (category or "").lower()
    def rank(s: str) -> int:
        sl = s.lower(); score = 0
        if cat == "rna_synthesis":
            if "rna synthesis" in sl: score += 3
            if "replication" in sl: score += 2
        elif cat == "virion_assembly":
            if "assembly" in sl: score += 3
            if "particle" in sl or "virion" in sl: score += 2
        for kw in ["abolish","abolished","impair","impaired","reduce","reduced",
                   "increase","increased","block","blocked","disable","disabled","enhance","enhanced"]:
            if kw in sl: score += 1
        if "no effect" in sl or "not affect" in sl or "did not affect" in sl: score -= 2
        return score
    ranked = sorted(summaries, key=lambda s: (-rank(s), summaries.index(s)))
    primary = ranked[0]
    supporting = [s for s in ranked[1:] if s != primary]
    return primary, supporting

def _flag_conflict(summaries):
    sl = [s.lower() for s in summaries if isinstance(s, str)]
    has_no_effect = any(("no effect" in s) or ("not affect" in s) or ("did not affect" in s) for s in sl)
    has_strong = any(k in s for s in sl for k in [
        "abolish","abolished","impair","impaired","reduce","reduced","increase","increased","block","blocked","disable","disabled"
    ])
    return bool(has_no_effect and has_strong)

def _coalesce_by_key(feats):
    bykey: Dict[Tuple[str,str,str,str], Dict[str,Any]] = {}
    for f in feats:
        key = (f.get("virus") or "", f.get("protein") or "", f.get("mutation") or "", f.get("effect_category") or "")
        existing = bykey.get(key)
        if not existing:
            bykey[key] = f; continue
        if existing.get("position") is None and f.get("position") is not None:
            existing["position"] = f.get("position")
        existing["evidence_quotes"] = _merge_unique_strs(existing.get("evidence_quotes"), f.get("evidence_quotes"))
        existing["cross_refs"] = _merge_xrefs(existing.get("cross_refs"), f.get("cross_refs"))
        ectx = existing.get("experiment_context") or {}; fctx = f.get("experiment_context") or {}
        for k in ("system","assay","temperature"):
            if not ectx.get(k) and fctx.get(k): ectx[k] = fctx.get(k)
        existing["experiment_context"] = ectx
        es = existing.get("effect_summary"); fs = f.get("effect_summary")
        summaries = []
        if es: summaries.append(es)
        if fs and fs not in summaries: summaries.append(fs)
        summaries += existing.get("supporting_summaries", [])
        primary, supporting = _choose_primary_summary(existing.get("effect_category"), [s for s in summaries if s])
        existing["effect_summary"] = primary
        existing["supporting_summaries"] = supporting
        if _flag_conflict([primary] + supporting):
            notes = existing.setdefault("notes", {}); notes["potential_conflict"] = True
        try:
            existing["confidence"] = max(existing.get("confidence", 0) or 0, f.get("confidence", 0) or 0)
        except Exception:
            pass
    return list(bykey.values())

# ===== Regex-based deterministic miner (new) =====

_PROTEIN_NEAR_RE = re.compile(r"\b(NS1|NS2A|NS2B|NS3|NS4A|NS4B|NS5|prM|Capsid|C|E|envelope|membrane|M)\b", re.IGNORECASE)
_VIRUS_NEAR_RE = re.compile(r"\b(DENV(?:-\d)?|dengue virus|dengue)\b", re.IGNORECASE)
_STRAIN_RE = re.compile(r"\bDENV-(1|2|3|4)\b", re.IGNORECASE)

def _window(text: str, i: int, left: int = 220, right: int = 220) -> str:
    start = max(0, i - left)
    end = min(len(text), i + right)
    # expand to sentence boundaries
    sent_start = max(text.rfind(".", 0, start), text.rfind(";", 0, start), text.rfind("\n", 0, start)) + 1
    sent_end = end
    stop = min([p for p in [text.find(".", i), text.find(";", i), text.find("\n", i)] if p != -1] + [end])
    if stop > 0:
        sent_end = stop + 1
    return _normalize_for_match(text[sent_start:sent_end]).strip()

def _nearest_protein(text: str, i: int) -> Optional[str]:
    span = text[max(0, i - 300): i + 50]
    m = None
    for m in _PROTEIN_NEAR_RE.finditer(span):
        pass
    if not m:
        return None
    name = m.group(1).lower()
    return _canon_protein({"envelope":"E","c":"Capsid"}.get(name, name))

def _infer_effect_category(sentence: str) -> Tuple[str, str]:
    s = sentence.lower()
    cat = "other"
    if any(k in s for k in ["assembly","assemble","particle","virion"]):
        cat = "virion_assembly"
    if any(k in s for k in ["replication","replicate"]):
        cat = "replication"
    if "rna synthesis" in s or "polymerase" in s or "rd rp" in s or "rdrp" in s:
        cat = "RNA_synthesis"
    if any(k in s for k in ["infectivity","titre","titer","fusion capacity","fusion threshold","fusion-defective"]):
        cat = "infectivity"
    if any(k in s for k in ["attenuat", "virulen"]):
        cat = "virulence"
    if any(k in s for k in ["temperature-sensitive","temperature sensitive"]):
        cat = "temperature_sensitivity"
    if "binding" in s:
        cat = "binding"
    # Use concise effect_summary = sentence (trimmed)
    return cat, sentence

def _token_position(token: str) -> Optional[int]:
    nums = re.findall(r"\d+", token)
    try:
        return int(nums[0]) if nums else None
    except Exception:
        return None

def _regex_mine_features(aug_text: str) -> List[Dict[str, Any]]:
    feats: List[Dict[str, Any]] = []
    for m in _MUTATION_TOKEN_RE.finditer(aug_text):
        token = m.group(0)
        i = m.start()
        sentence = _window(aug_text, i)
        protein = _nearest_protein(aug_text, i)
        virus = None
        strain = None
        vnear = _VIRUS_NEAR_RE.search(aug_text[max(0, i-150): i+150])
        if vnear:
            vv = vnear.group(1).lower()
            virus = "Dengue virus"
            st = _STRAIN_RE.search(vnear.group(0))
            if st:
                strain = f"DENV-{st.group(1)}"
            elif "denv-1" in vv: strain = "DENV-1"
            elif "denv-2" in vv: strain = "DENV-2"
            elif "denv-3" in vv: strain = "DENV-3"
            elif "denv-4" in vv: strain = "DENV-4"
        cat, eff = _infer_effect_category(sentence)
        quote = _trim_quote(sentence, 20)
        pos = _token_position(token)
        feat = {
            "virus": virus or "Dengue virus",
            "source_strain": strain,
            "protein": protein,
            "mutation": token.strip(),
            "position": pos,
            "effect_category": cat,
            "effect_summary": eff,
            "mechanism_hypothesis": None,
            "experiment_context": {"system": None, "assay": None, "temperature": None},
            "evidence_quotes": [quote] if quote else [token],
            "cross_refs": [],
            "confidence": 0.8  # heuristic baseline; will be filtered by sanitizer if quotes don't match
        }
        feats.append(feat)
    return feats

# ===== Core IE =====

def extract_sequence_features(
    paper_text: str,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    virus_filter: Optional[str] = None,
    protein_filter: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    # Augment the text with canonical tokens so the strict grounding gate can succeed
    augmented = _augment_text_with_canonical_tokens(paper_text)
    messages = _build_messages(augmented, pmid, pmcid, virus_filter, protein_filter)
    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    parsed = _parse_json(raw, pmid, pmcid)
    return sanitize_and_filter_strict(parsed, augmented)

# --- Back-compat wrapper for older pipeline code expecting `clean_and_ground` ---
def clean_and_ground(
    result: dict,
    full_text: str,
    restrict_to_paper: bool = True,
    *,
    require_mutation_in_quote: bool = True,
    min_confidence: float = 0.0,
    propagate_class_statements: bool = False,
    dengue_only: bool = True,
) -> dict:
    """
    Back-compat: apply strict sanitization using the same augmented text that the model saw.
    Drops items without a recognizable token or a verbatim grounded quote containing that token.
    Respects a minimal confidence cutoff if present in items (safe no-op if absent).
    """
    augmented = _augment_text_with_canonical_tokens(full_text or "")
    safe_result = result if isinstance(result, dict) else {"paper": {}, "sequence_features": []}
    cleaned = sanitize_and_filter_strict(safe_result, augmented)
    try:
        thr = float(min_confidence or 0.0)
    except Exception:
        thr = 0.0
    feats = []
    for f in cleaned.get("sequence_features", []):
        conf = f.get("confidence", None)
        if conf is None or (isinstance(conf, (int, float)) and conf >= thr):
            feats.append(f)
    cleaned["sequence_features"] = feats
    return cleaned

# ===== Public (runs LLM + regex miner, merges, sanitizes) =====

def run_on_paper(full_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    pmid = meta.get("pmid"); pmcid = meta.get("pmcid")
    virus_filter = meta.get("virus_filter"); protein_filter = meta.get("protein_filter")
    model = meta.get("model", DEFAULT_MODEL); temperature = meta.get("temperature", 0.0)
    max_tokens = meta.get("max_tokens", 2048)
    exhaustive = bool(meta.get("exhaustive", True))
    chunk_chars = int(meta.get("chunk_chars", 16000)); overlap_chars = int(meta.get("overlap_chars", 500))
    delay_ms = int(meta.get("delay_ms", 400))

    # Use one augmented copy across both LLM and regex miner
    augmented_full = _augment_text_with_canonical_tokens(full_text or "")

    def _pass(text: str, vflt: Optional[str], pflt: Optional[str]):
        chunks = _chunk_text(text, max_chars=chunk_chars, overlap=overlap_chars)
        feats: List[Dict[str, Any]] = []; v_cands: List[str] = []; p_cands: List[str] = []
        for idx, ch in enumerate(chunks, start=1):
            tagged = f"[PART {idx}/{len(chunks)}]\n" + ch
            res = extract_sequence_features(tagged, pmid=pmid, pmcid=pmcid, virus_filter=vflt, protein_filter=pflt,
                                            model=model, temperature=temperature, max_tokens=max_tokens)
            paper = res.get("paper", {}) if isinstance(res, dict) else {}
            v_cands = _merge_lists_unique(v_cands, paper.get("virus_candidates", []), limit=8)
            p_cands = _merge_lists_unique(p_cands, paper.get("protein_candidates", []), limit=12)
            feats.extend(res.get("sequence_features", []) if isinstance(res, dict) else [])
            if delay_ms > 0: time.sleep(delay_ms / 1000.0)
        return v_cands, p_cands, feats

    # LLM pass(es) over augmented text
    virus_cands, protein_cands, all_feats = _pass(augmented_full, virus_filter, protein_filter)
    if exhaustive:
        v2, p2, f2 = _pass(augmented_full, None, None)
        virus_cands = _merge_lists_unique(virus_cands, v2, limit=8)
        protein_cands = _merge_lists_unique(protein_cands, p2, limit=12)
        all_feats.extend(f2)

    # Regex miner over augmented text (deterministic)
    mined_feats = _regex_mine_features(augmented_full)
    all_feats.extend(mined_feats)

    merged_feats = _merge_features(all_feats)
    return {
        "paper": {"pmid": pmid, "pmcid": pmcid, "title": None,
                  "virus_candidates": virus_cands, "protein_candidates": protein_cands},
        "sequence_features": merged_feats
    }

__all__ = [
    "run_on_paper",
    "clean_and_ground",
    "extract_sequence_features",
    "sanitize_and_filter_strict",
]

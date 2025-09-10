
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
        if resp.status_code == 429:
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

# ===== Generalized Prompt =====

SCHEMA_EXPLANATION = """
You are an information extraction model for biological literature. Your goal is to extract
sequence- or variant-level functional findings from ANY organism (viruses, bacteria, eukaryotes),
focusing on proteins/genes and the impact of specific mutations or modifications.

SCOPE & SIGNALS (soft hints; do not hallucinate)
- Prefer spans that mention a protein/gene-like entity ("protein", "gene", "ORF", "polyprotein", "NS", "E", "spike", etc.).
- If optional filters (organism or protein keywords) are provided, prioritize spans that match them.
- Target concepts (examples, not exhaustive): active site, interaction, disease association, mutation, virulence,
  temperature sensitivity, activity change, virion/particle assembly, viral replication/synthesis, binding,
  post-translational modifications (phosphorylation/glycosylation/etc.), drug interaction/resistance,
  effect on transmission, infectivity, cleavage/protease recognition, immune evasion, host factor dependence.

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
      "mutation": "<e.g., N501Y, D125A, del69-70, p.Asp125Ala, K70*, etc.>",
      "position": "<integer or null>",
      "effect_category": "<one of: RNA_synthesis | virion_assembly | binding | replication | infectivity | virulence | immune_evasion | drug_interaction | temperature_sensitivity | activity_change | modification | other>",
      "effect_summary": "<1-2 sentence, as concrete as the paper allows>",
      "mechanism_hypothesis": "<brief mechanistic explanation if present, else null>",
      "experiment_context": {
        "system": "<cell/animal/in vitro/in silico, if stated>",
        "assay": "<method/assay type if stated>",
        "temperature": "<value if stated, else null>"
      },
      "evidence_quotes": ["<short quote 1>", "<short quote 2>"],
      "cross_refs": [
        { "pmid": "<numeric PMID mentioned in the paper text, else null>", "note": "<why referenced>" }
      ]
    }
  ]
}

RULES
- Only report findings explicitly supported by the PAPER_TEXT; avoid generic background not shown in the text.
- Provide quotes that appear verbatim in the PAPER_TEXT (case-insensitive), each up to ~20 words.
- If the paper groups multiple mutations in a single claim (e.g., “X, Y, Z impaired assembly”), create separate entries per mutation
  and include the group claim as a quote for each relevant entry.
- Prefer precise language about direction and magnitude (e.g., “abolished”, “reduced”, “increased”, “no effect”). Avoid vague wording.
- Keep position numeric where possible; else null.
- cross_refs.pmid must be strictly numeric; if not numeric or absent, set pmid to null or omit.
- The output MUST be valid JSON using standard ASCII quotes, with no extra commentary.
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
    messages = _build_messages(paper_text, pmid, pmcid, virus_filter, protein_filter)
    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    return _parse_json(raw, pmid, pmcid)

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

# ===== Canonicalization & scoring =====

_CANON_PROTEIN = {
    "ns1 protein": "NS1", "ns1": "NS1",
    "ns2a protein": "NS2A", "ns2a": "NS2A",
    "ns2b protein": "NS2B", "ns2b": "NS2B",
    "ns3 protein": "NS3", "ns3": "NS3",
    "ns4a protein": "NS4A", "ns4a": "NS4A",
    "ns4b protein": "NS4B", "ns4b": "NS4B",
    "e protein": "E", "e": "E",
    "prm protein": "prM", "prm": "prM",
    "capsid protein": "Capsid", "capsid": "Capsid",
}

def _canon_protein(name: Optional[str]) -> Optional[str]:
    if not name: return None
    key = str(name).strip().lower()
    return _CANON_PROTEIN.get(key, name)

def _canon_virus_and_strain(virus: Optional[str], source_strain: Optional[str]):
    v = virus or None
    s = source_strain or None
    return v, s

_TARGET_HINTS = {"mutation","virion","assembly","rna synthesis","replication","infectivity","binding",
                 "cleavage","protease","virulence","temperature","activity","interaction","modification"}

def _contains_hint(text: str) -> bool:
    tl = text.lower()
    return any(h in tl for h in _TARGET_HINTS)

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

# ===== Merge helpers & conflict handling =====

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
        summaries = []; 
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

# ===== Class-statement propagation (generalized) & strain inference =====

_MUTATION_REGEX = re.compile(
    r"(?:\b[A-Z]\d+[A-Z]\b|\b[A-Z]\d+\*\b|\bdel\d+(?:-\d+)?\b|\bΔ\d+(?:-\d+)?\b|\bins\d+\b|\bdup\d+\b|\bp\.[A-Za-z]{3}\d+[A-Za-z]{3}\b)"
)

def _extract_class_statements(text: str):
    out = []
    t = re.sub(r"\s+", " ", text)
    patterns = [
        r"mutations?\s*(?:in\s+[A-Za-z0-9\-\_/]+)?\s*\(([^)]+)\)\s*([^\.]{0,240}?\.)",
        r"\(([^)]+)\)\s*([^\.]{0,240}?\.)\s*(?:mutations?|substitutions?)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            muts_blob = m.group(1)
            clause = m.group(2).strip()
            muts = _MUTATION_REGEX.findall(muts_blob)
            if muts and clause:
                quote = (m.group(0)[:180]).strip()
                out.append((muts, clause, quote))
    return out

def _infer_source_strain(text: str) -> Optional[str]:
    # Generic hook; can be extended per-domain. Keep conservative.
    t = text.lower()
    if "strain" in t or "serotype" in t or "lineage" in t:
        # Leave null unless explicitly found elsewhere; avoid hallucination.
        return None
    return None

def _apply_class_statements(full_text: str, feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    class_items = _extract_class_statements(full_text)
    if not class_items:
        return feats
    by_mut: Dict[str, List[int]] = {}
    for i, f in enumerate(feats):
        mut = (f.get("mutation") or "").upper()
        if mut: by_mut.setdefault(mut, []).append(i)
    for muts, clause, quote in class_items:
        cl = clause.lower()
        for mtoken in muts:
            for idx in by_mut.get(mtoken.upper(), []):
                f = feats[idx]
                eq = f.get("evidence_quotes") or []
                if quote not in eq:
                    eq.append(quote); f["evidence_quotes"] = eq
                es = f.get("effect_summary") or ""
                if "assembly" in cl and "rna" in cl and "without" in cl and f.get("effect_category") == "virion_assembly":
                    if "without" not in es.lower():
                        f["effect_summary"] = es.rstrip(".") + " without significantly affecting RNA synthesis."
                if ("rna synthesis" in cl or "replication" in cl) and ("abolish" in cl or "abolished" in cl) and f.get("effect_category") == "RNA_synthesis":
                    if "abolish" not in es.lower():
                        f["effect_summary"] = "Selectively abolished RNA synthesis."
    return feats

# ===== Sanitizer & Grounding =====

def clean_and_ground(
    result: Dict[str, Any],
    full_text: str,
    restrict_to_paper: bool = True,
    *,
    require_mutation_in_quote: bool = False,
    min_confidence: float = 0.0,
    propagate_class_statements: bool = True,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"paper": {"title": None, "pmid": None, "pmcid": None, "virus_candidates": [], "protein_candidates": []},
                "sequence_features": []}

    paper = result.get("paper", {}) or {}
    feats = result.get("sequence_features", []) or []

    # Normalize paper fields
    paper["title"] = paper.get("title") or None
    paper["pmid"] = str(paper.get("pmid")) if paper.get("pmid") else None
    paper["pmcid"] = str(paper.get("pmcid")) if paper.get("pmcid") else None
    paper["virus_candidates"] = [v for v in (paper.get("virus_candidates") or []) if isinstance(v, str)]
    pc_raw = [p for p in (paper.get("protein_candidates") or []) if isinstance(p, str)]
    # Canonicalize proteins a bit
    pc = []; pseen = set()
    for pstr in pc_raw:
        cp = _canon_protein(pstr) or pstr
        if cp not in pseen:
            pseen.add(cp); pc.append(cp)
    paper["protein_candidates"] = pc

    if propagate_class_statements and feats:
        feats = _apply_class_statements(full_text, feats)

    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for f in feats:
        if not isinstance(f, dict): continue
        virus = f.get("virus") or None
        protein = _canon_protein(f.get("protein") or None)
        mutation = f.get("mutation") or None
        pos = _coerce_int(f.get("position"))
        cat = f.get("effect_category") or None
        summary = f.get("effect_summary") or None
        mech = f.get("mechanism_hypothesis") or None
        ctx = f.get("experiment_context") or {}
        quotes = [q for q in (f.get("evidence_quotes") or []) if isinstance(q, str)]
        quotes = [_trim_quote(q) for q in quotes if q]; quotes = [q for q in quotes if q]

        grounded_quotes = [q for q in quotes if _quote_in_text(q, full_text)]
        if restrict_to_paper and not grounded_quotes: continue

        if require_mutation_in_quote and isinstance(mutation, str):
            if not any(mutation.lower() in q.lower() for q in grounded_quotes): continue

        xrefs_in = f.get("cross_refs") or []
        xrefs_out = []
        for xr in xrefs_in:
            if not isinstance(xr, dict): continue
            pmid = xr.get("pmid"); note = xr.get("note")
            if pmid is None or _is_valid_pmid(str(pmid)):
                xrefs_out.append({"pmid": (str(pmid) if pmid is not None else None), "note": note or None})

        virus, source_strain = _canon_virus_and_strain(virus, f.get("source_strain"))
        if not source_strain:
            inferred = _infer_source_strain(full_text)
            if inferred: source_strain = inferred

        feat = {
            "virus": virus,
            "source_strain": source_strain or None,
            "protein": protein,
            "mutation": mutation,
            "position": pos,
            "effect_category": cat,
            "effect_summary": summary,
            "mechanism_hypothesis": mech,
            "experiment_context": {
                "system": (ctx.get("system") or None),
                "assay": (ctx.get("assay") or None),
                "temperature": (ctx.get("temperature") or None),
            },
            "evidence_quotes": grounded_quotes,
            "cross_refs": xrefs_out,
        }

        conf = _score_feature(feat, full_text)
        if conf < float(min_confidence): continue
        feat["confidence"] = round(conf, 3)

        key = (
            str(feat["virus"]).lower() if feat["virus"] else "",
            str(feat["protein"]).lower() if feat["protein"] else "",
            str(feat["mutation"]).lower() if feat["mutation"] else "",
            str(feat["effect_category"]).lower() if feat["effect_category"] else "",
            (feat["effect_summary"] or "").strip().lower(),
        )
        if key in seen: continue
        seen.add(key)
        cleaned.append(feat)

    cleaned = _coalesce_by_key(cleaned)
    return {"paper": paper, "sequence_features": cleaned}

# ===== Public =====

def run_on_paper(full_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    pmid = meta.get("pmid"); pmcid = meta.get("pmcid")
    virus_filter = meta.get("virus_filter"); protein_filter = meta.get("protein_filter")
    model = meta.get("model", DEFAULT_MODEL); temperature = meta.get("temperature", 0.0)
    max_tokens = meta.get("max_tokens", 2048)
    exhaustive = bool(meta.get("exhaustive", True))
    chunk_chars = int(meta.get("chunk_chars", 16000)); overlap_chars = int(meta.get("overlap_chars", 500))
    delay_ms = int(meta.get("delay_ms", 400))

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

    virus_cands, protein_cands, all_feats = _pass(full_text, virus_filter, protein_filter)
    if exhaustive:
        v2, p2, f2 = _pass(full_text, None, None)
        virus_cands = _merge_lists_unique(virus_cands, v2, limit=8)
        protein_cands = _merge_lists_unique(protein_cands, p2, limit=12)
        all_feats.extend(f2)

    merged_feats = _merge_features(all_feats)
    return {"paper": {"pmid": pmid, "pmcid": pmcid, "title": None, "virus_candidates": virus_cands,
                      "protein_candidates": protein_cands},
            "sequence_features": merged_feats}

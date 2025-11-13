# llm/utils.py - Shared utilities for all LLM backends
# All text processing, parsing, and extraction utilities extracted from gemini.py
# This ensures ALL LLMs use the exact same logic

from __future__ import annotations

import os
import re
import json
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set

from llm.prompts import PROMPTS


# ========== Constants & Compiled Regexes (OPTIMIZATION) ==========
ALLOWED_CATEGORIES = {
    "RNA_synthesis", "virion_assembly", "binding", "replication",
    "infectivity", "virulence", "immune_evasion", "drug_interaction",
    "temperature_sensitivity", "activity_change", "modification", "other"
}

ALLOWED_CONTINUITY = {"continuous", "discontinuous", "point", "unknown"}

# Compile once for performance
STANDARD_MUT_RE = re.compile(r"^[A-Z]\d+[A-Z*]$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
MUT_TOKEN_RE = re.compile(r"\b(?:[A-Z][0-9]{1,5}[A-Z*]|\d{1,5}[A-Z])\b")
HGVS_RE = re.compile(r"\bp\.[A-Z][a-z]{2}\d{1,5}[A-Z][a-z]{2}\b")
PROTEIN_RE = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9\-\/]{1,30})\s+(protein|glycoprotein|polyprotein|capsid|envelope|domain|subunit|chain|peptide|complex)\b",
    re.IGNORECASE
)
RESIDUE_RE = re.compile(r"\b(?:[A-Z][a-z]{2}|[A-Z])\s*(\d{1,5})\b")
RANGE_RE = re.compile(r"\b(\d{1,5})\s*[-–—]\s*(\d{1,5})\s*(?:aa|amino acids?|residues?)\b", re.IGNORECASE)
RANGE_WORD_RE = re.compile(r"\bresidues?\s+(\d{1,5})\s*(?:to|through|–|-)\s*(\d{1,5})\b", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(\d{1,5})\s*(?:aa|amino acids?)\b", re.IGNORECASE)
MOTIF_RE = re.compile(r"\b[A-Z][A-ZxX\-]{2,9}\b")
AA_SINGLE_RE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])[-/]?(?:residue|position)?[-/]?(?P<pos>\d{1,5})\b", re.IGNORECASE)

# Amino acid lookup table (cached)
AA_NAME_TO_CODES = {
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

AA_NAME_PATTERN = "|".join(
    sorted((re.escape(k) for k in AA_NAME_TO_CODES.keys()), key=len, reverse=True)
)

SPELLED_MUT_RE = re.compile(
    rf"(?P<from>{AA_NAME_PATTERN})\s*(?:residue\s*)?(?P<pos>\d{{1,5}})"
    r"(?:\s*(?:to|into|->|→|for|with|by|in|changed\s+to|replaced\s+by|replaced\s+with|substituted\s+with|substituted\s+by))"
    r"\s*(?:a\s+|an\s+)?(?P<to>{AA_NAME_PATTERN})(?:\s+residue|\s+residues)?",
    re.IGNORECASE
)

AA_WORD_POS_RE = re.compile(
    rf"\b(?P<name>{AA_NAME_PATTERN})(?:[\s\-]*(?:residue|position|site))?\s*(?P<pos>\d{{1,5}})\b",
    re.IGNORECASE
)


# ========== Helper functions ==========
def normalize_ws(text: str) -> str:
    """Normalize whitespace in text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def normalize_for_match(text: str) -> str:
    """Downcase, strip spacing/punctuation, fold accents."""
    if not isinstance(text, str) or not text:
        return ""
    try:
        nfkd = unicodedata.normalize("NFKD", text)
    except Exception:
        nfkd = text
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii", "ignore")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def expand_to_sentence(full_text: str, fragment: str) -> Optional[str]:
    """Locate the sentence containing the fragment."""
    if not isinstance(full_text, str) or not isinstance(fragment, str):
        return None
    frag_norm = normalize_ws(fragment).lower()
    if not frag_norm:
        return None

    corpus = full_text.replace("\n", " ").replace("\r", " ")
    sentences = SENTENCE_SPLIT_RE.split(corpus)
    for sentence in sentences:
        sent_norm = normalize_ws(sentence).lower()
        if frag_norm in sent_norm:
            return sentence.strip()
    return None


def lookup_aa_codes(name: str) -> Optional[Tuple[str, str]]:
    """Lookup amino acid codes from name."""
    return AA_NAME_TO_CODES.get(name.lower())


def extract_spelled_mutation(text: str) -> Optional[Tuple[str, str]]:
    """Extract spelled mutation (e.g., 'alanine 226 to valine') and return (HGVS, short)."""
    if not isinstance(text, str):
        return None
    normalized = text.replace("–", "-").replace("—", "-").replace("→", "->")
    for match in SPELLED_MUT_RE.finditer(normalized):
        pos = match.group("pos")
        aa_from = lookup_aa_codes(match.group("from"))
        aa_to = lookup_aa_codes(match.group("to"))
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


def scan_text_candidates(text: str) -> Dict[str, List[str]]:
    """
    Pre-scan for candidate mentions to guide LLM.
    
    DISABLED: Returns empty dict to let LLM extract without regex hints.
    Uncomment the code below to re-enable regex-based candidate scanning.
    """
    # DISABLED - Returning empty dict so LLM works without regex hints
    return {}
    
    # ========== COMMENTED OUT - REGEX-BASED HINT GENERATION ==========
    # if not isinstance(text, str):
    #     return {}
    # 
    # candidates: Dict[str, Set[str]] = {
    #     "mutation_tokens": set(),
    #     "hgvs_tokens": set(),
    #     "protein_terms": set(),
    #     "residue_numbers": set(),
    #     "motif_candidates": set(),
    #     "amino_acid_mentions": set(),
    # }
    # 
    # for m in MUT_TOKEN_RE.finditer(text):
    #     candidates["mutation_tokens"].add(m.group(0))
    # 
    # for m in HGVS_RE.finditer(text):
    #     candidates["hgvs_tokens"].add(m.group(0))
    # 
    # for m in PROTEIN_RE.finditer(text):
    #     term = f"{m.group(1)} {m.group(2)}"
    #     candidates["protein_terms"].add(term)
    # 
    # for m in RESIDUE_RE.finditer(text):
    #     candidates["residue_numbers"].add(m.group(1))
    # 
    # for m in RANGE_RE.finditer(text):
    #     candidates["residue_numbers"].add(f"{m.group(1)}-{m.group(2)}")
    # 
    # for m in RANGE_WORD_RE.finditer(text):
    #     candidates["residue_numbers"].add(f"{m.group(1)}-{m.group(2)}")
    # 
    # for m in COUNT_RE.finditer(text):
    #     candidates["residue_numbers"].add(m.group(1))
    # 
    # for m in MOTIF_RE.finditer(text):
    #     token = m.group(0)
    #     if len(token) >= 3:
    #         candidates["motif_candidates"].add(token)
    # 
    # for m in AA_SINGLE_RE.finditer(text):
    #     aa = m.group(1).upper()
    #     pos = m.group("pos")
    #     candidates["amino_acid_mentions"].add(f"{aa}{pos}")
    # 
    # for m in AA_WORD_POS_RE.finditer(text):
    #     name = m.group("name")
    #     pos = m.group("pos")
    #     codes = lookup_aa_codes(name)
    #     if codes:
    #         one_letter, three_letter = codes
    #         candidates["amino_acid_mentions"].add(f"{three_letter}{pos}")
    #         candidates["amino_acid_mentions"].add(f"{one_letter}{pos}")
    # 
    # def _trim(seq: Set[str], limit: int = 200) -> List[str]:
    #     return sorted(seq)[:limit] if seq else []
    # 
    # return {
    #     "mutation_tokens": _trim(candidates["mutation_tokens"]),
    #     "hgvs_tokens": _trim(candidates["hgvs_tokens"]),
    #     "protein_terms": _trim(candidates["protein_terms"]),
    #     "residue_numbers": _trim(candidates["residue_numbers"]),
    #     "motif_candidates": _trim(candidates["motif_candidates"]),
    #     "amino_acid_mentions": _trim(candidates["amino_acid_mentions"]),
    # }


def token_context_windows(full_text: str,
                           token: str,
                           *,
                           left: int = 900,
                           right: int = 900,
                           max_windows: int = 4,
                           max_total_chars: int = 12000) -> str:
    """Extract context windows around a token."""
    corpus = normalize_ws(full_text)
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


def chunk_text(text: str, max_chars: int = 12000, overlap: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
    t = normalize_ws(text)
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


def safe_json_value(raw: str):
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


def normalize_prompt_feature(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    if continuity not in ALLOWED_CONTINUITY:
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


def collect_extracted_tokens(features: List[Dict[str, Any]]) -> Set[str]:
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


def pass2_prompt(full_text: str,
                  target_token: str,
                  pmid: Optional[str],
                  pmcid: Optional[str],
                  token_type: str,
                  scan_candidates: Optional[Dict[str, List[str]]] = None) -> str:
    """Build prompt using the analyst template."""
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

    # DISABLED: Regex-based hints section - LLM will extract without prior hints
    # Uncomment below to re-enable KNOWN_MENTIONS hints in the prompt
    # if scan_candidates:
    #     hints = {k: v for k, v in scan_candidates.items() if v}
    #     if hints:
    #         hints_json = json.dumps(hints, ensure_ascii=False)
    #         body += ("\n\nKNOWN_MENTIONS (use internally to ensure coverage; do not list separately):\n"
    #                  f"{hints_json}")

    if wrapper:
        return f"{wrapper}\n\n{body}"
    return body


def convert_bio_schema_feature(f: Dict[str, Any]) -> Dict[str, Any]:
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
            spelled = extract_spelled_mutation(first_variant)
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
            spelled = extract_spelled_mutation(text)
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
    """
    Clean and validate extracted features with EARLY confidence filtering.
    
    CRITICAL FIX: Calculate confidence BEFORE filtering to ensure min_confidence works.
    """
    if require_target_in_quote is None:
        require_target_in_quote = require_mutation_in_quote
    
    paper = (raw or {}).get("paper") or {
        "pmid": None, "pmcid": None, "title": None, 
        "virus_candidates": [], "protein_candidates": []
    }
    
    feats = (raw or {}).get("sequence_features") or []
    
    # Convert bio schema if needed
    if feats and isinstance(feats, list) and isinstance(feats[0], dict) and 'feature' in feats[0]:
        feats = [convert_bio_schema_feature(x) for x in feats]
    
    kept = []
    norm_text = normalize_ws(full_text).lower() if isinstance(full_text, str) else ""
    norm_text_ascii = normalize_for_match(full_text) if isinstance(full_text, str) else ""
    
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
            expanded = expand_to_sentence(full_text, q)
            expanded_quotes.append(expanded if expanded else q.strip())

        quote_ok = False
        if expanded_quotes and isinstance(full_text, str):
            for usable in expanded_quotes:
                ql = normalize_ws(usable).lower()
                if ql in norm_text:
                    quote_ok = True
                    break
                ql_ascii = normalize_for_match(usable)
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
        if cat and cat not in ALLOWED_CATEGORIES:
            f["effect_category"] = "other"

        # ===== CONFIDENCE SCORING (IMPROVED & EARLIER) =====
        conf = 0.0
        
        # Determine feature type for adaptive scoring
        target_type = f.get("target_type") or ""
        effect_cat = (f.get("effect_category") or "").lower()
        
        # 1. Evidence quality (40% weight)
        if quotes:
            conf += 0.4
            # Bonus for multiple independent quotes
            if len(quotes) > 1:
                conf += 0.05
        
        # 2. Position specificity (15% weight)
        # For structural regions, check residue_positions instead of just position
        pos = f.get("position")
        has_position = False
        try:
            if isinstance(pos, int) or (isinstance(pos, str) and pos.strip().isdigit()):
                has_position = True
                conf += 0.15
        except Exception:
            pass
        
        # Alternative: check for residue_positions (domain ranges)
        if not has_position:
            residue_pos = f.get("residue_positions") or []
            if residue_pos and isinstance(residue_pos, list):
                for rp in residue_pos:
                    if isinstance(rp, dict) and rp.get("start") is not None:
                        conf += 0.15
                        break
        
        # 3. Mutation format quality OR structural annotation (15% weight)
        mutation = (f.get("mutation") or "").strip()
        if mutation:
            # Standard format like A226V
            if STANDARD_MUT_RE.match(mutation):
                conf += 0.15
            # Partial credit for other formats
            elif re.search(r'\d+', mutation):
                conf += 0.08
        else:
            # No mutation but has structural annotation (domain, region, etc.)
            if target_type in ("protein", "region", "domain") or effect_cat == "structural":
                # Give credit for valid structural feature
                conf += 0.12
        
        # 4. Protein context (10% weight)
        if f.get("protein"):
            conf += 0.10
        
        # 5. Virus context (5% weight)
        if f.get("virus"):
            conf += 0.05
        
        # 6. Experimental context (10% weight)
        ctx = f.get("experiment_context") or {}
        if any(ctx.get(k) for k in ("system", "assay", "temperature")):
            conf += 0.10
        
        # 7. Effect description (5% weight)
        if f.get("effect_category") and f.get("effect_category") != "unknown":
            conf += 0.05
        
        # Cap at 1.0
        f["confidence"] = min(conf, 1.0)

        # ===== APPLY MIN_CONFIDENCE FILTER HERE (CRITICAL FIX) =====
        if f["confidence"] < float(min_confidence or 0.0):
            continue

        # Fix mutation field if needed
        if not f.get("mutation") and f.get("target_token"):
            if f.get("target_type") == "mutation":
                f["mutation"] = f["target_token"]
        
        kept.append(f)
    
    return {"paper": paper, "sequence_features": kept}


__all__ = [
    # Constants
    "ALLOWED_CATEGORIES",
    "ALLOWED_CONTINUITY",
    "STANDARD_MUT_RE",
    # Functions
    "normalize_ws",
    "normalize_for_match",
    "expand_to_sentence",
    "lookup_aa_codes",
    "extract_spelled_mutation",
    "scan_text_candidates",
    "token_context_windows",
    "chunk_text",
    "safe_json_value",
    "normalize_prompt_feature",
    "collect_extracted_tokens",
    "pass2_prompt",
    "convert_bio_schema_feature",
    "clean_and_ground",
]


# pipeline/batch_analyze.py
from __future__ import annotations

from typing import Dict, List, Optional
import time
import pandas as pd

from extractor import get_pmc_fulltext_with_meta  # your existing fetcher
#from llm_groq import run_on_paper, clean_and_ground  # your existing LLM + cleaner
from llm_gemini import run_on_paper, clean_and_ground 
import os
import time


def fetch_all_fulltexts(pmids: List[str],
                        delay_ms: int = 200,
                        retries: int = 3,
                        backoff: float = 0.8) -> Dict[str, Dict]:
    """
    Fetch PMC full text for each PMID with a few retries to handle transient hiccups.
    Returns: dict[pmid] = { pmcid, title, text, status, error }
      - status in {"ok", "no_pmc_fulltext", "error"}
    Special-cases 403 (Forbidden) to wait longer and emit a clearer message.
    """
    out: Dict[str, Dict] = {}
    for pmid in pmids:
        entry = {"pmid": pmid, "pmcid": None, "title": None, "text": "", "status": "error", "error": None}
        attempt = 0
        while attempt < retries:
            try:
                pmcid, text, title = get_pmc_fulltext_with_meta(pmid)
                entry["pmcid"] = pmcid
                entry["title"] = title
                if not pmcid or not text:
                    entry["status"] = "no_pmc_fulltext"
                else:
                    entry["text"] = text
                    entry["status"] = "ok"
                break  # success
            except Exception as e:
                msg = str(e)
                entry["error"] = msg
                attempt += 1

                # If we hit a 403, pause longer before retrying (server-side throttle)
                if "403" in msg or "Forbidden" in msg:
                    # polite wait to cool off
                    time.sleep(4.0 * attempt)
                else:
                    # exponential-ish backoff for other transient errors
                    time.sleep((backoff ** attempt))

                if attempt >= retries:
                    # final failure recorded as "error"
                    entry["status"] = "error"
                    break

        out[pmid] = entry
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
    return out


def analyze_texts(papers: dict,
        *,
        virus_filter: str = "",
        protein_filter: str = "",
        exhaustive: bool = True,
        chunk_chars: int = 12000,
        overlap_chars: int = 500,
        delay_ms: int = 0,
        min_confidence: float = 0.6,
        require_mut_quote: bool = True,
        llm_meta: dict | None = None,
        paper_pause_sec: float | None = None,  # <-- NEW: gentle pacing between papers
        ) -> Dict[str, Dict]:
    """
    Run the same LLM prompt (run_on_paper) on each 'ok' paper, then clean+ground.
    Returns dict[pmid] = { status, pmcid, title, result? }
    """

    # Default from env if not provided; sane free-tier friendly default
    if paper_pause_sec is None:
        try:
            paper_pause_sec = float(os.getenv("PAPER_PAUSE_SEC", "2.0"))
        except Exception:
            paper_pause_sec = 2.0

    results: Dict[str, Dict] = {}
    for pmid, info in papers.items():
        if info.get("status") != "ok":
            results[pmid] = {
                "status": info.get("status"),
                "pmcid": info.get("pmcid"),
                "title": info.get("title"),
                "error": info.get("error"),
            }
            continue

        text = info["text"]
        pmcid = info.get("pmcid")
        title = info.get("title")

        # ---- pass meta through to the backend (Groq or Gemini) ----
        meta = {
            "pmid": pmid,
            "pmcid": pmcid,
            "virus_filter": (virus_filter or None),
            "protein_filter": (protein_filter or None),
            "exhaustive": exhaustive,
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "delay_ms": delay_ms,
        }
        if llm_meta:
            meta.update(llm_meta)

        raw = run_on_paper(text, meta=meta)

        cleaned = clean_and_ground(
            raw,
            text,
            restrict_to_paper=True,
            require_mutation_in_quote=require_mut_quote,
            min_confidence=min_confidence,
        )
        if "paper" in cleaned:
            cleaned["paper"]["title"] = title

        results[pmid] = {
            "status": "ok",
            "pmcid": pmcid,
            "title": title,
            "result": cleaned,
        }

        # ---- NEW: gentle pacing between papers to avoid 429s ----
        if paper_pause_sec and paper_pause_sec > 0:
            time.sleep(paper_pause_sec)

    return results


def flatten_to_rows(batch: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert batch LLM outputs into a row-per-finding table.

    Columns:
      pmid, pmcid, title, virus, source_strain, protein, mutation, position,
      effect_category, confidence, effect_summary, quote_1, quote_2
    """
    rows: List[Dict] = []
    for pmid, entry in batch.items():
        pmcid = entry.get("pmcid")
        title = entry.get("title")
        if entry.get("status") != "ok":
            # Skip non-ok rows in the findings table; they appear elsewhere in UI
            continue

        seq = (entry.get("result", {}) or {}).get("sequence_features", []) or []
        for f in seq:
            quotes = [q for q in (f.get("evidence_quotes") or []) if isinstance(q, str)]
            q1 = quotes[0] if len(quotes) > 0 else ""
            q2 = quotes[1] if len(quotes) > 1 else ""

            rows.append({
                "pmid": pmid,
                "pmcid": pmcid or "",
                "title": title or "",
                "virus": f.get("virus") or "",
                "source_strain": f.get("source_strain") or "",
                "protein": f.get("protein") or "",
                "mutation": f.get("mutation") or "",
                "position": f.get("position"),
                "effect_category": f.get("effect_category") or "",
                "confidence": f.get("confidence"),
                "effect_summary": f.get("effect_summary") or "",
                "quote_1": q1,
                "quote_2": q2,
            })
    return pd.DataFrame(rows)

#!/usr/bin/env python3
# compare_quotes.py — PMID-scoped matching that understands "URL row + following quote rows"
# v2025-11-05

import argparse
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import pandas as pd

# ===== DEFAULT WINDOWS PATHS (override via CLI if needed) =====
DEFAULT_MANUAL_XLSX = Path(r"/Users/shubhamlaxmikantdeshmukh/Downloads/Manually_Curatd_Influenza.xlsx")
DEFAULT_LATEST_CSV  = Path(r"/Users/shubhamlaxmikantdeshmukh/Downloads/Test_pubmed_mutation_findings.csv")

PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", re.IGNORECASE)

# Choose an Excel engine based on file extension
def _choose_excel_engine(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return "openpyxl"   # pip install openpyxl  (or conda install openpyxl)
    if ext in {".xls"}:
        return "xlrd"       # pip install xlrd     (or conda install xlrd)
    if ext in {".ods"}:
        return "odf"        # pip install odfpy    (or conda install odfpy)
    # Fall back to openpyxl for unknown-but-probably-xlsx
    return "openpyxl"


# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "â€²": "′", "â€“": "–", "â€”": "—", "â€¦": "…",
        "Î³": "γ", "Î±": "α", "Â": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s).lower()
    s = " ".join(s.split())
    return s

def fuzzy_ratio(a: str, b: str) -> int:
    return int(SequenceMatcher(a=a, b=b).ratio() * 100)

def split_cell(cell: str) -> List[str]:
    """Split a cell into one or more sentences conservatively (newlines and ' | ')."""
    if not isinstance(cell, str):
        return []
    out: List[str] = []
    for chunk in cell.splitlines():
        for sub in chunk.split(" | "):
            sub = sub.strip()
            if sub:
                out.append(sub)
    return out

# ---------- Extraction from Excel (wide + tall) ----------
def extract_manual_quotes_from_excel(xlsx_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Reads the Excel with header=None, scans row-by-row.
    State machine:
      - If a row contains a PubMed URL -> set current_pmid to that PMID
      - Collect all non-URL text cells in that row as quotes (wide)
      - For following rows without a PubMed URL, while current_pmid is set,
        collect their non-empty cells as quotes (tall)
      - When a new row contains another PubMed URL -> switch current_pmid
    Returns:
      df_manual_quotes: [pmid, quote, source_row, source_col]
      pmid_to_manual: pmid -> list of quotes
    """
    engine = _choose_excel_engine(xlsx_path)
    try:
        df_raw = pd.read_excel(xlsx_path, header=None, dtype=str, engine=engine)
    except ValueError as e:
        raise RuntimeError(
            f"Failed to read Excel file '{xlsx_path}' with engine='{engine}'. "
            f"Install the required engine (e.g., 'pip install {engine}' or 'conda install {engine}'). "
            f"Original error: {e}"
        )

    rows: List[Dict] = []
    pmid_to_manual: Dict[str, List[str]] = {}

    current_pmid: str | None = None

    for r_idx in range(df_raw.shape[0]):
        row = df_raw.iloc[r_idx]
        # Collect all non-empty strings in the row with their column index
        cells = [(c_idx, c) for c_idx, c in enumerate(row.tolist()) if isinstance(c, str) and c.strip()]
        if not cells:
            # empty row; keep current_pmid in case quotes continue further down
            continue

        # Does this row start a new paper?
        found_pmid = None
        url_cols = set()
        for c_idx, c in cells:
            m = PMID_URL_RE.search(c)
            if m:
                found_pmid = m.group(1)
                url_cols.add(c_idx)

        if found_pmid is not None:
            current_pmid = found_pmid

        # If we have an active PMID, collect quotes from all non-URL cells in the row
        if current_pmid is not None:
            text_cells = [(c_idx, c) for c_idx, c in cells if c_idx not in url_cols]
            # Split and dedupe by normalized text within this row
            seen_row = set()
            row_quotes: List[Tuple[int, str]] = []
            for c_idx, c in text_cells:
                for s in split_cell(c):
                    ns = normalize_text(s)
                    if ns and ns not in seen_row:
                        seen_row.add(ns)
                        row_quotes.append((c_idx, s))
            if row_quotes:
                pmid_to_manual.setdefault(current_pmid, []).extend([s for _, s in row_quotes])
                for c_idx, s in row_quotes:
                    rows.append({"pmid": current_pmid, "quote": s, "source_row": r_idx, "source_col": c_idx})

    df_manual_quotes = pd.DataFrame(rows, columns=["pmid", "quote", "source_row", "source_col"])
    return df_manual_quotes, pmid_to_manual

# ---------- Extraction from CSV ----------
def extract_latest_quotes_from_csv(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[int, str, str]]], pd.DataFrame]:
    """
    Ensures CSV has pmid & quote. Returns:
      df_latest_quotes: flattened quotes table (pmid, quote, + metadata)
      pmid_to_latest: pmid -> list of (global_row_index, quote, normalized_quote)
      df_latest: full CSV for lookups
    """
    df_latest = pd.read_csv(csv_path, dtype=str)
    if "pmid" not in df_latest.columns:
        raise ValueError(f"Latest CSV missing 'pmid'. Columns: {list(df_latest.columns)}")
    if "quote" not in df_latest.columns:
        guess = [c for c in df_latest.columns if "quote" in c.lower()]
        if not guess:
            raise ValueError("Latest CSV needs a 'quote' column (or similar).")
        df_latest["quote"] = df_latest[guess[0]]

    df_latest["pmid"] = df_latest["pmid"].astype(str)
    df_latest["pmid_str"] = df_latest["pmid"].str.extract(r"(\d+)")[0]

    pmid_to_latest: Dict[str, List[Tuple[int, str, str]]] = {}
    for i, row in df_latest.iterrows():
        pm = row["pmid_str"]
        q = (row["quote"] or "").strip()
        qn = normalize_text(q)
        if pm:
            pmid_to_latest.setdefault(pm, []).append((i, q, qn))

    cols = ["pmid", "quote"]
    extra = [c for c in ["pmcid", "title", "virus", "protein", "mutation", "position", "confidence", "target_type"]
             if c in df_latest.columns]
    df_latest_quotes = df_latest[cols + extra].copy()
    return df_latest_quotes, pmid_to_latest, df_latest

# ---------- Matching ----------
def match_by_pmid(
    pmid_to_manual: Dict[str, List[str]],
    pmid_to_latest: Dict[str, List[Tuple[int, str, str]]],
    df_latest: pd.DataFrame,
    fuzzy_threshold: int = 85,
    show_non_matches: bool = False,
):
    summary_rows, detail_rows, totals_rows = [], [], []
    pmids_without_csv: List[str] = []

    for pmid, manual_sentences in pmid_to_manual.items():
        latest_group = pmid_to_latest.get(pmid, [])
        if not latest_group:
            pmids_without_csv.append(pmid)
            if show_non_matches:
                print(f"[PMID {pmid}] present in Excel but missing in CSV.")
            for m_idx, s in enumerate(manual_sentences):
                summary_rows.append({
                    "pmid": pmid, "pmid_manual_sentence_index": m_idx,
                    "manual_sentence": s,
                    "exact_match_count": 0, "substring_match_count": 0,
                    f"fuzzy_match_count(>={fuzzy_threshold})": 0,
                    "best_fuzzy_score": 0, "best_row_index": None,
                })
            totals_rows.append({
                "pmid": pmid, "manual_sentence_count": len(manual_sentences),
                "csv_quote_count": 0, "matched_sentence_count": 0,
            })
            continue

        latest_indices = [i for i, _, _ in latest_group]
        latest_quotes  = [q for _, q, _ in latest_group]
        latest_norms   = [qn for _, _, qn in latest_group]

        print(f"[PMID {pmid}] manual sentences: {len(manual_sentences)} | csv quotes: {len(latest_group)}")
        matched_sentences = 0

        for m_idx, s in enumerate(manual_sentences):
            s_norm = normalize_text(s)
            if not s_norm:
                continue

            exact_hits, substr_hits, fuzzy_hits = [], [], []  # fuzzy=(j, global_i, score)

            for j, qn in enumerate(latest_norms):
                if s_norm == qn:
                    exact_hits.append(j)
                if s_norm in qn or qn in s_norm:
                    substr_hits.append(j)
                sc = fuzzy_ratio(s_norm, qn)
                if sc >= fuzzy_threshold:
                    fuzzy_hits.append((j, latest_indices[j], sc))

            best_score = max([sc for _, _, sc in fuzzy_hits], default=0)
            best_row_global = None
            if fuzzy_hits:
                best_row_global = max(fuzzy_hits, key=lambda x: x[2])[1]
            elif substr_hits:
                best_row_global = latest_indices[substr_hits[0]]

            if exact_hits or substr_hits or fuzzy_hits:
                matched_sentences += 1

            # Debug print for this sentence
            print(f"  - [{m_idx}] {s}")
            print(f"    exact rows(j): {exact_hits}")
            print(f"    substr rows(j): {substr_hits[:10]}{' ...' if len(substr_hits)>10 else ''}")
            print(f"    fuzzy rows(j,global_i,score≥{fuzzy_threshold}): {[(j,g,sc) for (j,g,sc) in fuzzy_hits[:10]]}"
                  f"{' ...' if len(fuzzy_hits)>10 else ''}")
            if best_row_global is not None:
                print("    BEST MATCH QUOTE:", df_latest.loc[best_row_global, "quote"])

            summary_rows.append({
                "pmid": pmid, "pmid_manual_sentence_index": m_idx,
                "manual_sentence": s,
                "exact_match_count": len(exact_hits),
                "substring_match_count": len(substr_hits),
                f"fuzzy_match_count(>={fuzzy_threshold})": len(fuzzy_hits),
                "best_fuzzy_score": best_score,
                "best_row_index": best_row_global,
            })

            # details across all hits
            match_js = set(exact_hits) | set(substr_hits) | {j for (j, _, _) in fuzzy_hits}
            for j in sorted(match_js):
                global_i = latest_indices[j]
                row = {
                    "pmid": pmid,
                    "pmid_manual_sentence_index": m_idx,
                    "manual_sentence": s,
                    "latest_global_index": global_i,
                    "fuzzy_score": fuzzy_ratio(s_norm, latest_norms[j]),
                    "latest_quote": latest_quotes[j],
                }
                for col in ["pmid", "pmcid", "title", "virus", "protein", "mutation", "position", "confidence", "target_type"]:
                    if col in df_latest.columns:
                        row[col] = df_latest.loc[global_i, col]
                detail_rows.append(row)

        totals_rows.append({
            "pmid": pmid,
            "manual_sentence_count": len(manual_sentences),
            "csv_quote_count": len(latest_group),
            "matched_sentence_count": matched_sentences,
        })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(detail_rows),
        pd.DataFrame(totals_rows),
        pd.DataFrame({"pmid": pmids_without_csv})
    )

# ---------- Main ----------
def run(
    manual_xlsx: Path,
    latest_csv: Path,
    save_dir: Path,
    fuzzy_threshold: int,
    show_non_matches: bool,
    limit_pmids: Iterable[str] | None,
    preview_pmid: str | None,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract BOTH sides (and export raw quotes so you can check what was read)
    df_manual_quotes, pmid_to_manual = extract_manual_quotes_from_excel(manual_xlsx)
    df_latest_quotes, pmid_to_latest, df_latest = extract_latest_quotes_from_csv(latest_csv)

    path_manual_quotes = save_dir / "debug_manual_quotes.csv"
    path_latest_quotes = save_dir / "debug_latest_quotes.csv"
    df_manual_quotes.to_csv(path_manual_quotes, index=False, encoding="utf-8")
    df_latest_quotes.to_csv(path_latest_quotes, index=False, encoding="utf-8")

    print("\n===== PREVIEW: MANUAL EXCEL QUOTES (first 12) =====")
    print(df_manual_quotes.head(12).to_string(index=False) if not df_manual_quotes.empty else "(none)")

    print("\n===== PREVIEW: LATEST CSV QUOTES (first 12) =====")
    print(df_latest_quotes.head(12).to_string(index=False) if not df_latest_quotes.empty else "(none)")

    if preview_pmid:
        print(f"\n===== FOCUSED PREVIEW FOR PMID {preview_pmid} =====")
        man_subset = df_manual_quotes[df_manual_quotes["pmid"] == str(preview_pmid)]
        lat_subset = df_latest_quotes[df_latest_quotes["pmid"].astype(str).str.contains(str(preview_pmid))]
        print("\n-- Manual Excel quotes --")
        print(man_subset[["pmid", "quote"]].to_string(index=False) if not man_subset.empty else "(none)")
        print("\n-- Latest CSV quotes --")
        print(lat_subset[["pmid", "quote"]].to_string(index=False) if not lat_subset.empty else "(none)")

    # Optional: restrict PMIDs
    if limit_pmids:
        keep = {str(x) for x in limit_pmids}
        pmid_to_manual = {p: s for p, s in pmid_to_manual.items() if p in keep}

    # Matching (PMID-scoped)
    print("\n=== START MATCHING (within PMID) ===")
    df_summary, df_detail, df_totals, df_missing = match_by_pmid(
        pmid_to_manual, pmid_to_latest, df_latest,
        fuzzy_threshold=fuzzy_threshold,
        show_non_matches=show_non_matches
    )

    # ---- Add match percentage column (e.g., "23/31 (74.2%)")
    if not df_totals.empty:
        pct = (df_totals["matched_sentence_count"] / df_totals["manual_sentence_count"] * 100)
        df_totals["match_percentage"] = (
            df_totals["matched_sentence_count"].astype(str)
            + "/"
            + df_totals["manual_sentence_count"].astype(str)
            + " ("
            + pct.round(1).astype(str)
            + "%)"
        )

    # Save matching outputs
    out_summary = save_dir / "match_summary.csv"
    out_details = save_dir / "match_details.csv"
    out_totals  = save_dir / "pmid_totals.csv"
    out_missing = save_dir / "pmid_no_match.csv"

    df_summary.to_csv(out_summary, index=False, encoding="utf-8")
    df_detail.to_csv(out_details, index=False, encoding="utf-8")
    df_totals.to_csv(out_totals, index=False, encoding="utf-8")
    df_missing.to_csv(out_missing, index=False, encoding="utf-8")

    print("\n== DONE ==")
    print(f"Saved: {path_manual_quotes}   (all manual quotes)")
    print(f"Saved: {path_latest_quotes}   (all latest quotes)")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_details}")
    print(f"Saved: {out_totals}")
        # ---- Unmatched exports ----
    # 1) Manual sentences that didn't match anything
    if not df_summary.empty:
        zero_match_mask = (
            (df_summary["exact_match_count"] == 0)
            & (df_summary["substring_match_count"] == 0)
            & (df_summary[f"fuzzy_match_count(>={fuzzy_threshold})"] == 0)
        )
        df_unmatched_manual = df_summary.loc[
            zero_match_mask,
            ["pmid", "pmid_manual_sentence_index", "manual_sentence", "best_fuzzy_score", "best_row_index"]
        ].copy()
    else:
        df_unmatched_manual = pd.DataFrame(columns=[
            "pmid", "pmid_manual_sentence_index", "manual_sentence", "best_fuzzy_score", "best_row_index"
        ])

    # 2) Latest quotes that were never hit by any match
    if not df_detail.empty:
        matched_idx = set(df_detail["latest_global_index"].dropna().astype(int).unique())
    else:
        matched_idx = set()
    df_unmatched_latest_all = df_latest.loc[~df_latest.index.isin(matched_idx)].copy()

    # 3) Same as (2) but restricted to PMIDs that also appear in the MANUAL Excel
    manual_pmids = set(pmid_to_manual.keys())
    if "pmid_str" not in df_latest.columns:
        df_latest["pmid_str"] = df_latest["pmid"].astype(str).str.extract(r"(\d+)")[0]

    if not df_unmatched_latest_all.empty:
        df_unmatched_latest_within = df_unmatched_latest_all[
            df_unmatched_latest_all["pmid_str"].isin(manual_pmids)
        ].copy()
    else:
        df_unmatched_latest_within = pd.DataFrame(columns=df_latest.columns)

    # Write them out
    out_unmatched_manual = save_dir / "unmatched_manual_sentences.csv"
    out_unmatched_latest_all = save_dir / "unmatched_latest_quotes_all.csv"
    out_unmatched_latest_within = save_dir / "unmatched_latest_quotes_within_excel_pmids.csv"

    df_unmatched_manual.to_csv(out_unmatched_manual, index=False, encoding="utf-8")
    df_unmatched_latest_all.to_csv(out_unmatched_latest_all, index=False, encoding="utf-8")
    df_unmatched_latest_within.to_csv(out_unmatched_latest_within, index=False, encoding="utf-8")

    print(f"Saved: {out_unmatched_manual}  (manual sentences with no matches)")
    print(f"Saved: {out_unmatched_latest_all}  (latest quotes with no matches, all PMIDs)")
    print(f"Saved: {out_unmatched_latest_within}  (latest quotes with no matches, limited to Excel PMIDs)")


    if not df_missing.empty:
        print(f"Saved: {out_missing}  (PMIDs present in Excel but not in CSV)")
    else:
        print("All Excel PMIDs appeared in the CSV.")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Export quotes from Excel & CSV, then match them by PMID (wide + tall Excel layout).")
    ap.add_argument("--manual-xlsx", type=Path, default=DEFAULT_MANUAL_XLSX)
    ap.add_argument("--latest-csv",  type=Path, default=DEFAULT_LATEST_CSV)
    ap.add_argument("--save-dir",    type=Path, default=Path(".\\out"))
    ap.add_argument("--fuzzy-threshold", type=int, default=85)
    ap.add_argument("--show-non-matches", action="store_true")
    ap.add_argument("--limit-pmids", nargs="*", default=None)
    ap.add_argument("--preview-pmid", default="40925098", help="Focused preview PMID (blank to skip).")
    return ap.parse_args()

def main():
    args = parse_args()
    run(
        manual_xlsx=args.manual_xlsx,
        latest_csv=args.latest_csv,
        save_dir=args.save_dir,
        fuzzy_threshold=args.fuzzy_threshold,
        show_non_matches=args.show_non_matches,
        limit_pmids=args.limit_pmids,
        preview_pmid=(args.preview_pmid or None),
    )

if __name__ == "__main__":
    main()

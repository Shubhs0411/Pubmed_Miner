# PubMed Miner

A Streamlit app to search **PubMed** for review articles, fetch **PMC** full text (when available), and run a **Gemini**-powered extractor to mine mutation/protein findings with grounded snippets.

---

## Quick Start

### 1) Clone the repo
```bash
git clone https://github.com/Shubhs0411/Pubmed_Miner.git
cd Pubmed_Miner
```

### 2) Create & activate a virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure environment variables
Create a **.env** file in the project root with the following keys (adjust values as needed).

```dotenv
# Required
GEMINI_API_KEY="PUT_YOUR_KEY_HERE"

# Soft rate limits for Gemini (you can tune these if you hit quota)
GEMINI_RPM=10
GEMINI_TPM=180000

# Pause between papers in the batch pipeline (seconds)
PAPER_PAUSE_SEC=3.0
```

> Tip: You can export these as real environment variables instead of using `.env` if you prefer.

### 5) Run the app
```bash
streamlit run app.py
```
Open the URL that Streamlit prints (usually `http://localhost:8501`).

---

## Project Structure

```
Pubmed_Miner/
├─ app.py                 # Streamlit UI
├─ services/
│  ├─ pubmed.py           # PubMed search, summaries, date utils
│  └─ pmc.py              # PMC fetching (JATS-first with HTML fallback)
├─ pipeline/
│  └─ batch_analyze.py    # Batch fulltext fetch + LLM analysis + tabular flattening
├─ llm/
│  ├─ __init__.py         # Facade exports for Gemini/Groq
│  ├─ gemini.py           # Gemini two-pass extractor (tokens + attribution)
│  └─ groq.py             # Thin wrapper over legacy llm_groq.py for compatibility
├─ (extractor.py removed) # All callers should import from services.pmc
├─ llm_gemini.py          # Legacy location (kept for compatibility; consider migrating imports)
├─ llm_groq.py            # Legacy location (kept for compatibility; consider migrating imports)
```

Notes:
- Backwards compatibility is preserved. Existing imports like `from extractor import get_pmc_fulltext_with_meta` still work.
- New code should import from `llm.gemini` or `llm.groq` and `services.pmc` directly.

---

## Sample PubMed Query

When the app starts, you can try a query like:

```
((Dengue[Title]) AND (protein)) AND ((active site[Text Word]) OR (mutation[Text Word]))
```

This will search for Dengue-related protein review literature mentioning an active site or mutations.

---

## How to Use

1. **Search PubMed** – Enter your query and press search.
2. **Select PMIDs** – Choose papers to include in analysis.
3. **Fetch Full Text** – The app tries to pull **PMC** full text (if available).
4. **Run Extraction** – Use the Gemini-based pipeline to enumerate tokens and attribute them with short, grounded quotes.
5. **Review & Export** – Inspect results in the UI; download CSV/JSON as needed.

---

## Troubleshooting

- **`GEMINI_API_KEY not set`** – Add your key to `.env` (or export as an environment variable) and restart.
- **Rate limit/quota errors** – Lower `GEMINI_RPM` and/or `GEMINI_TPM`, or increase `PAPER_PAUSE_SEC`.
- **Some PMIDs show no PMC text** – The paper may be embargoed or not deposited in PMC; the app will still process available items.

---

## Notes

- The app focuses on **Review** articles and handles date filters and pagination.
- Fetching relies on PMC availability; for non-PMC papers, full text may not be obtainable.
- The extractor runs in two passes to improve grounding and precision.

---

## License

MIT (update as you prefer).

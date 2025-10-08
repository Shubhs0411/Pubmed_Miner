# PubMed Review Miner (Streamlit)

Search PubMed for **Review** articles, fetch **PMC** full text, and run a two‑pass Gemini extractor to mine mutation/protein findings. The app is built with **Streamlit** and a small pipeline around PubMed E‑utilities + PMC HTML parsing.

![Folder structure](assets/folder-structure.png)

---

## Features (high level)
- PubMed search restricted to **Review** articles, with robust date handling and pagination.
- Fetch **PMC** full text (when available) and gracefully handle **embargo/403** and **rate limits**.
- Two‑pass **Gemini** pipeline: enumerate tokens (mutations/proteins/amino acids) then attribute/ground with short quotes.
- Streamlit UI: search → select PMIDs → fetch full text → run LLM → view/download **findings CSV** and **raw JSON**.
- Optional publisher fallback metadata via **Unpaywall** when PMC is unavailable.

> Key modules: `services/pubmed.py`, `pipeline/batch_analyze.py`, `extractor.py`, `llm_gemini.py`, and `app.py`.

---

## Quickstart

### 1) Clone & create a virtual environment
```bash
# Clone your fork
git clone <your-fork-url> PUBMED_MINER
cd PUBMED_MINER

# Create & activate a virtual environment (pick one)
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2) Install Python dependencies
```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt`, the core packages are roughly:
> `streamlit pandas python-dotenv requests urllib3 beautifulsoup4 google-generativeai`

### 3) Create a `.env` file (API keys & pacing)
Create a file named `.env` in the project root with at least your Gemini key. Reasonable free‑tier defaults are shown below—you can tune later.

```dotenv
# --- required ---
GEMINI_API_KEY=""

# --- soft rate limits (tune for your account tier) ---
# requests per minute and tokens per minute for Gemini
GEMINI_RPM=10
GEMINI_TPM=180000

# pause between papers in the batch pipeline (seconds)
PAPER_PAUSE_SEC=3.0

# --- optional but recommended ---
# Improves PubMed reliability when searching/fetching
NCBI_API_KEY=""

# Identifies your requests to NCBI (used in request headers)
CONTACT_EMAIL="you@example.com"

# Enables publisher access checks when PMC is not available
UNPAYWALL_EMAIL="you@example.com"

# If you want to try a Groq-based LLM (not required for Gemini path)
GROQ_API_KEY=""
```

### 4) Run the Streamlit app
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## How the pieces fit

- **PubMed search & date parsing** – robust E‑utilities wrapper with tolerant JSON parsing and date‑range filtering in the app (see `services/pubmed.py`).  
- **Fetching PMC full text** – maps PMID⇄PMCID, scrapes PMC HTML, strips figures/tables/nav to plain text; handles embargo/403 and retryable errors (see `extractor.py`).  
- **Batch pipeline** – fetch all full texts for selected PMIDs, then analyze each paper; includes pacing between papers via `PAPER_PAUSE_SEC` (see `pipeline/batch_analyze.py`).  
- **Gemini two‑pass extractor** – token enumeration over chunks → token‑focused attribution with grounded quotes, with soft RPM/TPM gates; configure via `.env` (see `llm_gemini.py`).  
- **Streamlit UI** – one‑page app to search, select, fetch, inspect full text, run the LLM, and download results (see `app.py`).

---

## Usage tips

- Start with a narrow PubMed query (e.g., a virus and “mutation”) and a smaller date range, then scale up.
- If you see rate‑limit messages from Gemini, lower `GEMINI_RPM` / `GEMINI_TPM` in `.env` or widen pauses.
- If some PMIDs show “no PMC full text,” they may be **embargoed** or not deposited in PMC. The app surfaces these and still lets you download whatever was fetched for others.
- Set `NCBI_API_KEY` to reduce throttling when doing large searches.

---

## Troubleshooting

- **`RuntimeError: GEMINI_API_KEY not set`** – you must put your key in `.env` (or environment) before running.
- **429 / quota or 5xx from Gemini** – the pipeline implements soft gates and retries; reduce `GEMINI_RPM/TMP`, increase `PAPER_PAUSE_SEC`, or try again.
- **`pmc_embargo_or_blocked`** – PMC exists but access is blocked (often an embargo). The tool records PMCID and skips text.

---

## Minimal programmatic example

If you prefer Python over the UI, you can fetch and analyze a small list of PMIDs like so:

```python
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts, flatten_to_rows

pmids = ["12345678", "34567890"]
papers = fetch_all_fulltexts(pmids, delay_ms=200)
results = analyze_texts(papers, virus_filter="Dengue virus", protein_filter="protein")
df = flatten_to_rows(results)
print(df.head())
```

---

## License
MIT (or your preferred license).


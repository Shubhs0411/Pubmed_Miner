# PubMed Miner

A Streamlit app to search **PubMed** for review articles, fetch **PMC** full text (when available), and run a **Gemini**-powered extractor to mine mutation/protein findings with grounded snippets.

---

## 🚀 Choose Your Setup Method

### Option 1: Traditional Python Setup
For developers who want to run the app directly with Python and have full control over the environment.

### Option 2: Docker Setup (Recommended)
For users who want a consistent, isolated environment that works the same on any system.

---

## 📋 Prerequisites

### For Python Setup
- **Python 3.11+** (tested with Python 3.11.8)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB+ RAM recommended
- **Internet**: Required for API calls and PMC fetching

### For Docker Setup
- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Git** for cloning the repository
- **Memory**: 4GB+ RAM recommended
- **Internet**: Required for API calls and PMC fetching

---

## 🐍 Option 1: Traditional Python Setup

### Step 1: Python Version Setup

#### Check your Python version:
```bash
python --version
# or
python3 --version
```

#### If you need to install Python 3.11+:

**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Choose Python 3.11+ (latest stable)
3. Check "Add Python to PATH" during installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install python3.11 python3.11-venv python3.11-pip
```

### Step 2: Clone the repository
```bash
git clone https://github.com/Shubhs0411/Pubmed_Miner.git
cd Pubmed_Miner
```

### Step 3: Create & activate virtual environment

#### Create virtual environment:
```bash
# Using Python 3.11+ (replace with your Python version)
python3.11 -m venv .venv
# or
python -m venv .venv
```

#### Activate virtual environment:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

#### Verify activation:
```bash
# You should see (.venv) in your prompt
which python
# Should show: /path/to/Pubmed_Miner/.venv/bin/python
```

#### Deactivate when done:
```bash
deactivate
```

### Step 4: Install dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install additional ML libraries if needed
# pip install torch transformers huggingface-hub
```

### Step 5: Configure API keys

#### Required: Get a Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API key" → "Create API key"
4. Copy the generated key

#### Required: Get an NCBI API Key
1. Go to [NCBI API Key Registration](https://www.ncbi.nlm.nih.gov/account/settings/)
2. Sign in to your NCBI account (create one if needed)
3. Go to "API Key Management" → "Create API Key"
4. Copy the generated key

**Why NCBI API key is required?**
- **Without it**: 3 requests/second, 10 requests/minute
- **With it**: 10 requests/second, 50 requests/minute
- **Result**: Faster PMC fetching, fewer rate limit errors
- **Required for**: Reliable operation and better performance

#### Create .env file
Create a **.env** file in the project root:

```dotenv
# Required: Choose ONE LLM backend
GEMINI_API_KEY="your_gemini_api_key_here"
# OR
GROQ_API_KEY="your_groq_api_key_here"

# Required: NCBI API key for reliable operation
NCBI_API_KEY="your_ncbi_api_key_here"

# Optional: Rate limiting (adjust if you hit quotas)
GEMINI_RPM=10
GEMINI_TPM=180000
PAPER_PAUSE_SEC=3.0

# Optional: Contact info for NCBI
CONTACT_EMAIL="your_email@example.com"
```

> **Note**: You can export these as environment variables instead of using `.env` if you prefer.

### Step 6: Run the app
```bash
# Method 1: Using the root shim (recommended)
streamlit run app.py

# Method 2: Direct execution
streamlit run app/app.py
```

Open the URL that Streamlit prints (usually `http://localhost:8501`).

### Step 7: Verify installation
The app should display:
- ✅ PubMed Review Miner title
- ✅ LLM Settings sidebar
- ✅ Query input area
- ✅ Date range selector

---

## 🐳 Option 2: Docker Setup (Recommended)

### Step 1: Clone the repository
```bash
git clone https://github.com/Shubhs0411/Pubmed_Miner.git
cd Pubmed_Miner
```

### Step 2: Configure environment
Create a `.env` file with your API keys:
```bash
# Copy the example and edit
cp env.example .env
# Edit .env with your API keys
```

**Required API Keys:**
- **GEMINI_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/)
- **NCBI_API_KEY**: Get from [NCBI API Key Registration](https://www.ncbi.nlm.nih.gov/account/settings/)

### Step 3: Run with Docker Compose
```bash
# Build and start the application
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Step 4: Access the application
- Open your browser to: `http://localhost:8502`
- The app will be running in a containerized environment

### Step 5: Stop the application
```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Docker Commands Reference

```bash
# Build the image
docker build -t pubmed-miner .

# Run the container
docker run -p 8502:8501 --env-file .env pubmed-miner

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up --build --force-recreate

# Clean up
docker-compose down -v --rmi all
```

### Docker Benefits
- ✅ **Consistent environment** across all systems
- ✅ **No Python version conflicts**
- ✅ **Easy deployment** and scaling
- ✅ **Isolated dependencies**
- ✅ **Production-ready** setup

---

## Project Structure

```
Pubmed_Miner/
├── app/                          # Streamlit UI module
│   ├── __init__.py              # Package initialization
│   └── app.py                   # Main Streamlit application
├── app.py                       # Root shim (redirects to app/app.py)
├── llm/                         # LLM backends module
│   ├── __init__.py              # Package initialization
│   ├── gemini.py                # Google Gemini integration
│   └── groq.py                  # Groq API integration
├── pipeline/                    # Batch processing module
│   ├── __init__.py              # Package initialization
│   └── batch_analyze.py         # Batch fetch + LLM analysis
├── services/                    # External API services
│   ├── __init__.py              # Package initialization
│   ├── pmc.py                   # PMC full-text fetching
│   └── pubmed.py                # PubMed search & metadata
├── venv/                        # Virtual environment (excluded from git)
├── .env                         # Environment variables (excluded from git)
├── .gitignore                   # Git ignore patterns
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

### Module Responsibilities

**`app/`** - User Interface
- `app.py`: Streamlit web interface with search, selection, and results display
- Handles user interactions and data visualization

**`llm/`** - Language Model Integration  
- `gemini.py`: Google Gemini API integration for mutation extraction
- `groq.py`: Groq API integration (alternative backend)
- Both provide identical APIs: `run_on_paper()`, `clean_and_ground()`

**`pipeline/`** - Batch Processing
- `batch_analyze.py`: Orchestrates PMC fetching + LLM analysis
- Functions: `fetch_all_fulltexts()`, `analyze_texts()`, `flatten_to_rows()`

**`services/`** - External APIs
- `pmc.py`: PMC full-text retrieval (JATS XML + HTML fallback)
- `pubmed.py`: PubMed search, metadata, and date filtering
- Functions: `esearch_reviews()`, `esummary()`, `get_pmc_fulltext_with_meta()`

### Import Examples
```python
# UI imports
from app.app import main

# LLM backends
from llm.gemini import run_on_paper, clean_and_ground
from llm.groq import run_on_paper, clean_and_ground

# Services
from services.pmc import get_pmc_fulltext_with_meta, get_last_fetch_source
from services.pubmed import esearch_reviews, esummary

# Pipeline
from pipeline.batch_analyze import fetch_all_fulltexts, analyze_texts
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

### Common Issues

**`GEMINI_API_KEY not set` or `GROQ_API_KEY not set`**
- Add your API key to `.env` file or export as environment variable
- Restart the app after adding the key

**`NCBI_API_KEY not set`**
- NCBI API key is required for reliable operation
- Get your key from [NCBI API Key Registration](https://www.ncbi.nlm.nih.gov/account/settings/)
- Add it to your `.env` file: `NCBI_API_KEY="your_key_here"`
- Restart the app after adding the key

**Rate limit/quota errors**
- Lower `GEMINI_RPM` and/or `GEMINI_TPM` in `.env`
- Increase `PAPER_PAUSE_SEC` for slower processing
- **Ensure you have an NCBI API key** - it's required for reliable operation (3→10 req/sec, 10→50 req/min)

**Some PMIDs show no PMC text**
- The paper may be embargoed or not deposited in PMC
- The app will still process available items

**Import errors or missing modules**
- Ensure you're using Python 3.11+
- Check virtual environment is activated: `which python` should show `.venv/bin/python`
- Reinstall dependencies: `pip install -r requirements.txt`
- Try: `pip install --upgrade pip` then `pip install -r requirements.txt`

**Blank page or app won't start**
- Try: `streamlit run app/app.py` directly
- Check console for error messages
- Ensure all dependencies are installed

### System Requirements
- **Python**: 3.11+ (tested with 3.11.8)
- **RAM**: 4GB+ recommended
- **Storage**: 2GB+ free space
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

---

## Notes

- The app focuses on **Review** articles and handles date filters and pagination.
- Fetching relies on PMC availability; for non-PMC papers, full text may not be obtainable.
- The extractor runs in two passes to improve grounding and precision.

---

## License

MIT (update as you prefer).

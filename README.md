# Multilingual Agentic Fact-Checker (Fake News Checker Assistant)

An end-to-end, multi-step Generative AI verification agent designed to fact-check suspicious WhatsApp forwards and social media messages across **English**, **Hindi (हिन्दी)**, **Marathi (मराठी)**, and **Hinglish/Mixed** text. The system executes a transparent four-stage pipeline (**Pre-Scan → Multilingual Claim Extractor → Live Web & Fact-Check Retriever → Evidence-Grounded Synthesizer**) backed by source credibility tiering, indirect prompt-injection defenses, bilingual explanations, and a 30-message multilingual evaluation harness.

> **Verification Policy & Disclaimer:** This assistant synthesizes retrieved live web and fact-check evidence to support human judgment. It is not an infallible arbiter of truth. Whenever live search is disabled or returns no credible sources, verdicts automatically default to **Insufficient Evidence** rather than guessing from static LLM memory. Always inspect the linked primary sources before forwarding.

## Key Capabilities

* **4-Stage Observable Verification Agent (`factcheck_agent.py`):**
  * **Stage 0 (Language & Security Pre-Scan):** Detects English, Hindi, Marathi (including Marathi-specific Devanagari character `ळ` and regional vocabulary), and Hinglish, while scanning for indirect prompt-injection directives.
  * **Stage 1 (Multilingual Claim Extractor):** Separates check-worthy factual claims from opinions and forwarding pleas, preserves the original-language claim, translates it to canonical English, and generates dual verification search queries.
  * **Stage 2 (Live Web & Fact-Check Retriever):** Queries DuckDuckGo live (and the Google Fact Check Tools API automatically when `GOOGLE_FACTCHECK_API_KEY` is configured), deduplicates results by URL, and ranks sources using a 3-tier domain credibility rubric.
  * **Stage 3 (Evidence-Grounded Synthesizer & URL Gate):** Assigns one of four calibrated verdicts (`Supported by Evidence`, `Contradicted by Evidence`, `Misleading / Out of Context`, or `Insufficient Evidence`), generates bilingual English + Hindi/Marathi explanations, and deterministically verifies that every cited URL came verbatim from the retrieved evidence pool.
* **30-Message Multilingual Benchmark (`factcheck_evaluator.py`, `benchmark_30_messages.csv`):** Evaluates Claim Extraction Recall, Verdict Precision/Recall/Macro F1, Confusion Matrix, Language-Wise Accuracy (13 English, 9 Hindi, 8 Marathi messages), Citation Grounding Rate, and per-stage latency.
* **Indirect Prompt-Injection Red-Team Lab (`factcheck_evaluator.py`):** Tests four built-in payloads (`Direct Verdict Override`, `System Prompt Spoof in Hindi`, `XML Tag Escape Attempt`, and a `Benign Real News Control`) and computes the Attack Block Rate.
* **4-Tab Streamlit Workbench (`factcheck_app.py`):** Includes demo presets, clickable Tier-1/2/3 evidence cards, an expandable Agent Execution Trace, JSON/CSV report downloads, live benchmark execution, and a transparent Source Credibility Rubric tab.

## Repository Structure

| File | Responsibility |
| :--- | :--- |
| `factcheck_app.py` | 4-tab Streamlit UI (Live Agent, 30-Message Benchmark, Red-Team Lab, Source Credibility Rubric) with JSON/CSV exports. |
| `factcheck_agent.py` | 4-stage multilingual agent orchestrator, language detector, injection guard, bilingual synthesizer, and deterministic URL grounding gate. |
| `live_retriever.py` | DuckDuckGo live search, optional Google Fact Check Tools API client, offline fallback mode, and Tier 1/2/3 domain credibility classifier. |
| `factcheck_schemas.py` | Strict Pydantic v2 data contracts (`ConfigDict(extra="forbid")`) for extracted claims, evidence items, verified verdicts, and full reports. |
| `factcheck_evaluator.py` | 30-message multilingual benchmark runner, language-wise breakdown, confusion matrix plotter, and indirect prompt-injection red-team suite. |
| `benchmark_30_messages.csv` | Curated 30-message test set (18 hoaxes, 11 verified real news items, 1 insufficient-evidence control across English, Hindi, and Marathi). |
| `factcheck_requirements.txt` | Python package dependencies. |

## Technology Stack

| Component | Technology | Role in System |
| :--- | :--- | :--- |
| LLM Inference | Groq API (`openai/gpt-oss-120b`) | Deterministic JSON-mode claim extraction and evidence-grounded synthesis (`temperature=0`). |
| Live Web Retrieval | DuckDuckGo (`ddgs`) & Google Fact Check API | Real-time web search and IFCN fact-check lookup without requiring paid search APIs. |
| Schema Enforcement | Pydantic v2 (`pydantic>=2.7`) | Strict runtime validation of extracted claims, confidence bounds (`0.0–1.0`), and verdict enums. |
| Evaluation & Metrics | Scikit-Learn, Pandas, Matplotlib | Macro Precision/Recall/F1, Confusion Matrix, language-wise accuracy, and latency telemetry. |
| Web Interface | Streamlit | Interactive 4-tab demo workbench with Agent Trace inspection and UTF-8-SIG CSV / JSON downloads. |

## Source Credibility Rubric

| Tier | Priority | Example Domains |
| :--- | :--- | :--- |
| **Tier 1 (Official / Fact-Checker)** | Highest | `pib.gov.in`, `altnews.in`, `boomlive.in`, `factly.in`, `vishvasnews.com`, `rbi.org.in`, `isro.gov.in`, `who.int`, `reuters.com`, `apnews.com`, `*.gov.in`, `*.nic.in` |
| **Tier 2 (Established News)** | Medium | `thehindu.com`, `indianexpress.com`, `ndtv.com`, `bbc.com`, `lokmat.com`, `maharashtratimes.com`, `esakal.com`, `bhaskar.com` |
| **Tier 3 (General Web)** | Lowest | Unverified blogs, forums, and general web pages |

## Quickstart

### 1. Prerequisites

* Python 3.10+
* A Groq API key (`GROQ_API_KEY`)
* Optional: A Google Fact Check Tools API key (`GOOGLE_FACTCHECK_API_KEY`)

### 2. Installation

```bash
git clone <your-repository-url>
cd Fake-News-Checker-Assistant
python -m pip install -r factcheck_requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_api_key
GROQ_MODEL=openai/gpt-oss-120b
# Optional: enables Google Fact Check Tools API alongside DuckDuckGo
# GOOGLE_FACTCHECK_API_KEY=your_google_factcheck_key
```

### 4. Launch the Application

```bash
python -m streamlit run factcheck_app.py
```

To run the full 30-message benchmark from the command line:

```bash
python factcheck_evaluator.py
```

## Security & Grounding Guardrails

* **Deterministic URL Verification:** Any URL cited by the synthesizer that does not exist verbatim in the retrieved evidence pool is stripped; verdicts lacking at least one verified URL are downgraded automatically to `Insufficient Evidence`.
* **Indirect Prompt-Injection Defense:** Suspicious override directives inside forwarded messages are flagged via regex pre-scanning and sandboxed inside `<UNTRUSTED_WHATSAPP_FORWARD>` XML delimiters.
* **Offline Demo Safety:** If internet access is unavailable during a live demo, the retriever switches to `offline_no_live_search` and enforces `Insufficient Evidence` instead of hallucinating external sources.

## Known Limitations & Roadmap

* **Image/Audio Forwards:** Currently verifies text forwards; future versions can integrate OCR (for screenshot hoaxes) and Whisper STT (for forwarded voice notes).
* **Snippet-Level Evidence:** Live search snippets may omit full article context; adding full-page fetching for top Tier-1 URLs can further improve nuanced verdict accuracy.
* **Future Enhancements:** Regional fact-check RSS caching, WhatsApp bot webhook integration, and NLI entailment scoring between claims and retrieved passages.

## Authors & Academic Context

Developed for the **Generative AI — Jury-Based Examination (Final Jury)** at **MIT ADT University, MIT School of Computing, Pune** (Department of Computer Science and Engineering).

* **Team:** `[Group No. / Member Names]`
* **Faculty Guide:** Prof. Chhaya Mhaske

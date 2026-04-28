# PEPG 2.0 — A Neuro-Symbolic RAG Framework for Graduate Program Evaluation

> **MSc Thesis | FURG PPGComp | Femi Samuel Adeola**  
> Supervisor: Prof. Eduardo N. Borges · Co-supervisor: Prof. Rodrigo De Bem  
> Centro de Ciências Computacionais (C3) / GInfo Lab — Universidade Federal do Rio Grande (FURG)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.25+-orange)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co)

---

## Overview

PEPG 2.0 is a **domain-agnostic institutional evaluation framework** that combines deterministic
rule-based scoring (Symbolic Engine) with retrieval-augmented generation (Neural Engine) to predict
and explain the official CAPES quadrennial grades assigned to Brazilian graduate programs.

The framework is built as a **Universal Institutional Software Framework** — the hybrid grading
engine is completely decoupled from any single knowledge area. Expanding to a new CAPES department
(Engineering, Law, Medicine, etc.) requires only a new YAML configuration file and a single
environment variable change. No Python code is modified.

A live Gradio dashboard is deployed on HuggingFace Spaces, featuring real-time KPI benchmark
comparison, neural semantic score transparency, holistic researcher profiling, and grade confidence indicators.

---

## Architecture

```text
Raw CAPES Data (Sucupira Platform)
             │
             ▼
  ┌──────────────────────┐
  │    ETL Pipeline      │  ← src/etl.py  +  area_profiles/{area}.yaml
  │    (Notebook 00)     │    Isolates all outputs per active area tag
  └──────┬───────────────┘    *Directly matches & enriches Qualis grades
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  ┌──────────────────┐          ┌────────────────────────┐
  │  Analytical CSVs │          │  FAISS Vector Index    │
  │  (×4, per area)  │          │  (Proposal Texts)      │
  │  Symbolic Data   │          │  Neural Data           │
  └────────┬─────────┘          └────────────┬───────────┘
           │                                 │
           ▼                                 ▼
  ┌──────────────────┐          ┌────────────────────────┐
  │ Symbolic Engine  │          │    Neural Engine       │
  │ 10 KPI Rules     │          │  RAG + LLM Reasoning   │
  │ symbolic.py      │          │  neural.py             │
  └────────┬─────────┘          └────────────┬───────────┘
           │                                 │
           └─────────────┬───────────────────┘
                         ▼
             ┌────────────────────────┐
             │  calculate_final_grade │  ← Thresholds loaded from YAML profile
             │  Hybrid Grade Engine   │
             └────────────┬───────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
     Predicted Grade (1–7)       Strategic Report
     + Confidence Indicator      + KPI Benchmark Panel
                                 + Semantic Score Panel
                                 + Top 15 Excellence Nucleus
                                 (Gradio Dashboard)
```

---

## Repository Structure

```text
project-root/
│
├── app.py                           ← Gradio dashboard (HuggingFace entry point)
├── src/
│   ├── config.py                    ← Profile loader + all shared dynamic constants
│   ├── prompts.py                   ← All LangChain prompt templates (EN + PT)
│   ├── etl.py                       ← Ingests data, maps Qualis grades, builds FAISS
│   ├── evaluator.py                 ← Benchmark and population evaluation runners
│   └── engine/
│       ├── neural.py                ← FAISS loader + document formatter
│       └── symbolic.py              ← KPI computation + grade engine rules
│
├── .env                             ← API keys + PEPG_AREA (never commit)
├── requirements.txt                 ← Production dependencies (HuggingFace Spaces)
├── requirements-dev.txt             ← Full development environment
│
├── area_profiles/                   ← "CAPES Rules Cartridges" — one file per department
│   ├── computacao.yaml              ← Active profile for Computing
│   └── engenharia_template.yaml     ← Ready-to-use template for Engineering
│
├── 00_run_etl.ipynb                 ← Step 0: ingest data, build FAISS index
├── 01_run_benchmark.ipynb           ← Controlled benchmark — N=150 fixed dataset
├── 02_run_population.ipynb          ← Population-scale evaluation — all programs
├── 03_run_analytics.ipynb           ← Prediction aggregation + population figures
├── 04_run_offline_calibration.ipynb ← Zero-cost deterministic engine calibrator
│
├── output/                          ← All generated files (area-tagged)
│   ├── db/knowledgebase_faiss_{area}/
│   ├── programs.csv
│   ├── ui_metadata_{area}.csv
│   ├── master_sucupira_index.csv
│   ├── analytical_docentes_{area}.csv
│   ├── analytical_discentes_{area}.csv
│   ├── analytical_producoes_{area}.csv
│   ├── analytical_participantes_{area}.csv
│   ├── figures/
│   └── computing_evaluation_benchmark/
│       ├── FULL_PREDICTIONS_cycle_{x}_{model}.csv
│       ├── BASELINE_PREDICTIONS_cycle_{x}_{model}.csv
│       └── SEMANTIC_BENCHMARK_{model}.csv
│
└── parsed/                          ← Raw data dropzone
    ├── programs.csv
    ├── qualis/                      ← Area-specific Qualis rankings (referenced by YAML)
    │   └── computacao/
    │       ├── qualis_periodicos_computacao_2013-16.csv
    │       ├── qualis_periodicos_computacao_2017-20.csv
    │       ├── qualis_periodicos_computacao_2021-24.csv
    │       └── qualis_eventos_computacao_2021.csv
    └── {year}/{program_id}/
        ├── docentes.csv
        ├── discentes.csv
        ├── producoes.csv
        ├── participantes_externos.csv
        └── proposta.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pepg2.git
cd pepg2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root. **This file must never be committed.**

```env
PEPG_AREA=computacao
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

`PEPG_AREA` tells the framework which YAML profile to load at startup. All output files
will be tagged with this area name, ensuring data from multiple departments can safely
coexist on the same machine.

`OPENROUTER_API_KEY` is used by Notebooks 01 and 02 for multi-model evaluation
(DeepSeek, GPT-4o, Gemini via OpenRouter).

`GOOGLE_API_KEY` is used exclusively by `app.py` for the live dashboard
(Gemini 2.5 Flash via the native Google API).

### 3. Prepare raw CAPES data

Organise the raw Sucupira exports under `parsed/` following this exact structure:

```text
parsed/programs.csv
parsed/qualis/computacao/qualis_periodicos_computacao_2021-24.csv
parsed/2017/12345/docentes.csv
parsed/2017/12345/discentes.csv
parsed/2017/12345/producoes.csv
parsed/2017/12345/participantes_externos.csv
parsed/2017/12345/proposta.txt
...
```

### 4. Run the ETL pipeline

```bash
jupyter notebook 00_run_etl.ipynb
```

*(Alternatively: `python -c "from src.etl import execute_pipeline; execute_pipeline()"`)*

This step is skip-safe. If the FAISS index and analytical CSVs for the active area already
exist on disk, the pipeline detects them and exits without reprocessing. To force a rebuild,
delete the `analytical_*.csv` files in the `output/` folder.

### 5. Launch the dashboard

```bash
python app.py
```

The dashboard header will display the active area profile name. To switch departments,
change `PEPG_AREA` in `.env` and restart.

---

## HuggingFace Spaces Deployment

PEPG 2.0 is deployed as an **instance-level deployment** — all rules and thresholds are
loaded once at startup from the active YAML profile. There are no area dropdowns in the UI.

**To deploy a new area instance:**

1. Create a new HuggingFace Space (or fork the existing one).
2. Upload the `output/` assets for the new area (analytical CSVs, FAISS index, UI metadata).
3. In the Space settings, add the following **Repository Secrets** (never commit these):
   - `PEPG_AREA` = `computacao` (or your area slug)
   - `GOOGLE_API_KEY` = your Google API key
   - `OPENROUTER_API_KEY` = your OpenRouter key (optional for dashboard-only deployment)
4. Push the code. The Space will boot and load the correct profile automatically.

Each department gets its own isolated Space instance. Data never mixes between areas.

---

## Execution Pipeline

| Step | Notebook | Purpose | API Cost |
|------|----------|---------|---------|
| 0 | `00_run_etl.ipynb` | Ingest raw data → maps Qualis → analytical CSVs + FAISS index | None |
| 1 | `01_run_benchmark.ipynb` | Controlled benchmark — 150-row fixed dataset, 3 LLMs | LLM calls |
| 2 | `02_run_population.ipynb` | Full population evaluation — all programs, 3 LLMs | LLM calls |
| 3 | `03_run_analytics.ipynb` | Aggregate predictions → generate all population figures | None |
| 4 | `04_run_offline_calibration.ipynb` | Test grade engine thresholds with zero API calls | None |

Each notebook declares its active `Target Area` at startup so accidental runs on the
wrong dataset are immediately visible before any API credits are consumed.

All notebooks are independently resumable. Existing outputs are never overwritten.

---

## KPI Framework — 10 Symbolic Indicators + 3 Semantic Scores

### Group A — Faculty Dimension

| # | KPI | Source | Measures |
|---|-----|--------|---------|
| 1 | `faculty_size` | analytical_docentes | Average number of faculty per year in the cycle |
| 2 | `impact_per_prof` | producoes + docentes | Total publications ÷ avg faculty — core productivity volume |
| 3 | `advisorship_variance` | analytical_discentes | Variance in students-per-advisor — workload equity indicator |

### Group B — Student Dimension

| # | KPI | Source | Measures |
|---|-----|--------|---------|
| 4 | `grad_efficiency` | analytical_discentes | Graduated ÷ (Graduated + Dropped) × 100 |
| 5 | `phd_ratio` | analytical_discentes | Doctoral students as % of all enrolled |
| 6 | `student_authored_pct` | analytical_producoes | Publications with at least one student co-author |

### Group C — Output Dimension

| # | KPI | Source | Measures |
|---|-----|--------|---------|
| 7 | `english_ratio` | producoes titles | % of publication titles in English — international output proxy |
| 8 | `qualis_top_pct` | producoes + Qualis CSV | % of classified peer-reviewed publications rated Qualis A (A1–A4) applying *Qualis Único* methodology |
| 9 | `conference_top_pct` | producoes + Qualis CSV | % of conferences fuzzy-matched to top Qualis events |

### Group D — Internationalisation

| # | KPI | Source | Measures |
|---|-----|--------|---------|
| 10 | `foreign_ratio` | analytical_participantes | External collaborators with confirmed foreign affiliation |

### Group E — Neural / Semantic Scores (LLM-Generated)

| Score | Measures | Scale |
|-------|---------|-------|
| `mission_score` | Clarity and strategic ambition of program mission and objectives | 1–5 |
| `social_score` | Documented social insertion, regional impact, industry partnerships | 1–5 |
| `planning_score` | Planning quality: measurable goals, self-assessment, accountability | 1–5 |

KPIs 1–10 are computed directly from analytical CSVs by the Symbolic Engine.
The three semantic scores are generated by the LLM from the program proposal text
and feed exclusively into the Grade 5/6 boundary refinement.

> **Note on Qualis matching:** The framework dynamically filters out non-classified publications
> (books, software, technical reports) from the denominator to ensure pure academic metrics
> strictly reflect valid, peer-reviewed venues.

---

## Grade Engine — `calculate_final_grade()`

All numeric thresholds are loaded from the active area YAML profile via `GRADE_THRESHOLDS`
in `config.py`. The same function evaluates any CAPES department without code changes.

```text
Grade 3 → Low productivity OR zero graduation efficiency OR no doctoral track + weak Qualis
Grade 4 → Baseline program — masters-only or doctoral track below threshold
Grade 5 → Doctoral training active + productivity above threshold, limited international reach
Grade 6 → Strong doctoral + BOTH English output ≥ 40% AND Qualis A (A1-A4) ≥ 28% (conjunction)
Grade 7 → Elite: high Qualis + high English + strong productivity + doctoral training
```

**Neuro refinement:** A program at Grade 5 with `sem_avg ≥ 4.0` AND at least a soft
international signal is promoted to Grade 6. A program at Grade 6 with `sem_avg ≤ 2.5`
is demoted to Grade 5. All other grade zones are determined purely by symbolic rules.

---

## Dashboard Features

The live dashboard provides the following outputs for each evaluated program:

**Quadrennial Cycle Evaluation tab:**
- **Strategic Narrative Report (Neural Engine):** Detailed SWOT analysis, recommendations, and evidence citations extracted via RAG.
- **Top 15 Holistic Excellence Nucleus (Symbolic Engine):** Ranks the top 15 researchers by Qualis A impact, providing a complete profile including their specific Qualis breakdown, total publication volume, internationalization (% English), and mentorship (% student co-authorship).
- **KPI Benchmark Panel:** The 10 symbolic KPIs compared against the area median with a colour-coded progress bar.
- **Neural Semantic Score Panel:** The three qualitative scores displayed with a Grade Confidence Indicator (Symbolic / Neuro-Promoted / Neuro-Demoted).
- **Predicted Grade (Hybrid Engine)** vs Official CAPES Grade side by side.

**Annual Progress Monitoring tab:**
- Single-year diagnostic report without grade prediction.
- Tracks faculty stability, student efficiency, and output quality across a specific audit year.

---

## Evaluation Results (DeepSeek — Computing Area, Population-Scale)

| Metric | Cycle 2 (2017–2020) | Cycle 3 (2021–2024) |
|--------|--------------------|--------------------|
| Total Programs | 76 | 90 |
| Exact Match | 52.6% | 45.6% |
| Tolerance ±1 | 82.9% | 74.4% |
| MAE | 0.711 | 0.878 |
| Grade 7 Exact | **100%** (8/8) | 62.5% (5/8) |
| Grade 6 Exact | 75.0% (3/4) | 75.0% (3/4) |
| Grade 4 Exact | 45.2% (14/31) | 45.9% (17/37) |
| Grade 3 Exact | 65.2% (15/23) | 55.2% (16/29) |

---

## Extending to a New Department

Deploying PEPG 2.0 for a new CAPES knowledge area requires **zero code changes**.

### Required inputs

| Input | Location | Description |
|-------|----------|-------------|
| Raw Sucupira CSVs | `parsed/{year}/{program_id}/` | Same folder structure as any existing area |
| `programs.csv` | `parsed/` | Master registry with program codes and modalities |
| `proposta.txt` per program | `parsed/{year}/{program_id}/` | Free-text proposal documents |
| Qualis CSV files | `parsed/qualis/{area_slug}/` | Area-specific journal and conference rankings from CAPES |
| `master_sucupira_index.csv` | `output/` | Official historical grades for validation |

### Steps

1. Copy `area_profiles/engenharia_template.yaml` and rename it to your area slug (e.g., `medicina.yaml`).
2. Fill in the Qualis file names, benchmark medians, and grade thresholds for your area.
   CAPES publishes area documents and benchmark statistics publicly — use these as the source.
3. Add your Qualis CSV files to `parsed/qualis/{area_slug}/`.
4. Set `PEPG_AREA={area_slug}` in `.env`.
5. Run `00_run_etl.ipynb` once to build the FAISS index and analytical CSVs for the new area.
   All output files will be tagged with the area slug (e.g., `analytical_docentes_medicina.csv`),
   so existing Computing data on the same machine is never overwritten.
6. Run the evaluation notebooks as normal. Everything else is identical.

---

## Citation

```bibtex
@mastersthesis{adeola2026pepg2,
  author    = {Femi Samuel Adeola},
  title     = {PEPG 2.0: A Universal Hybrid Neuro-Symbolic RAG Framework for CAPES Graduate Program Evaluation},
  school    = {Universidade Federal do Rio Grande (FURG)},
  year      = {2026},
  type      = {Dissertação de Mestrado},
  address   = {Rio Grande, RS, Brazil},
  note      = {Programa de Pós-Graduação em Computação (PPGComp)}
}
```

---

## License

This project is released for academic use. Data from the CAPES Sucupira Platform is subject
to its own terms of use. See `LICENSE` for details.

---

*Built with LangChain · FAISS · Gradio · HuggingFace Spaces · Google Gemini · DeepSeek · GPT-4o*

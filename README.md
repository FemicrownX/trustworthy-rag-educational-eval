# PEPG 2.0 — A Neuro-Symbolic RAG Framework for Graduate Program Evaluation

> **MSc Thesis · FURG PPGComp · Femi Samuel Adeola**
> Supervisor: Prof. Eduardo N. Borges · Co-supervisor: Prof. Rodrigo De Bem
> Centro de Ciências Computacionais (C3) / GInfo Lab — Universidade Federal do Rio Grande (FURG)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.25+-orange)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co)

---

## What is PEPG 2.0?

PEPG 2.0 is a **domain-agnostic institutional evaluation framework** that predicts and explains the official CAPES quadrennial grades assigned to Brazilian graduate programs. It combines two complementary engines:

- **Symbolic Engine** — 10 deterministic KPI rules computed directly from structured CAPES data exports.
- **Neural Engine** — Retrieval-Augmented Generation (RAG) over unstructured program proposal texts, using an LLM to produce qualitative semantic scores.

The framework is architecturally universal. Deploying it for a new CAPES knowledge area (Engineering, Medicine, Law, etc.) requires only a new YAML configuration file and one environment variable change — **no Python code is modified**.

A live Gradio dashboard deployed on HuggingFace Spaces provides real-time KPI benchmarking, semantic score transparency, holistic researcher profiling, and grade confidence indicators.

---

## Architecture

```
Raw CAPES Data (Sucupira Platform)
             │
             ▼
  ┌──────────────────────┐
  │    ETL Pipeline      │  ← src/etl.py  +  area_profiles/{area}.yaml
  │    (Notebook 00)     │    Isolates outputs per active area tag
  └──────┬───────────────┘    Directly matches & enriches Qualis grades
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
             │  calculate_final_grade │  ← Thresholds from YAML profile
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

```
project-root/
│
├── app.py                                ← Gradio dashboard (HuggingFace entry point)
│
├── src/
│   ├── config.py                         ← Profile loader + dynamic path discovery
│   ├── prompts.py                        ← LangChain prompt templates (EN + PT)
│   ├── etl.py                            ← Ingests data, maps Qualis grades, builds FAISS
│   ├── evaluator.py                      ← Benchmark and population evaluation runners
│   └── engine/
│       ├── neural.py                     ← FAISS loader + document formatter
│       └── symbolic.py                   ← KPI computation + grade engine rules
│
├── area_profiles/                        ← "CAPES Rules Cartridges" — one file per department
│   ├── computacao.yaml                   ← Active profile for Computing
│   └── engenharia_template.yaml          ← Ready-to-use template for Engineering
│
├── 00_run_etl.ipynb                      ← Step 0: ingest data, build FAISS index
├── 01_run_benchmark.ipynb                ← Controlled benchmark — N=150 fixed dataset
├── 02_run_population.ipynb               ← Population-scale evaluation — all programs
├── 03_run_analytics.ipynb                ← Prediction aggregation + population figures
├── 04_run_offline_calibration.ipynb      ← Zero-cost deterministic engine calibrator
│
├── output/                               ← All generated files (area-tagged)
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
├── programs/                             ← Raw data dropzone
│   ├── programs.csv
│   └── {PEPG_AREA}/                      ← Area-specific root (e.g., computacao)
│       ├── qualis/                       ← Flat folder; files are dynamically discovered
│       │   ├── qualis_periodicos_2013-16.csv
│       │   ├── qualis_periodicos_2017-20.csv
│       │   ├── qualis_periodicos_2021-24.csv
│       │   ├── qualis_eventos_2021.csv
│       │   └── qualis_eventos_2025.csv
│       └── {year}/{program_id}/
│           ├── docentes.csv
│           ├── discentes.csv
│           ├── producoes.csv
│           ├── participantes_externos.csv
│           └── proposta.txt
│
├── .env                                  ← API keys + PEPG_AREA (never commit)
├── requirements.txt                      ← Production dependencies (HuggingFace Spaces)
└── requirements-dev.txt                  ← Full development environment
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/pepg2.git
cd pepg2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root. **Never commit this file.**

```env
PEPG_AREA=computacao
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

| Variable | Purpose |
|---|---|
| `PEPG_AREA` | Which YAML profile to load at startup. All output files are tagged with this slug, so multiple areas can safely coexist on disk. |
| `OPENROUTER_API_KEY` | Used by Notebooks 01 and 02 for multi-model evaluation (DeepSeek, GPT-4o, Gemini via OpenRouter). |
| `GOOGLE_API_KEY` | Used exclusively by `app.py` for the live dashboard (Gemini 2.5 Flash via native Google API). |

### 3. Prepare raw CAPES data

Organise Sucupira exports under `programs/` following this exact structure:

```
programs/programs.csv
programs/computacao/qualis/qualis_periodicos_2021-24.csv
programs/computacao/2017/12345/docentes.csv
programs/computacao/2017/12345/discentes.csv
programs/computacao/2017/12345/producoes.csv
programs/computacao/2017/12345/participantes_externos.csv
programs/computacao/2017/12345/proposta.txt
```

> Qualis files are discovered dynamically via year/type keyword matching. Exact filenames are flexible as long as files are placed directly inside the `qualis/` directory.

### 4. Run the ETL pipeline

```bash
jupyter notebook 00_run_etl.ipynb
# or: python -c "from src.etl import execute_pipeline; execute_pipeline()"
```

The pipeline is **skip-safe** — if the FAISS index and analytical CSVs for the active area already exist, it exits without reprocessing. To force a full rebuild, delete the `analytical_*.csv` files from `output/`.

### 5. Launch the dashboard

```bash
python app.py
```

The dashboard header shows the active area profile. To switch departments, update `PEPG_AREA` in `.env` and restart.

---

## Execution Pipeline

| Step | Notebook | Purpose | API Cost |
|---|---|---|---|
| 0 | `00_run_etl.ipynb` | Ingest raw data → map Qualis → analytical CSVs + FAISS index | None |
| 1 | `01_run_benchmark.ipynb` | Controlled benchmark — 150-row fixed dataset, 3 LLMs | LLM calls |
| 2 | `02_run_population.ipynb` | Full population evaluation — all programs, 3 LLMs | LLM calls |
| 3 | `03_run_analytics.ipynb` | Aggregate predictions → generate population figures | None |
| 4 | `04_run_offline_calibration.ipynb` | Test grade engine thresholds with zero API calls | None |

Each notebook declares its active `Target Area` at startup — accidental runs on the wrong dataset are immediately visible before any API credits are consumed. All notebooks are independently resumable; existing outputs are never overwritten.

---

## KPI Framework

### Group A — Faculty

| # | KPI | Source | Measures |
|---|---|---|---|
| 1 | `faculty_size` | analytical_docentes | Average number of faculty per year in the cycle |
| 2 | `impact_per_prof` | producoes + docentes | Total publications ÷ avg faculty — core productivity volume |
| 3 | `advisorship_variance` | analytical_discentes | Variance in students-per-advisor — workload equity indicator |

### Group B — Students

| # | KPI | Source | Measures |
|---|---|---|---|
| 4 | `grad_efficiency` | analytical_discentes | Graduated ÷ (Graduated + Dropped) × 100 |
| 5 | `phd_ratio` | analytical_discentes | Doctoral students as % of all enrolled |
| 6 | `student_authored_pct` | analytical_producoes | Publications with at least one student co-author |

### Group C — Output

| # | KPI | Source | Measures |
|---|---|---|---|
| 7 | `english_ratio` | producoes titles | % of publication titles in English — international output proxy |
| 8 | `qualis_top_pct` | producoes + Qualis CSV | % of peer-reviewed publications rated Qualis A (A1–A4), applying *Qualis Único* methodology |
| 9 | `conference_top_pct` | producoes + Qualis CSV | % of conferences fuzzy-matched to top Qualis events |

### Group D — Internationalisation

| # | KPI | Source | Measures |
|---|---|---|---|
| 10 | `foreign_ratio` | analytical_participantes | External collaborators with confirmed foreign affiliation |

### Group E — Neural / Semantic Scores (LLM-generated from proposal text)

| Score | Measures | Scale |
|---|---|---|
| `mission_score` | Clarity and strategic ambition of program mission and objectives | 1–5 |
| `social_score` | Documented social insertion, regional impact, industry partnerships | 1–5 |
| `planning_score` | Planning quality: measurable goals, self-assessment, accountability | 1–5 |

KPIs 1–10 are computed deterministically by the Symbolic Engine. The three semantic scores feed exclusively into the Grade 5/6 boundary refinement step.

> **Qualis matching note:** The framework dynamically filters out non-classified publications (books, software, technical reports) from the denominator so that `qualis_top_pct` reflects only valid, peer-reviewed venues.

---

## Grade Engine

All numeric thresholds are loaded from the active YAML profile. The same function evaluates any CAPES department without code changes.

```
Grade 3 → Low productivity OR zero graduation efficiency OR no doctoral track + weak Qualis
Grade 4 → Baseline program — masters-only or doctoral track below threshold
Grade 5 → Doctoral training active + productivity above threshold, limited international reach
Grade 6 → Strong doctoral + BOTH English output ≥ 40% AND Qualis A ≥ 28% (conjunction)
Grade 7 → Elite: high Qualis + high English + strong productivity + doctoral training
```

**Neuro refinement:**
- Grade 5 with `sem_avg ≥ 4.0` AND at least a soft international signal → promoted to Grade 6.
- Grade 6 with `sem_avg ≤ 2.5` → demoted to Grade 5.
- All other grade zones are determined purely by symbolic rules.

---

## Dashboard Features

**Quadrennial Cycle Evaluation tab:**
- **Strategic Narrative Report** — Detailed SWOT analysis, recommendations, and evidence citations via RAG.
- **Top 15 Holistic Excellence Nucleus** — Ranked by Qualis A impact with full profiles: Qualis breakdown, publication volume, internationalisation (% English), mentorship (% student co-authorship).
- **KPI Benchmark Panel** — 10 symbolic KPIs compared against the area median with colour-coded progress bars.
- **Neural Semantic Score Panel** — Three qualitative scores with a Grade Confidence Indicator (Symbolic / Neuro-Promoted / Neuro-Demoted).
- **Predicted Grade** vs Official CAPES Grade side by side.

**Annual Progress Monitoring tab:**
- Single-year diagnostic without grade prediction.
- Tracks faculty stability, student efficiency, and output quality across a specific audit year.

---

## Evaluation Results

**DeepSeek — Computing area, population-scale**

| Metric | Cycle 2 (2017–2020) | Cycle 3 (2021–2024) |
|---|---|---|
| Total programs | 76 | 90 |
| Exact match | 52.6% | 45.6% |
| Tolerance ±1 | 82.9% | 74.4% |
| MAE | 0.711 | 0.878 |
| Grade 7 exact | **100%** (8/8) | 62.5% (5/8) |
| Grade 6 exact | 75.0% (3/4) | 75.0% (3/4) |
| Grade 4 exact | 45.2% (14/31) | 45.9% (17/37) |
| Grade 3 exact | 65.2% (15/23) | 55.2% (16/29) |

---

## Extending to a New Department

Zero code changes are required. Only the following inputs are needed:

| Input | Location | Description |
|---|---|---|
| Raw Sucupira CSVs | `programs/{area_slug}/{year}/{program_id}/` | Same folder structure as any existing area |
| `programs.csv` | `programs/` | Master registry with program codes and modalities |
| `proposta.txt` per program | `programs/{area_slug}/{year}/{program_id}/` | Free-text proposal documents |
| Qualis CSV files | `programs/{area_slug}/qualis/` | Area-specific journal and conference rankings from CAPES |
| `master_sucupira_index.csv` | `output/` | Official historical grades for validation |

**Steps:**

1. Copy `area_profiles/engenharia_template.yaml` and rename it to your area slug (e.g., `medicina.yaml`).
2. Fill in Qualis file names, benchmark medians, and grade thresholds. CAPES area documents and benchmark statistics are publicly available.
3. Add your Qualis CSV files to `programs/{area_slug}/qualis/`.
4. Set `PEPG_AREA={area_slug}` in `.env`.
5. Run `00_run_etl.ipynb` once. Output files are tagged with the area slug (e.g., `analytical_docentes_medicina.csv`) so existing data on the same machine is never overwritten.
6. Run evaluation notebooks as normal.

---

## HuggingFace Spaces Deployment

PEPG 2.0 uses an **instance-level deployment model** — all rules and thresholds are loaded once at startup from the active YAML profile. There are no area dropdowns in the UI; each department gets its own isolated Space instance.

**To deploy a new area instance:**

1. Create a new HuggingFace Space (or fork the existing one).
2. Upload the `output/` assets for the new area (analytical CSVs, FAISS index, UI metadata).
3. Add the following **Repository Secrets** in the Space settings:
   - `PEPG_AREA` — your area slug (e.g., `computacao`)
   - `GOOGLE_API_KEY` — your Google API key
   - `OPENROUTER_API_KEY` — your OpenRouter key (optional for dashboard-only deployment)
4. Push the code. The Space boots and loads the correct profile automatically.

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
pepg2-neuro-symbolic/
│
├── README.md                    ← Updated with full eval results
├── requirements.txt
├── computacao.yaml              ← The "rules cartridge" concept
│
├── src/
│   ├── config.py
│   ├── etl.py
│   ├── prompts.py
│   ├── engine/
│   │   ├── symbolic.py
│   │   └── neural.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── analytics.py
│   └── dashboard.py
│
├── notebooks/
│   ├── 00_ETL.ipynb
│   ├── 01_Controlled_Benchmark_150.ipynb    ← Conference paper evidence
│   ├── 02_Full_Population_Eval.ipynb        ← Thesis upgrade evidence
│   ├── 03_Annual_RAG.ipynb
│   ├── 04_Calibration.ipynb
│   └── 05_Analytics.ipynb
│
├── results/                                 ← Key: "Benchmark Results.csv" equivalent
│   ├── benchmark_150_deepseek.csv           ← Conference paper dataset
│   ├── CALIBRATED_EVALUATED1_*.csv          ← Full population results
│   └── ANNUAL_REPORTS_CACHE_*.csv
│
└── app.py
---

## License

This project is released for academic use. Data from the CAPES Sucupira Platform is subject to its own terms of use. See `LICENSE` for details.

---

*Built with LangChain · FAISS · Gradio · HuggingFace Spaces · Google Gemini · DeepSeek · GPT-4o*

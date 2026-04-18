# PEPG 2.0 — Evaluation Framework
### MSc Thesis | FURG PPGComp | Femi Samuel Adeola

---

## Overview

This framework evaluates the **PEPG 2.0 Hybrid Neuro-Symbolic RAG** system
for predicting and analysing CAPES graduate program outcomes.
It is organised into four self-contained execution notebooks sharing one central configuration engine.

---

## Repository Structure

```text
project-root/
│
├── config.py                     ← Shared config (paths, models, prompts, helpers, Qualis loader)
├── constant.py                   ← Shared constants and dataset mappings
├── .env                          ← API keys (never commit this file)
│
├── qualis_periodicos_computacao_2013-16.csv  ← Qualis journal rankings Cycle 2
├── qualis_periodicos_computacao_2017-20.csv  ← Qualis journal rankings Cycle 3
├── qualis_periodicos_computacao_2021-24.csv  ← Qualis journal rankings (Future/Reference)
├── qualis_eventos_computacao_2021.csv        ← Conference rankings (reference only)
├── qualis_eventos_computacao_2025.csv        ← Conference rankings (reference only)
│
├── 00_ETL_Pipeline.ipynb                ← Step 0: ingest raw data, build FAISS index
├── 01_Benchmark_N150.ipynb              ← Controlled benchmark — N=150 fixed dataset
├── 02_Full_Computing_Evaluation.ipynb   ← Population-scale execution — all Computing programs
├── 03_Population_Analytics.ipynb        ← Scoring aggregation + all population figures
├── 04_Offline_Engine_Calibrator.ipynb   ← Deterministic 10-KPI mathematical rules tester
├── PEPG2_Dashboard.py                   ← Gradio dashboard (run standalone)
│
├── output/
│   ├── db/knowledgebase_faiss/          ← Vector database
│   ├── programs.csv
│   ├── ui_metadata.csv
│   ├── master_sucupira_index.csv
│   ├── Master_Evaluation_Dataset_150.csv
│   ├── analytical_docentes.csv
│   ├── analytical_discentes.csv
│   ├── analytical_producoes.csv
│   ├── analytical_participantes.csv
│   │
│   └── computing_evaluation_benchmark/  ← Isolated population execution data
│       ├── figures/
│       ├── FULL_PREDICTIONS_cycle_x_{model}.csv
│       ├── BASELINE_PREDICTIONS_cycle_x_{model}.csv
│       └── POPULATION_EVALUATION_SUMMARY.csv
│
└── parsed/
    ├── programs.csv
    └── {year}/{program_id}/
        ├── docentes.csv
        ├── discentes.csv
        ├── producoes.csv
        ├── participantes_externos.csv
        └── proposta.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-openai
pip install langchain-google-genai langchain-huggingface
pip install faiss-cpu sentence-transformers
pip install pandas numpy matplotlib seaborn scikit-learn networkx
pip install bert-score rouge-score
pip install python-dotenv tqdm gradio
pip install google-generativeai
```

### 2. Create your `.env` file

```text
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

Place this file in the project root (same folder as `config.py`).

- `OPENROUTER_API_KEY` — used by Notebooks 01 and 02 for multi-model evaluation
- `GOOGLE_API_KEY` — used only by `PEPG2_Dashboard.py` (Gemini direct API)

### 3. Verify the raw data is present

```text
parsed/programs.csv
parsed/{year}/{program_id}/docentes.csv   (etc.)
```

If the FAISS index already exists from a previous run, Notebook 00 will skip the build step automatically.

### 4. Run the dashboard

```bash
python PEPG2_Dashboard.py
```

A public Gradio link will be printed to the terminal (`share=True` is enabled by default).

---

## Symbolic KPIs

The prediction engine computes these KPIs directly from the analytical CSVs for every program:

| KPI | Source | Notes |
|-----|--------|-------|
| `impact_per_prof` | producoes / docentes | Publications per faculty member |
| `grad_efficiency` | discentes | Graduated / (Graduated + Dropped + Abandoned) |
| `phd_ratio` | discentes | Doctoral students as % of all enrolled |
| `english_ratio` | producoes titles | English stopword heuristic |
| `qualis_top_pct` | producoes + Qualis CSV | % of journal pubs rated A1 or A2 — journals only |
| `foreign_ratio` | participantes.estrangeiro | Confirmed foreign participants as % of total |

*Note on Qualis:* Classification applies to journal publications only. Conference venues are excluded because free-text naming in Sucupira yields only 9.4% exact match coverage — currently insufficient for a reliable standalone KPI.

---

## Handling Existing Outputs & Pipeline Execution

If you already have output files from previous runs, do not move or rename them. The skip logic in every notebook checks whether each file exists before running — existing results are protected automatically. To regenerate specific figures while keeping everything else, set `REGENERATE_FIGURES = True` at the top of the relevant notebook. To force a full re-run of any inference step, delete only the specific CSV you want to regenerate.

| Step | File | What it does |
|------|------|--------------|
| **0** | `00_ETL_Pipeline.ipynb` | Ingests `parsed/`, builds 4 analytical CSVs, FAISS index, and `ui_metadata.csv`. **Run once before anything else.** |
| **1** | `01_Benchmark_N150.ipynb` | Runs inference + scoring on the 150-row controlled dataset. Generates laboratory visual proofs (Figs 1–8). |
| **2** | `02_Full_Computing_Evaluation.ipynb` | The nationwide execution engine. Generates predictions for all CAPES Computing programs across 3 LLMs. |
| **3** | `03_Population_Analytics.ipynb` | Aggregates all population predictions into a master summary and generates macro-level visual proofs (e.g., Grade 7 Ceiling). |
| **4** | `04_Offline_Engine_Calibrator.ipynb` | A zero-cost sandbox to test deterministic `config.py` mathematical thresholds against raw CAPES data without triggering LLM APIs. |

Each notebook is **independent after Step 0**. You can run just Notebook 03 to regenerate figures if the CSV outputs from Notebook 02 already exist on disk.

---

## Skip Logic

Every output-generating block checks whether its output file already exists before running. This means:

- **Existing results are never overwritten** by accident.
- **Interrupted runs resume** from the last saved program.
- The framework is safe to run on a machine that already has partial results.

To force regeneration of all figures, set at the top of any notebook:

```python
REGENERATE_FIGURES = True
```

---

## Models

Three models are evaluated in parallel, configured in `config.py`:

```python
MODELS = [
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "google/gemini-2.5-flash",
]
```

To add or remove a model, edit this list once — all notebooks read from it.

---

## Evaluation Cycles

| Cycle | Years | Grade Reference Year |
|-------|-------|---------------------|
| Cycle 2 | 2013–2016 | 2017 |
| Cycle 3 | 2017–2020 | 2021 |

---

## Key Output Files

| File | Produced by | Description |
|------|-------------|-------------|
| `FINAL_{model}_RESULTS_150.csv` | Notebook 01 | Raw LLM responses on 150-row benchmark |
| `FINAL_BENCHMARK_SUMMARY.csv` | Notebook 01 | Aggregated benchmark scores (N=150) |
| `SEMANTIC_BENCHMARK_{model}.csv` | Notebook 01 | Groundedness scores |
| `SYMBOLIC_BENCHMARK_cycle_2/3_{model}.csv` | Notebook 01 | KPI math accuracy vs LLM baseline |
| `FULL_PREDICTIONS_cycle_2/3_{model}.csv` | Notebook 02 | Neuro-symbolic grade predictions |
| `BASELINE_PREDICTIONS_cycle_2/3_{model}.csv` | Notebook 02 | Baseline LLM predictions |
| `POPULATION_EVALUATION_SUMMARY.csv` | Notebook 03 | Unified accuracy table — all models + cycles |

---

## Architecture

PEPG 2.0 combines two engines:

**Symbolic Engine** — Pandas-computed KPIs from CAPES analytical CSVs:
- Faculty productivity (publications per professor)
- Student graduation efficiency
- PhD training ratio
- International output ratio (English titles)

**Neural Engine** — FAISS vector retrieval over parsed proposal documents:
- `fetch_k=200000` required when filtering by `program_id` to avoid silent data loss.
- Embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Both engines feed into `calculate_final_grade()` in `config.py`, which applies the deterministic rule engine to produce a predicted CAPES grade (1–7).

---

## Known Limitations

1. **Grade 7 ceiling** — Missing Qualis journal/conference ranking KPIs cause systematic under-prediction of Grade 7 programs. Confirmed across all three models.
   *Fix: Advanced Qualis integration (Phase 2 remaining work).*
2. **5 top programs missing parsed folders** — UFRGS, UNICAMP, USP, UFPE, UFF have no proposal TXT in the parsed directory. This is a data availability gap, not a system bug.
3. **FAISS cross-domain scope** — Current index covers Computing programs only. Expansion to Engineering and one additional department planned for Phase 2.

---

## Contact

**Author:** Femi Samuel Adeola  
**Supervisor:** Prof. Eduardo N. Borges  
**Co-supervisor:** Prof. Rodrigo De Bem  
**Institution:** FURG PPGComp — Centro de Ciências Computacionais (C3) / GInfo Lab

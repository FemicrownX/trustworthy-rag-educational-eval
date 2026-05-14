# PEPG 2.0 — Neuro-Symbolic RAG Framework for Graduate Program Evaluation

> **MSc Thesis · FURG PPGComp · Femi Samuel Adeola**  
> Supervisor: Prof. Eduardo N. Borges · Co-supervisor: Prof. Rodrigo De Bem  
> Centro de Ciências Computacionais (C3) / GInfo Lab — FURG, Brazil

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-blueviolet)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-4.25+-orange)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/FemicrownX/pepg2_evaluator_v2)

---

## What is PEPG 2.0?

A hybrid Neuro-Symbolic RAG framework that predicts and explains CAPES quadrennial grades for Brazilian graduate programs. It fuses two complementary engines:

- **Symbolic Engine** — Deterministic KPI rule engine computed from structured CAPES Sucupira data. Six gate KPIs (`impact_per_prof`, `grad_efficiency`, `phd_ratio`, `english_ratio`, `qualis_top_pct`, `foreign_ratio`) drive grade decisions (3–7).
- **Neural Engine** — RAG over unstructured program proposal texts (FAISS + multilingual embeddings). Operates exclusively at promotion boundary via semantic scoring.

Deploying to a new CAPES area requires only a new YAML configuration file.
---

## Architecture

```
Raw CAPES Sucupira Data
        │
        ▼
┌──────────────────────────────────┐
│  ETL Pipeline                    │
│  Qualis enrichment (86.3% ISSN)  │
│  FAISS index from proposal texts │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
Symbolic       Neural Engine
Engine         FAISS + LLM
(6 KPIs)       Semantic Scores
    │             │
    └──────┬──────┘
           ▼
    Hybrid Decision Layer
           │
    ┌──────┴──────┐
    ▼             ▼
Dashboard     Evaluation
(Gradio)      Notebooks (00–05)
```

---

## Evaluation Datasets

This project contains two evaluation phases:

| Phase | Dataset | Purpose |
|-------|---------|---------|
| **Conference paper** | `results/benchmark_150_deepseek.csv` | 150-case controlled benchmark (50 symbolic · 50 semantic · 50 predictive) |
| **Thesis (full evaluation)** | `results/CALIBRATED_EVALUATED1_*.csv` | Full population across 3 CAPES cycles, 3 LLMs |

---

## Key Results

### Conference Benchmark (N=150, DeepSeek)

| Metric | Baseline LLM | Neuro-Symbolic RAG |
|--------|-------------|-------------------|
| Symbolic Accuracy | 0% | **86%** |
| Prediction Accuracy (±1) | 46% | **84%** |
| Groundedness | — | **0.676** |
| Recall@10 (FAISS) | — | 0.560 |

### Full Population Evaluation (Computing Area)

| Metric | Cycle 1 | Cycle 2 | Cycle 3 |
|--------|---------|---------|---------|
| Programs (N) | 69 | 76 | 90 |
| Exact match | 44.9% | 50.0% | 44.4% |
| ±1 adjacent | ~94% | ~91% | ~91% |
| MAE | 0.623 | 0.671 | 0.667 |
| Grade 7 exact | 100% | 100% | — |

Results are consistent across all three LLMs (DeepSeek, Gemini 2.5 Flash, GPT-4o), confirming the symbolic engine is model-agnostic.

---

## Repository Structure

```
pepg2/
├── app.py                        ← Gradio dashboard entry point
├── src/
│   ├── config.py                 ← Profile loader + path discovery
│   ├── etl.py                    ← Data ingestion, Qualis mapping, FAISS build
│   ├── prompts.py                ← LangChain prompt templates (PT + EN)
│   ├── analytics.py              ← Confusion matrices + visualisations
│   └── engine/
│       ├── symbolic.py           ← KPI computation + grade rules
│       └── neural.py             ← FAISS loader + document formatter
├── area_profiles/
│   ├── computacao.yaml           ← Active profile (Computing area)
│   └── interdisciplinar.yaml     ← Template for new areas
├── notebooks/
│   ├── 00_ETL.ipynb
│   ├── 01_Benchmark_150.ipynb    ← Conference paper evidence
│   ├── 02_Full_Population.ipynb  ← Thesis full evaluation
│   ├── 03_Annual_RAG.ipynb
│   ├── 04_Calibration.ipynb
│   └── 05_Analytics.ipynb
├── results/                      ← Evaluation CSVs (both phases)
├── output/figures/               ← Confusion matrices, elite plots (300 DPI)
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
git clone https://github.com/FemicrownX/PEPG-2.0.git
cd PEPG-2.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Create `.env` (never commit):
```env
PEPG_AREA="computacao"
OPENROUTER_API_KEY="your_key"
GOOGLE_API_KEY="your_key"
HF_TOKEN="your_hf_token"
```

Run in order:
```bash
jupyter notebook notebooks/00_ETL.ipynb       # Build FAISS index
jupyter notebook notebooks/02_Full_Population.ipynb  # Run evaluation
python app.py                                  # Launch dashboard
```

---

## Deploying to a New CAPES Area

1. Create `area_profiles/<area_name>.yaml` following `computacao.yaml`
2. Set `PEPG_AREA="<area_name>"` in `.env`
3. Place Sucupira CSVs in `data/` and run Notebook 00

## Citation

```bibtex
@mastersthesis{adeola2026pepg2,
  author  = {Femi Samuel Adeola},
  title   = {An AI-Powered Predictive and Analytical System Utilizing LLMs and RAG
             for Evaluating Graduate Programs},
  school  = {Universidade Federal do Rio Grande (FURG)},
  year    = {2026},
  type    = {Dissertação de Mestrado},
  address = {Rio Grande, RS, Brazil},
  note    = {PPGComp — GInfo Lab / C3}
}
```

---

## License

Academic use only. CAPES Sucupira data is subject to its own terms of use. See `LICENSE` for details.

# PEPG 2.0 — AI-Powered Predictive and Analytical System for Graduate Program Evaluation

### A Hybrid Neuro-Symbolic RAG Architecture for Trustworthy Educational Policy Decision-Support

> **Research Status: Active — M.Sc. Computer Engineering**
>
> This repository documents the architectural framework, methodology, evaluation protocols, and empirical results of my M.Sc. research at the **Federal University of Rio Grande (FURG)**, PPGComp.
>
> **Supervisors:** Prof. Eduardo N. Borges | Prof. Rodrigo De Bem
>
> *Due to LGPD data privacy protocols and active thesis development, raw data access is restricted. Code access may be granted to academic supervisors and collaborators upon request.*

---

## Project Overview

The Coordination for the Improvement of Higher Education Personnel (CAPES) evaluates over 7,000 graduate programs every four years — a complex, largely manual process that lacks real-time analytical tools for program coordinators and policymakers.

Standard LLMs cannot solve this problem alone. Without access to local institutional data, they hallucinate performance numbers and produce unverifiable outputs. RAG provides contextual grounding, but semantic retrieval alone lacks the deterministic logic needed for institutional accountability.

**PEPG 2.0** addresses this with a **Hybrid Neuro-Symbolic RAG architecture** combining two reasoning paths:

- A **Symbolic Engine** that computes KPIs directly from structured Sucupira CSV data — the LLM never guesses numbers.
- A **Neural RAG Engine** that retrieves and interprets unstructured program proposal documents via FAISS vector search.

Both paths feed into a single LLM inference step that produces a formal CAPES-style audit report with a predicted program grade.

---

## Architecture

```
+------------------------------+      +------------------------------+
|  SYMBOLIC ENGINE             |      |  NEURAL RAG ENGINE           |
|  (Zero Hallucination)        |      |  (Semantic Understanding)    |
|                              |      |                              |
|  docentes.csv       --+      |      |  proposta.txt      --+       |
|  discentes.csv      --+      |      |  Chunking (1000/100) |       |
|  producoes.csv      --+--->  |      |  paraphrase-multi  --+--->  |
|  participantes.csv  --+      |      |  FAISS Index       --+       |
|                              |      |                              |
|  8 KPIs via Pandas           |      |  Retrieval k=10              |
|  Symbolic Accuracy: 86%      |      |  Recall@10 = 0.818           |
|                              |      |  Groundedness  = 0.68        |
+-------------------------------+      +------------------------------+
               |                                     |
               +--------------------+----------------+
                                    |
               +--------------------v----------------+
               |  LLM GENERATOR                      |
               |  DeepSeek / Gemini 2.5 / GPT-4o     |
               |                                     |
               |  Formal CAPES Audit Report          |
               |  KPI Dashboard                      |
               |  Predicted Grade (3-7)              |
               +-------------------------------------+
```

### Symbolic KPIs

Eight KPIs are computed from the Sucupira analytical CSVs. Six are active in the grade prediction rule engine:

| KPI | Description | CAPES Pillar |
|---|---|---|
| `impact_per_prof` | Publications per faculty member | Producao Intelectual |
| `grad_efficiency` | Graduated / (Graduated + Dropped + Abandoned) | Corpo Discente |
| `phd_ratio` | Doctoral students as % of all enrolled | Corpo Discente |
| `english_ratio` | % of publication titles in English (proxy) | Insercao Social |
| `qualis_top_pct` | % of journal publications rated A1 or A2 | Producao Intelectual |
| `foreign_ratio` | Confirmed foreign participants as % of total | Insercao Social |

Note on Qualis: classification applies to journal publications only (86.3% ISSN match rate). Conference venues are excluded due to naming inconsistency in Sucupira source data (9.4% exact match). This limitation is acknowledged in the thesis methodology.

### Neural RAG Engine

| Component | Detail |
|---|---|
| Embedding model | paraphrase-multilingual-mpnet-base-v2 — 768-dim, 50+ languages |
| Chunking | RecursiveTextSplitter — chunk_size=1000, overlap=100 |
| Vector DB | FAISS |
| Retrieval | k=10 with fetch_k=200000 for program_id-filtered queries |

---

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| LLM Orchestration | LangChain |
| Vector Database | FAISS |
| Embeddings | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| LLMs Evaluated | DeepSeek · Google Gemini 2.5 Flash · OpenAI GPT-4o |
| Dashboard | Gradio (bilingual EN/PT) |
| Data Sources | CAPES Open Data Portal + Sucupira Platform |
| Data Coverage | Computing programs — 2013 to 2024 |

---

## Evaluation Framework

The system is evaluated across three dimensions — Retrieval, Generation, and Groundedness — using a 150-row human-verified benchmark dataset with a full RAG vs. No-RAG baseline comparison.

### Benchmark Dataset

| Property | Detail |
|---|---|
| Total rows | 150 (N=50 per task type) |
| Task types | Symbolic · Prediction · Semantic |
| Languages | Bilingual — Portuguese and English |
| Sampling | random.seed(42), replace=True |
| Gold standard | Human-verified Reference_Context column |
| Baseline condition | Same 150 questions — no RAG, no local context |

### Task Types

| Task | Description | Scorer |
|---|---|---|
| Symbolic | Exact KPI extraction from Sucupira CSVs | Word-boundary regex — zero tolerance |
| Prediction | CAPES grade inference (3-7) from KPIs | Adjacent +/-1 accuracy |
| Semantic | Mission/objectives from proposta.txt | BERTScore, ROUGE-L, Recall@k, Groundedness |

### Retrieval Metrics

All three models share identical retrieval scores — FAISS retrieval is model-independent.

| Metric | Result |
|---|---|
| Context Recall@10 | 0.818 (after fetch_k fix) |
| Context Precision | 0.169 |
| Context F1@10 | 0.258 |

### Generation Metrics

| Metric | Best Result |
|---|---|
| Symbolic Accuracy | 86% all models (RAG) vs 0% baseline |
| Prediction Accuracy +/-1 | 84% DeepSeek RAG vs 46% baseline |
| BERTScore (F1) | 0.753 — DeepSeek |
| ROUGE-L | 0.381 — GPT-4o |
| Groundedness | 0.676 — DeepSeek |

---

## Key Results

### Cross-Model Comparison — RAG Condition (N=150)

| Metric | DeepSeek | Gemini 2.5 | GPT-4o | Note |
|---|---|---|---|---|
| Symbolic Accuracy (%) | 86.00 | 86.00 | 86.00 | Equal — data gap ceiling |
| Prediction Accuracy (%) | **84.00** | 78.00 | 70.00 | DeepSeek leads |
| BERTScore (F1) | **0.753** | 0.739 | 0.749 | DeepSeek leads |
| ROUGE-L | 0.348 | 0.360 | **0.381** | GPT-4o leads |
| Groundedness | **0.676** | 0.665 | 0.660 | DeepSeek most grounded |
| Grade 7 Accuracy | 30% | 30% | 0% | Ceiling — all models |

### RAG vs. No-RAG Baseline

| Task | RAG | Baseline | Delta |
|---|---|---|---|
| Symbolic Accuracy | 86% | 0% | +86pp |
| Prediction Accuracy (DeepSeek) | 84% | 46% | +38pp |
| Prediction Accuracy (Gemini/GPT-4o) | 78% / 70% | ~0% | — |

### Population-Scale Results (N=70/85 deduplicated, both cycles)

| Finding | Detail |
|---|---|
| Baseline upward bias | LLM predicts Grade 6/7 for 61% of programs vs true 17% without RAG |
| Cross-model robustness | DeepSeek, Gemini, GPT-4o produce near-identical symbolic predictions |
| Grade 7 ceiling | Confirmed across all three models — root cause is missing Qualis KPI, not model failure |
| Semantic abstention | 67-93% not-available responses are grounded abstentions, not hallucinations |

### Grade-Level Prediction Accuracy — DeepSeek RAG

| CAPES Grade | Accuracy |
|---|---|
| Grade 3 | 100% |
| Grade 4 | 90% |
| Grade 5 | 100% |
| Grade 6 | 100% |
| Grade 7 | 30% — KPI ceiling |

### BERTScore vs. Groundedness Correlation

| Model | Pearson r |
|---|---|
| DeepSeek | 0.263 |
| Gemini 2.5 | 0.270 |
| GPT-4o | **0.494** |

Positive correlation between semantic quality and retrieval anchoring confirms that grounded responses are also semantically faithful.

---

## Known Limitations

| Limitation | Detail | Plan |
|---|---|---|
| Grade 7 ceiling | Qualis now integrated — further validation needed | Re-evaluate with new run |
| Computing dept. only | FAISS index covers Computing programs only | Add Engineering + 1 dept. |
| No human evaluation yet | Policy Usefulness rubric not completed | 3-5 program coordinators |
| English ratio proxy | Title-language heuristic, not guaranteed journal language | Complemented by Qualis and foreign_ratio |
| Conference Qualis excluded | 9.4% exact match due to free-text naming in Sucupira | Acknowledged in methodology |

---

## Repository Structure

```
pepg-2.0/
|
+-- config.py                    -- Shared engine: paths, KPIs, prompts, helpers, Qualis loader
|
+-- 00_ETL_Pipeline.ipynb        -- Ingests parsed/ data, builds analytical CSVs and FAISS index
+-- 01_Benchmark_N150.ipynb      -- Controlled benchmark on the 150-row fixed dataset
+-- 02_Population_Eval.ipynb     -- Full population evaluation across all Computing programs
+-- 03_Figures_and_Scoring.ipynb -- Master aggregator, accuracy scoring, and all figures
|
+-- PEPG2_Dashboard.py           -- Gradio dashboard (bilingual prescriptive interface)
|
+-- qualis/
|   +-- computacao/
|       +-- qualis_periodicos_computacao_2013-16.csv
|       +-- qualis_periodicos_computacao_2017-20.csv
|       +-- qualis_periodicos_computacao_2021-24.csv
|       +-- qualis_eventos_computacao_2021.csv   (reference only)
|       +-- qualis_eventos_computacao_2025.csv   (reference only)
|
+-- .env.example
+-- .gitignore
+-- requirements.txt
+-- README.md
```

Not in the repository (LGPD restricted or generated at runtime):
- `parsed/` — raw CAPES Sucupira data
- `AI-Powered PEPG 2.0 Evaluator/output/` — generated CSVs, figures, and FAISS index

---

## How to Run

### 1. Install dependencies

```bash
git clone https://github.com/FemicrownX/trustworthy-rag-educational-eval.git
cd trustworthy-rag-educational-eval
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in the project root (see `.env.example`):

```
OPENROUTER_API_KEY=your_openrouter_key
GOOGLE_API_KEY=your_gemini_key
```

`OPENROUTER_API_KEY` is used by the evaluation notebooks. `GOOGLE_API_KEY` is used only by the dashboard.

### 3. Run the ETL pipeline (first time only)

Open `00_ETL_Pipeline.ipynb` and run all cells. This builds the four analytical CSVs, the FAISS index, and the UI metadata file. All steps skip if outputs already exist.

### 4. Run evaluation notebooks in order

```
01_Benchmark_N150.ipynb      -- N=150 controlled benchmark
02_Population_Eval.ipynb     -- full population run
03_Figures_and_Scoring.ipynb -- scoring and figures
```

Each notebook skips outputs that already exist. Set `REGENERATE_FIGURES = True` at the top of any notebook to force figure regeneration.

### 5. Launch the dashboard

```bash
python PEPG2_Dashboard.py
```

Serves at `http://127.0.0.1:7860` and generates a public Gradio link. FAISS and PyTorch load on the first query, not at startup.

---

## Research Phases

**Phase 1 — Conference Paper (complete)**

Full population evaluation of all 98 Computing programs across two quadrennial cycles using three LLMs. Results include exact match, +/-1 tolerance, and MAE comparisons. Grade 7 ceiling documented as a KPI feature gap.

**Phase 2 — Thesis and Journal Paper (in progress)**

- Qualis A1/A2 journal integration — done
- Foreign participation KPI — done
- Interdisciplinary department expansion — data collection in progress
- Human evaluation with 3-5 program coordinators — planned
- Extended benchmark question types — planned

---

## Manuscripts

**Conference Paper**
> Adeola, F.S. et al. *Navigating the Tension: Balancing Accuracy and Interpretability in a Hybrid RAG Framework for Educational Policy — A CAPES Case Study.*

**Thesis**
> Adeola, F.S. *An AI-Powered Predictive and Analytical System Utilizing LLMs and RAG for Evaluating Graduate Programs.* M.Sc. Thesis, FURG PPGComp, 2026.

---

## References

- Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020.*
- Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop.*
- Es, S. et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *ArXiv.*
- Rudner, L.M. et al. (2006). An Application of Automated Essay Scoring Models. *Practical Assessment.*
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS.*
- Takami, K. (2024). Utilization of Japanese Public Educational Data by RAG for Policy Research.
- CAPES Open Data Portal: [dadosabertos.capes.gov.br](https://dadosabertos.capes.gov.br/)

---

## Contact

**Femi Samuel Adeola**
M.Sc. Candidate — Computer Engineering, FURG
Research Focus: Trustworthy AI · Neuro-Symbolic Systems · Educational Data Mining

[Email](mailto:femi@furg.br) | [LinkedIn](https://linkedin.com/in/femicrownx)

---

*© 2025-2026 Femi Samuel Adeola — Federal University of Rio Grande (FURG), PPGComp. All rights reserved. Data used under CAPES open data licence.*

# 🎓 PEPG 2.0 — AI-Powered Predictive & Analytical System for Graduate Program Evaluation

### *A Hybrid Neuro-Symbolic RAG Architecture for Trustworthy Educational Policy Decision-Support*

> **⚠️ Research Status: Active — M.Sc. Computer Engineering**
>
> This repository documents the **architectural framework, methodology, evaluation protocols, and empirical results** of my M.Sc. research at the **Federal University of Rio Grande (FURG)**, PPGComp — Graduate Program in Computer Engineering.
>
> **Supervisors:** Prof. Eduardo N. Borges | Prof. Rodrigo De Bem
>
> *Due to LGPD data privacy protocols and active thesis development, source code access is currently restricted. Access may be granted to academic supervisors and collaborators upon request.*

---

## 🔗 Live System Demo

The PEPG 2.0 Evaluator is deployed as a bilingual (🇧🇷 PT / 🇺🇸 EN) Gradio dashboard with two evaluation modes:

| Tab | Function |
|---|---|
| **Annual Progress Monitoring** | Single-year diagnostic — select Year → Institution → Program → Generate Report |
| **Quadrennial Cycle Evaluation** | 4-year evolution analysis with AI-predicted CAPES grade (3–7) |

> 🔗 **[Launch Live System](https://31d070dd8f6f147069.gradio.live/)**

---

## 📖 Project Overview

In the sensitive domain of educational policy, **"black box" AI models present significant risks**. The Coordination for the Improvement of Higher Education Personnel (CAPES) evaluates over 7,000 graduate programs every four years — a complex, largely manual process that lacks real-time analytical tools for program coordinators and policymakers.

Standard Large Language Models (LLMs) cannot solve this problem alone. Without access to local institutional data, they hallucinate key performance numbers and produce unverifiable outputs. While Retrieval-Augmented Generation (RAG) provides a bridge for contextual grounding, semantic retrieval alone lacks the deterministic logic required to ensure institutional accountability or absolute transparency.

**PEPG 2.0** addresses this by introducing a **Hybrid Neuro-Symbolic RAG architecture** that combines two reasoning paths:

- A **Symbolic Logic Engine** that computes KPIs deterministically from structured Sucupira CSV data — with zero tolerance for hallucination on quantitative facts.
- A **Semantic Neural Engine** that retrieves and interprets unstructured program proposal text (`proposta.txt`) via FAISS vector search — providing contextual grounding for qualitative analysis.

Both paths are fused into a single LLM inference step, producing a formal CAPES-style audit report with a predicted program grade.

---

## 🧠 The Dual-Brain Architecture

The central design insight is that educational program evaluation requires two fundamentally different types of reasoning — conflating them into a single generative model produces unreliable results.

```
┌──────────────────────────────┐      ┌──────────────────────────────┐
│  LEFT BRAIN — Symbolic       │      │  RIGHT BRAIN — Neural RAG    │
│  Engine  (Zero Hallucination)│      │  (Semantic Understanding)    │
│                              │      │                              │
│  docentes.csv       ──┐      │      │  proposta.txt      ──┐       │
│  discentes.csv      ──┤      │      │  Chunking (1000/100) │       │
│  producoes.csv      ──┼──▶   │      │  paraphrase-multi  ──┼──▶   │
│  participantes.csv  ──┘      │      │  FAISS Index       ──┘       │
│                              │      │                              │
│  6 KPIs — Pandas formulas    │      │  Semantic retrieval k=10     │
│  Symbolic Accuracy: 86%      │      │  Recall@10 = 0.56            │
│                              │      │  Groundedness  = 0.68        │
└──────────────┬───────────────┘      └──────────────┬───────────────┘
               │                                     │
               └─────────────────┬───────────────────┘
                                 ▼
               ┌─────────────────────────────────┐
               │  LLM GENERATOR                  │
               │  DeepSeek / Gemini 2.5 / GPT-4o │
               │                                 │
               │  Formal CAPES Audit Report      │
               │  KPI Dashboard                  │
               │  Predicted Grade (3–7)          │
               └─────────────────────────────────┘
```

### 🧮 Left Brain — Symbolic Logic Engine

| Component | Detail |
|---|---|
| **Goal** | Zero-tolerance accuracy on quantitative KPIs |
| **Method** | Pandas/Python deterministic formulas — the LLM never guesses numbers |
| **Input files** | `analytical_docentes.csv` · `analytical_discentes.csv` · `analytical_producoes.csv` · `analytical_participantes.csv` |
| **Output** | 7 pre-computed KPIs injected into the prompt as immutable arithmetic facts |

**The 6 KPIs and their CAPES pillar mapping:**

| KPI | CAPES Pillar |
|---|---|
| Faculty Stability Index | Corpo Docente |
| Student Success Rate | Corpo Discente |
| PhD Training Density | Corpo Discente |
| Total Intellectual Output | Produção Intelectual |
| Internationalization Ratio | Inserção Social |
| Collaboration Intensity | Inserção Social |

### 🎨 Right Brain — Semantic Neural Engine (RAG)

| Component | Detail |
|---|---|
| **Goal** | High interpretability — retrieve contextual evidence from program proposals |
| **Input** | `proposta.txt` — unstructured program proposal documents |
| **Embedding model** | `paraphrase-multilingual-mpnet-base-v2` (HuggingFace) — 768-dim, 50+ languages |
| **Chunking** | `RecursiveTextSplitter` — chunk_size=1000, overlap=100 |
| **Vector DB** | FAISS — millisecond-latency dense vector retrieval |
| **Retrieval depth** | k=10 — no `program_id` filter for Semantic tasks (100% coverage) |

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.x |
| **LLM Orchestration** | LangChain |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **LLMs Evaluated** | DeepSeek · Google Gemini 2.5 Flash · OpenAI GPT-4o |
| **Dashboard** | Gradio (bilingual EN/PT) |
| **Data Sources** | CAPES Open Data Portal API + Sucupira Platform |
| **Data Coverage** | Computing programs — 2013 to 2024 (longitudinal) |

---

## 📊 Evaluation Framework — RAG³

The system is evaluated using a three-dimensional framework covering Retrieval, Generation, and Groundedness — applied across a 150-row human-verified benchmark dataset with a full RAG vs. No-RAG baseline comparison.

### Benchmark Dataset

| Property | Detail |
|---|---|
| **Total rows** | 150 (N=50 per task type) |
| **Task types** | Symbolic · Prediction · Semantic |
| **Languages** | Bilingual — Portuguese and English |
| **Sampling** | `random.seed(42)`, `replace=True` |
| **Gold standard** | Human-verified `Reference_Context` column (manually extracted from `proposta.txt`) |
| **Baseline condition** | Same 150 questions — no RAG, no local context (LLM-only) |

**Three task types:**

| Task | Description | Scorer |
|---|---|---|
| **Symbolic** | Exact KPI count extraction from Sucupira CSVs | Word-boundary regex `\b N \b` — zero tolerance |
| **Prediction** | CAPES grade inference (3–7) from KPI indicators | Adjacent ±1 accuracy — Rudner et al. (2006) |
| **Semantic** | Mission/objectives extraction from `proposta.txt` | BERTScore · ROUGE-L · Recall@k · Groundedness |

### 5.1 Retrieval Metrics

| Metric | Definition | Result (all 3 models) |
|---|---|---|
| **Context Recall@k** | `overlap / gold_words` — proportion of gold Reference_Context retrieved by FAISS | 0.560 at k=10 |
| **Context Precision** | `overlap / ctx_words` — proportion of retrieved content that was relevant | 0.169 at k=10 |
| **Context F1@k** | Harmonic mean of Recall and Precision | 0.258 at k=10 |

> All retrieval scores are **identical across all three LLMs** — confirming FAISS retrieval is model-independent and that retrieval evaluation is fully isolated from generation quality.

### 5.2 Generation Metrics

| Metric | Definition | Best Result |
|---|---|---|
| **Symbolic Accuracy** | Zero-tolerance exact match via word-boundary regex — prevents 13≠130 or 13≠2013 | 86% all models (RAG) vs **0% baseline** |
| **Prediction Accuracy ±1** | Adjacent ±1 accuracy for CAPES grade inference — Rudner et al. (2006) | 84% DeepSeek RAG vs **46% baseline** |
| **BERTScore (F1)** | Token-level semantic similarity vs `Ground_Truth_Answer` — `bert-base-multilingual-cased` — Zhang et al. (2020) | 0.753 — DeepSeek |
| **ROUGE-L** | LCS-based lexical faithfulness — higher RAG score = better anchoring to official CAPES vocabulary — Lin (2004) | 0.381 — GPT-4o |

### 5.3 Groundedness

| Metric | Definition | Best Result |
|---|---|---|
| **Groundedness** | BERTScore F1 between `Generated_Report` and inference-time `Retrieved_Context` — soft hallucination resistance proxy — Es et al. (2023) RAGAS spirit | 0.676 — DeepSeek |

### 5.4 Human Metrics *(Planned — post *

**Policy Usefulness** — evaluated by 3–5 program coordinators using a 4-dimension rubric, each scored 1–5:

| Dimension | Scale |
|---|---|
| Factual Accuracy (Symbolic & Predictive Alignment) | 1–5 |
| Auditorial Tone & Institutional Style | 1–5 |
| Strategic Utility & Actionability | 1–5 |
| Explainability & Transparency | 1–5 |

---

## 📈 Key Results

### Cross-Model Comparison — RAG Condition (N=150)

| Metric | DeepSeek | Gemini 2.5 | GPT-4o | Note |
|---|---|---|---|---|
| Symbolic Accuracy (%) | **86.00** | **86.00** | **86.00** | Equal — data gap ceiling |
| Prediction Accuracy (%) | **84.00** 🏆 | 78.00 | 70.00 | DeepSeek leads |
| BERTScore (F1) | **0.753** 🏆 | 0.739 | 0.749 | DeepSeek leads |
| ROUGE-L | 0.348 | 0.360 | **0.381** 🏆 | GPT-4o leads |
| Context Recall@10 | 0.560 | 0.560 | 0.560 | Equal — FAISS independent |
| Context Precision | 0.169 | 0.169 | 0.169 | Equal — FAISS independent |
| Context F1@10 | 0.258 | 0.258 | 0.258 | Equal — FAISS independent |
| Groundedness | **0.676** 🏆 | 0.665 | 0.660 | DeepSeek most grounded |
| Grade 7 Accuracy | 30% ⚠️ | 30% ⚠️ | 0% ❌ | Ceiling — all models |

### RAG vs. No-RAG Baseline

| Task | RAG | Baseline | Delta |
|---|---|---|---|
| Symbolic Accuracy | 86% | 0% — all models refuse without local data | **+86pp** |
| Prediction Accuracy (DeepSeek) | 84% | 46% | **+38pp** |
| Prediction Accuracy (Gemini / GPT-4o) | 78% / 70% | ~0% — format non-compliance | — |

### Grade-Level Prediction Accuracy — DeepSeek RAG

| CAPES Grade | Accuracy |
|---|---|
| Grade 3 | 100% |
| Grade 4 | 90% |
| Grade 5 | 100% |
| Grade 6 | 100% |
| Grade 7 | **30%** ⚠️ |

> **Grade 7 Ceiling Finding:** All three models systematically underperform at Grade 7 — proven cross-model, not model-specific. Root cause: the 6 KPI features lack journal quality signals (Qualis rankings). This is an identified research gap and the primary target for improvement.

### Novel Finding — BERTScore vs. Groundedness Correlation

Pearson r analysis across 50 Semantic task rows reveals a positive correlation between semantic output quality and retrieval anchoring:

| Model | Pearson r |
|---|---|
| DeepSeek | 0.263 |
| Gemini 2.5 | 0.270 |
| GPT-4o | **0.494** |

> This confirms that grounded responses are also semantically faithful — the system is not merely retrieving relevant context, it is generating outputs that align with that context. GPT-4o exhibits the tightest coupling between quality and grounding.

---

## 🔍 Known Limitations

| Limitation | Detail | Plan |
|---|---|---|
| **Grade 7 Ceiling** | KPI features lack Qualis journal quality signals | Qualis integration - post |
| **Computing dept. only** | FAISS index currently covers Computing programs only | Index Engineering + 1 additional dept. — ETL requires zero rewrites |
| **No human evaluation yet** | Policy Usefulness metric not yet completed | 3–5 program coordinators |
| **Word-overlap retrieval proxy** | Recall/Precision use character-level overlap, not binary chunk relevance | Disclosed — consistent with RAG evaluation literature |
| **@k char-truncation approximation** | Recall@k simulated by truncating Retrieved_Context to k×1000 chars | Disclosed — consistent with chunk_size=1000 setting |

---

## 📚 Manuscripts in Preparation

**Paper 1**
> Adeola, F.S. et al. *"Navigating the Tension: Balancing Accuracy and Interpretability in Retrieval-Augmented Generation for Educational Policy"* — Research Paper

**Thesis**
> Adeola, F.S. *"A Framework for Trustworthy Policy Insights: Mitigating Hallucination in Retrieval-Augmented Generation for Brazilian Educational Data"* — M.Sc. Thesis, FURG PPGComp, 2026

---

## 🌍 Impact & Alignment

This project aligns with **UN Sustainable Development Goal 4 (Quality Education)** by providing policymakers with transparent, evidence-based tools for graduate program evaluation — moving from static annual snapshots to dynamic, evolutionary analysis grounded in official CAPES and Sucupira data.

- **Traceability** — every AI-generated insight is cited back to its source document (CSV row or `proposta.txt` passage)
- **Anti-hallucination** — the Symbolic Engine prevents quantitative hallucination; Groundedness measures qualitative anchoring
- **Scalability** — the ETL pipeline is department-agnostic; adding Engineering or Medicine requires zero system rewrites
- **Bilingual** — full Portuguese and English support via multilingual embeddings and bilingual prompting
- **Key contribution** — delivers significant performance gains over LLM-only models through a hallucination-resistant, traceable, and scalable dashboard system

---

## 🗂️ Repository Structure

```text
pepg-2.0/
│
├── etl/                              # Data ingestion and preprocessing pipeline
│   ├── data_ingestion.py             # CAPES API retrieval and CSV parsing
│   ├── text_extraction.py            # proposta.txt parsing and cleaning
│   └── chunking.py                   # RecursiveTextSplitter (chunk=1000, overlap=100)
│
├── embeddings/                       # Vector index creation
│   ├── embed_proposals.py            # paraphrase-multilingual-mpnet-base-v2
│   └── faiss_index/                  # Persisted FAISS index files [RESTRICTED]
│
├── evaluation/                       # RAG³ evaluation framework & dataset generation
│   ├── build_dataset.py              # Generates 150-row Master benchmark dataset
│   ├── run_inference.py              # Batch LLM inference (Symbolic + Semantic routing)
│   ├── scoring.py                    # Orchestrator — Symbolic, Prediction, Semantic scoring
│   ├── retrieval_metrics.py          # Context Recall@k, Precision, F1
│   └── generation_metrics.py         # BERTScore, ROUGE-L, Groundedness
│
├── dashboard/                        # Gradio bilingual prescriptive interface
│   └── app.py                        # Live RAG + Symbolic engine execution & UI rendering
│
├── results/                          # Benchmark outputs and visualisation scripts
│   ├── generate_etl_figures.py       # ETL and conceptual thesis diagrams
│   ├── generate_benchmark_figures.py # Final model benchmark result plots
│   ├── Fig1_RAG_vs_Baseline.png
│   ├── Fig2_Recall_at_K.png
│   ├── Fig3_Grade_Accuracy_Heatmap.png
│   ├── Fig4_Confusion_Matrix.png
│   ├── Fig5_Semantic_Radar.png
│   ├── Fig6_BERT_vs_Groundedness.png
│   ├── Fig7_Symbolic_Failures.png
│   └── Fig8_Full_Benchmark_Heatmap.png
│
├── .gitignore                        # LGPD privacy rules and environment exclusions
├── requirements.txt                  # Project dependencies
└── README.md
```

> **Note:** Files marked [RESTRICTED] contain identifiable institutional data protected under Brazil's LGPD data privacy law. Access available to academic supervisors upon request.

---

## 🚀 How to Run

### 1. Installation

```bash
git clone https://github.com/yourusername/pepg-2.0.git
cd pepg-2.0
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory (see `.env.example` for reference):

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 3. Launch the Dashboard

```bash
python dashboard/app.py
```

The system will initialise the FAISS vector store, load the LLM, and serve a local Gradio interface at `http://127.0.0.1:7860`.

### 4. Reproduce the Benchmark (RAG³ Evaluation)

Run each step in sequence to reproduce all thesis metrics from scratch:

```bash
# Step A — Generate the 150-row Master Dataset
python evaluation/build_dataset.py

# Step B — Run LLM inference (DeepSeek, Gemini 2.5, GPT-4o)
python evaluation/run_inference.py

# Step C — Score all metrics (BERTScore, ROUGE-L, Recall@k, Groundedness)
python evaluation/scoring.py

# Step D — Generate all 8 thesis figures
python results/generate_benchmark_figures.py
```

---

## 📋 Key References

- Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020.*
- Lin, C.Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop.*
- Es, S. et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *ArXiv.*
- Rudner, L.M. et al. (2006). An Application of Automated Essay Scoring Models. *Practical Assessment.*
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS.*
- Takami, K. (2024). Utilization of Japanese Public Educational Data by RAG for Policy Research.
- CAPES Open Data Portal: [dadosabertos.capes.gov.br](https://dadosabertos.capes.gov.br/)

---

## 📬 Contact

**Femi Samuel Adeola**
M.Sc. Candidate — Computer Engineering, FURG
Research Focus: Trustworthy AI · Neuro-Symbolic Systems · Educational Data Mining

[📧 Email](mailto:femi@furg.br) | [🔗 LinkedIn](https://linkedin.com/in/femicrownx)

---

*© 2025–2026 Femi Samuel Adeola — Federal University of Rio Grande (FURG), PPGComp. All rights reserved. Data used under CAPES open data licence.*

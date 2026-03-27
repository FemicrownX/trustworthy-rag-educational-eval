# 🎓 PEPG 2.0 — AI-Powered Predictive & Analytical System for Graduate Program Evaluation
### *A Hybrid Neuro-Symbolic RAG Architecture for Trustworthy Educational Policy Decision-Support*

> **⚠️ Research Status: Active — M.Sc. Qualification Completed (March 2026)**
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

> 🔗 **[Launch Live System](https://gradio.live)** ← *Replace with your current Gradio URL*

---

## 📖 Project Overview

In the domain of Brazilian educational policy, **"black box" AI models are dangerous**. The Coordination for the Improvement of Higher Education Personnel (CAPES) evaluates over 7,000 graduate programs every four years — a complex, largely manual process that lacks real-time analytical tools for program coordinators and policymakers.

Standard Large Language Models (LLMs) cannot solve this problem alone. Without access to local institutional data, they hallucinate key performance numbers and produce unverifiable outputs. Retrieval-Augmented Generation (RAG) partially addresses this — but retrieval alone does not guarantee transparency or institutional accountability.

**PEPG 2.0** introduces a **Hybrid Neuro-Symbolic RAG architecture** that solves this by combining two reasoning paths:

- A **Symbolic Logic Engine** that computes KPIs deterministically from structured Sucupira CSV data — with zero tolerance for hallucination on quantitative facts.
- A **Semantic Neural Engine** that retrieves and interprets unstructured program proposal text (proposta.txt) via FAISS vector search — providing contextual grounding for qualitative analysis.

Both paths are fused into a single LLM inference step, producing a formal CAPES-style audit report with a predicted program grade.

---

## 🧠 The Dual-Brain Architecture

The central design insight is that educational program evaluation requires two fundamentally different types of reasoning — and conflating them into a single generative model produces unreliable results.

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│   LEFT BRAIN — Symbolic     │      │   RIGHT BRAIN — Neural RAG   │
│   Engine (Zero Hallucination)│      │   (Semantic Understanding)   │
│                             │      │                              │
│  docentes.csv       ──┐     │      │  proposta.txt   ──┐          │
│  discentes.csv      ──┤     │      │  Chunking (1000/100)         │
│  producoes.csv      ──┼──▶  │      │  paraphrase-multilingual ──▶ │
│  participantes.csv  ──┘     │      │  FAISS Index    ──┘          │
│                             │      │                              │
│  7 KPIs computed via        │      │  Semantic retrieval k=10     │
│  Pandas/Python formulas     │      │  Recall@10 = 0.56            │
│  Symbolic Accuracy: 86%     │      │  Groundedness = 0.68         │
└──────────────┬──────────────┘      └──────────────┬───────────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   LLM GENERATOR               │
              │   DeepSeek / Gemini 2.5 /     │
              │   GPT-4o                      │
              │                               │
              │   Formal CAPES Audit Report   │
              │   KPI Dashboard               │
              │   Predicted Grade (3–7)       │
              └───────────────────────────────┘
```

### 🧮 Left Brain — Symbolic Logic Engine

| Component | Detail |
|---|---|
| **Goal** | Zero-tolerance accuracy on quantitative KPIs |
| **Method** | Pandas/Python deterministic formulas — LLM never guesses numbers |
| **Input files** | `analytical_docentes.csv`, `analytical_discentes.csv`, `analytical_producoes.csv`, `analytical_participantes.csv` |
| **Output** | 7 pre-computed KPIs injected into the prompt as immutable arithmetic facts |

**The 7 KPIs computed:**

| KPI | Maps to CAPES Pillar |
|---|---|
| Faculty Stability Index | Corpo Docente |
| Student Success Rate | Corpo Discente |
| PhD Training Density | Corpo Discente |
| Total Intellectual Output | Produção Intelectual |
| Productivity per Professor | Produção Intelectual |
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
| **Retrieval depth** | k=10 (no program_id filter for Semantic tasks — 100% coverage) |

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.x |
| **LLM Orchestration** | LangChain |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **LLMs Evaluated** | DeepSeek, Google Gemini 2.5 Flash, OpenAI GPT-4o |
| **Dashboard** | Gradio (bilingual EN/PT) |
| **Data Sources** | CAPES Open Data Portal API + Sucupira Platform |
| **Data Coverage** | Computing programs — 2013 to 2024 (longitudinal) |

---

## 📊 Evaluation Framework — RAG³

The system is evaluated using a three-dimensional framework covering Retrieval, Generation, and Groundedness — applied across a 150-row human-verified benchmark dataset.

### The Benchmark Dataset

| Property | Detail |
|---|---|
| **Total rows** | 150 (N=50 per task type) |
| **Task types** | Symbolic · Prediction · Semantic |
| **Languages** | Bilingual — Portuguese and English |
| **Sampling** | `random.seed(42)`, `replace=True` |
| **Gold standard** | Human-verified `Reference_Context` column (manually extracted from proposta.txt) |
| **Baseline condition** | Same 150 questions — no RAG, no local context (LLM-only) |

**Three task types:**

| Task | Description | Scorer |
|---|---|---|
| **Symbolic** | Exact KPI count extraction from Sucupira CSVs | Word-boundary regex `\b N \b` — zero tolerance |
| **Prediction** | CAPES grade inference (3–7) from KPI indicators | Adjacent ±1 accuracy — Rudner et al. (2006) |
| **Semantic** | Mission/objectives extraction from proposta.txt | BERTScore · ROUGE-L · Recall@k · Groundedness |

### 5.1 Retrieval Metrics

| Metric | Definition | Result (all models) |
|---|---|---|
| **Context Recall@k** | `overlap / gold_words` — how much of the gold Reference_Context did FAISS retrieve | 0.560 at k=10 |
| **Context Precision** | `overlap / ctx_words` — of everything retrieved, how much was relevant | 0.169 at k=10 |
| **Context F1@k** | Harmonic mean of Recall and Precision | 0.258 at k=10 |

> All retrieval scores are **identical across all three LLMs** — confirming FAISS retrieval is model-independent and that retrieval evaluation is fully isolated from generation quality.

### 5.2 Generation Metrics

| Metric | Definition | Best Score |
|---|---|---|
| **Symbolic Accuracy** | Zero-tolerance exact match — word-boundary regex prevents 13≠130 | 86% — all 3 models (RAG) vs 0% baseline |
| **Prediction Accuracy ±1** | Adjacent accuracy for CAPES grade inference — Rudner et al. (2006) | 84% — DeepSeek RAG vs 46% baseline |
| **BERTScore (F1)** | Token-level semantic similarity vs Ground_Truth_Answer — `bert-base-multilingual-cased` — Zhang et al. (2020) | 0.753 — DeepSeek |
| **ROUGE-L** | LCS-based lexical faithfulness — higher RAG score = better anchoring to official CAPES vocabulary — Lin (2004) | 0.381 — GPT-4o |

### 5.3 Groundedness

| Metric | Definition | Best Score |
|---|---|---|
| **Groundedness** | BERTScore F1 between Generated_Report and inference-time Retrieved_Context — soft hallucination resistance proxy — Es et al. (2023) RAGAS spirit | 0.676 — DeepSeek |

### 5.4 Human Metrics (Planned — Before Final Defence)

**Policy Usefulness** — evaluated by program coordinators (3–5 domain experts) using a 4-dimension rubric, each scored 1–5:

- Factual Accuracy (Symbolic & Predictive Alignment)
- Auditorial Tone & Institutional Style
- Strategic Utility & Actionability
- Explainability & Transparency

---

## 📈 Key Results

### Cross-Model Comparison (RAG Condition, N=150)

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
| Symbolic Accuracy | 86% | 0% (all models refuse without data) | **+86pp** |
| Prediction Accuracy (DeepSeek) | 84% | 46% | **+38pp** |
| Prediction Accuracy (Gemini/GPT-4o) | 78% / 70% | ~0% (format non-compliance) | — |

### Grade-Level Prediction Accuracy (DeepSeek RAG)

| Grade | Accuracy |
|---|---|
| Grade 3 | 100% |
| Grade 4 | 90% |
| Grade 5 | 100% |
| Grade 6 | 100% |
| Grade 7 | **30%** ⚠️ |

> **Grade 7 Ceiling Finding:** All three models fail at Grade 7 — proven systematic across models, not model-specific. Root cause: the 7 KPI features lack journal quality signals (Qualis rankings). This is a publishable finding and the primary target for improvement before final defence.

### Novel Finding — BERTScore vs. Groundedness Correlation

Pearson r analysis across 50 Semantic task rows reveals a positive correlation between semantic quality and retrieval anchoring:

| Model | Pearson r |
|---|---|
| DeepSeek | 0.263 |
| Gemini 2.5 | 0.270 |
| GPT-4o | **0.494** |

> This confirms that grounded responses are also semantically faithful — the system is not merely retrieving relevant context, it is generating outputs that align with that context.

---

## 🔍 Known Limitations

| Limitation | Status |
|---|---|
| **Grade 7 Ceiling** | Qualis ranking integration planned before final defence |
| **Computing department only** | ETL pipeline fully adaptable — Engineering + 1 dept planned |
| **No human evaluation yet** | 3–5 program coordinators targeted before final defence |
| **Word-overlap retrieval proxy** | Acknowledged — consistent with RAG evaluation literature |
| **@k char-truncation approximation** | Disclosed — consistent with chunk_size=1000 setting |

---

## 📚 Manuscripts in Preparation

**Paper 1:**
> Adeola, F.S. et al. *"Navigating the Tension: Balancing Accuracy and Interpretability in Retrieval-Augmented Generation for Educational Policy"* — Manuscript in preparation, target: International AI Conference.

**Paper 2:**
> Adeola, F.S. *"A Framework for Trustworthy Policy Insights: Mitigating Hallucination in Retrieval-Augmented Generation for Brazilian Educational Data"* — Manuscript in preparation, based on M.Sc. research.

---

## 🌍 Impact & Alignment

This project aligns with **UN Sustainable Development Goal 4 (Quality Education)** by providing policymakers with transparent, evidence-based tools for graduate program evaluation. It allows program coordinators to move from static manual snapshots to dynamic, evolutionary analysis grounded in official CAPES and Sucupira data.

- **Traceability:** Every AI-generated insight is cited back to its source document — CSV row or proposta.txt passage.
- **Anti-hallucination:** The Symbolic Engine prevents quantitative hallucination. The Groundedness metric measures qualitative anchoring.
- **Scalability:** The ETL pipeline is department-agnostic — adding Engineering or Medicine requires zero system rewrites.
- **Bilingual:** Full Portuguese and English support via multilingual embeddings and bilingual prompting.

---

## 🗂️ Repository Structure

```
pepg-2.0/
│
├── etl/                        # Data ingestion and preprocessing pipeline
│   ├── data_ingestion.py       # CAPES API retrieval and CSV parsing
│   ├── text_extraction.py      # proposta.txt parsing and cleaning
│   └── chunking.py             # RecursiveTextSplitter (chunk=1000, overlap=100)
│
├── embeddings/                 # Vector index creation
│   ├── embed_proposals.py      # paraphrase-multilingual-mpnet-base-v2
│   └── faiss_index/            # Persisted FAISS index files [RESTRICTED]
│
├── symbolic_engine/            # Left Brain — deterministic KPI computation
│   ├── kpi_calculator.py       # 7 KPI formulas (Pandas)
│   └── logic_layer.py          # Prompt injection of immutable facts
│
├── rag_engine/                 # Right Brain — semantic retrieval
│   ├── retriever.py            # FAISS retrieval (k=10)
│   └── generator.py            # LLM inference (DeepSeek / Gemini / GPT-4o)
│
├── evaluation/                 # RAG³ evaluation framework
│   ├── benchmark_dataset.csv   # 150-row master dataset [RESTRICTED]
│   ├── scoring.py              # Symbolic, Prediction, Semantic scorers
│   ├── retrieval_metrics.py    # Recall@k, Precision, F1
│   └── generation_metrics.py   # BERTScore, ROUGE-L, Groundedness
│
├── dashboard/                  # Gradio bilingual interface
│   └── app.py                  # Annual Monitoring + Quadrennial Cycle tabs
│
├── results/                    # Benchmark outputs and figures
│   ├── Fig1_RAG_vs_Baseline.png
│   ├── Fig2_Recall_at_K.png
│   ├── Fig3_Grade_Accuracy_Heatmap.png
│   ├── Fig4_Confusion_Matrix.png
│   ├── Fig5_Semantic_Radar.png
│   ├── Fig6_BERT_vs_Groundedness.png
│   ├── Fig7_Symbolic_Failures.png
│   └── Fig8_Full_Benchmark_Heatmap.png
│
└── README.md
```

> **Note:** Files marked [RESTRICTED] contain identifiable institutional data protected under LGPD. Access available to academic supervisors upon request.

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

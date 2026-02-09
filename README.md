# 🎓 Trustworthy Neuro-Symbolic RAG for Educational Policy
### *Reconciling Interpretability and Accuracy in High-Stakes Decision Making*

> **⚠️ Research Status: Active / Confidential (M.Sc. Thesis)**
> This repository documents the **architectural framework, methodology, and evaluation protocols** for my ongoing research at the **Federal University of Rio Grande (FURG)**.
>
> *Due to data privacy protocols (LGPD) and active development, the source code is currently restricted. Access may be granted to academic supervisors upon request.*

---

## 📖 Project Overview
In the domain of educational policy, "black box" AI models are dangerous. When evaluating graduate programs (PPGs) for funding and accreditation, stakeholders need **mathematical precision** combined with **contextual nuance**.

This research introduces a **Hybrid Neuro-Symbolic RAG Architecture** that combines the rigid accuracy of structured data analysis with the flexible understanding of Large Language Models (LLMs). By grounding the model in official **CAPES** and **Sucupira** reports, we mitigate hallucinations and provide verifiable, evidence-based policy insights.

---

## 🧠 The Concept: A "Dual-Brain" Architecture

To solve the tension between **Creativity (Generative AI)** and **Factuality (Auditing)**, this system operates on two distinct logical planes:

### 🧮 1. The "Left Brain" (Symbolic Logic Layer)
* **Goal:** Zero-tolerance error for quantitative metrics.
* **Method:** Python & Pandas execute rigid mathematical formulas on raw CSV data. The LLM is **not allowed to guess** numbers; it is fed these pre-calculated immutable facts.
* **Key Performance Indicators (KPIs) Calculated:**
    * **Faculty Stability Index ($S$):** Measuring the reliance on permanent vs. temporary staff.
    * **Student Success Rate ($R_{success}$):** Quantifying graduation efficiency vs. dropout rates.
    * **PhD Training Density:** Determining the program's maturity ceiling.
    * **Internationalization Ratio:** Mapping the density of external collaboration networks.

### 🎨 2. The "Right Brain" (Semantic Neural Layer)
* **Goal:** High interpretability and contextual understanding.
* **Method:** A **Retrieval-Augmented Generation (RAG)** pipeline retrieves unstructured narrative text (program proposals) to explain *why* the numbers look the way they do.
* **Mechanism:** Uses **FAISS** for dense vector retrieval to find semantic evidence (e.g., "Social Insertion" strategies) buried in thousands of pages of PDF reports.

---

## 🛠️ Methodology & Technical Stack

### Data Pipeline (ETL)
* **Structured Sources:** CAPES Open Data (Discentes, Docentes, Produções, Participantes).
* **Unstructured Sources:** Sucupira Platform Reports (`proposta.txt`).
* **Preprocessing:** Recursive Text Splitting (Chunk Size: 2000 | Overlap: 200) to maintain narrative flow.

### The RAG Engine
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** **FAISS** (Facebook AI Similarity Search) optimized for millisecond-latency retrieval.
* **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (HuggingFace) for high-fidelity Portuguese semantic understanding.
* **Inference Model:** **Gemini 2.0 Flash** / **Mixtral** (chosen for long-context reasoning capabilities).

### Evaluation Framework: RAG³
We utilize a multi-dimensional validation strategy to ensure trustworthiness:
1.  **Retrieval Metrics:** Context Recall@k, Context Precision (ensuring we find the right documents).
2.  **Generation Metrics:** ROUGE-L (structural alignment), Groundedness (fact-checking against source text).
3.  **Human Metrics:** Policy Usefulness & Expert Judgment (evaluated by domain specialists).

---

## 🌍 Impact & Alignment
This project aligns with **Sustainable Development Goal 4 (Quality Education)** by democratizing access to complex institutional data. It allows policymakers to move from **static annual snapshots** to **dynamic, evolutionary analysis** of graduate programs.

* **Bias Mitigation:** Fairness-aware retrieval audits to prevent historical prestige bias.
* **Traceability:** Every AI-generated insight is cited with a direct link to the source document (row in CSV or page in PDF).

---

## 📬 Contact
**Femi Samuel Adeola**
* *M.Sc. Candidate, Computer Engineering (FURG)*
* *Research Focus: Trustworthy AI, Neuro-Symbolic Systems, & Educational Data Mining*

[📧 Email](mailto:femi@furg.br) | [🔗 LinkedIn](https://linkedin.com/in/femicrownx)

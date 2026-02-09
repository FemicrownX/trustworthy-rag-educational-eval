# 🎓 Trustworthy RAG System for Educational Program Evaluation
### *Reconciling Interpretability and Accuracy in High-Stakes Decision Making*

> **⚠️ Research Status: Active / Confidential**
> This repository contains the **architectural framework, methodology, and documentation** for my ongoing M.Sc. thesis at the **Federal University of Rio Grande (FURG)**.
>
> *Due to the active nature of this research and data privacy protocols, the source code is currently restricted. Access may be granted to academic supervisors or collaborators upon request.*

---

## 📖 Project Overview
In the domain of education, "black box" AI models are insufficient. When evaluating graduate programs that impact funding and student futures, decision-makers need **traceability, fairness, and truth**.

This research proposes an **AI-Powered Predictive and Analytical System** that integrates **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)**. By grounding the model in official, unstructured reports from the **Sucupira Platform** and **CAPES**, we aim to eliminate "hallucinations" and provide policymakers with verifiable, evidence-based insights.

### 🎯 Core Objectives
* **Mitigate Hallucination:** Reduce the risk of LLMs inventing facts by grounding every response in retrieved, authoritative documents.
* **Ensure Fairness:** Implement bias mitigation strategies to ensure equitable evaluation across diverse regional institutions in Brazil.
* **Human-Verifiable Insights:** Transform complex, unstructured institutional data into transparent policy insights that humans can audit.

---

## 🛠️ System Architecture & Methodology

This system utilizes a **RAG framework** to bridge the gap between static datasets and generative AI.

### 1. Data Acquisition & Preprocessing
* **Source:** **CAPES Open Data Portal** (Structured) & **Sucupira Platform** (Unstructured PDF Reports).
* **Pipeline:**
    * **Text Extraction:** Parsing PDF reports to isolate relevant evaluation criteria.
    * **Chunking:** Using **LangChain** text splitters to segment documents into semantic units.
    * **Normalization:** Cleaning text to remove artifacts and standardize formatting.

### 2. Retrieval Module (The "Memory")
* **Vector Database:** **ChromaDB** is used to index high-dimensional embeddings of the educational data.
* **Search Mechanism:** Semantic similarity search retrieves the specific "chunks" of policy documents relevant to a user's query.

### 3. Generator Module (The "Reasoning")
* **LLM Integration:** A fine-tuned Large Language Model (e.g., via **Hugging Face Transformers**) synthesizes the retrieved context.
* **Output:** Generates a coherent, factual response with citations pointing back to the original CAPES documents.

---

## 🧰 Technical Stack

* **Orchestration:** [LangChain](https://www.langchain.com/) (for chaining retrieval and generation steps).
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (for efficient embedding indexing).
* **Embeddings:** Hugging Face Transformers (BERT/GPT-based models).
* **Language:** Python (PyPDF, Pandas, NumPy).
* **Visualization:** Power BI & R (ggplot2) for output analysis.

---

## 🌍 Ethical Alignment & Social Impact
This project is strictly aligned with **Sustainable Development Goal 4 (SDG 4)**: *Ensure inclusive and equitable quality education*.

* **Bias Mitigation:** We actively audit the retrieval process to prevent historical biases in educational data from influencing future predictions.
* **Data Privacy:** All data handling complies with **LGPD** (Brazil's General Data Protection Law) and ethical research standards defined by the university.
* **Human-in-the-Loop:** This system is designed to *support* human decision-makers, not replace them.

---

## 📬 Contact & Access
**Femi Samuel Adeola**
* *M.Sc. Candidate, Computer Engineering (FURG)*
* *Research Focus: Trustworthy AI, RAG, & Educational Data Mining*

[📧 Email](mailto:femi@furg.br) | [🔗 LinkedIn](https://linkedin.com/in/femicrownx)

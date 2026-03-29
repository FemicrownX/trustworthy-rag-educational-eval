import os
import re
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
PROJECT_ROOT      = r"C:\GINFO LAB\data-parsing-private"
OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "AI-Powered PEPG 2.0 Evaluator", "output")
DB_PATH           = os.path.join(OUTPUT_DIR, "db", "knowledgebase_faiss")
UI_META_PATH      = os.path.join(OUTPUT_DIR, "ui_metadata.csv")
MASTER_INPUT_PATH = os.path.join(OUTPUT_DIR, "Master_Evaluation_Dataset_150.csv")

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
or_key = os.getenv("OPENROUTER_API_KEY")

ENG_STOPS = {'the', 'of', 'and', 'in', 'to', 'a', 'for', 'with', 'by', 'an', 'analysis', 'system', 'based'}

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================
# (These match your notebook exactly. I have consolidated them here for brevity but keep them identical in your file)
PROMPT_ANNUAL_EN = PromptTemplate(template="""You are a CAPES Auditor... [INSERT YOUR EXACT ANNUAL EN PROMPT HERE]""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])
PROMPT_ANNUAL_PT = PromptTemplate(template="""Você é um Auditor da CAPES... [INSERT YOUR EXACT ANNUAL PT PROMPT HERE]""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])
PROMPT_QUADRENNIAL_EN = PromptTemplate(template="""You are a CAPES Auditor performing a **FULL QUADRENNIAL CYCLE EVALUATION**... [INSERT YOUR EXACT QUAD EN PROMPT HERE]""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])
PROMPT_QUADRENNIAL_PT = PromptTemplate(template="""Você é um Auditor da CAPES realizando uma **AVALIAÇÃO DE CICLO QUADRIENAL**... [INSERT YOUR EXACT QUAD PT PROMPT HERE]""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])
PROMPT_SEMANTIC_PT = PromptTemplate(template="Você é um Auditor da CAPES. Baseado APENAS no contexto, extraia fatos específicos.\nContexto: {context}\nPergunta: {question}\nResponda de forma concisa em Português:", input_variables=["context", "question"])
PROMPT_SYMBOLIC = PromptTemplate(template="""You are a CAPES data auditor. Answer the following question with a single integer only.\nDo NOT write a report. Do NOT explain. Do NOT add any text before or after the number.\nOutput ONLY the integer that answers the question.\n\nAudit Metrics (use these first):\n{audit_data}\n\nSupporting Context:\n{context}\n\nQuestion: {question}\n\nAnswer (integer only):""", input_variables=["context", "audit_data", "question"])
PROMPT_BASELINE = PromptTemplate(template="""Responda APENAS com base em seu conhecimento interno sobre os programas da CAPES no Brasil.\nPergunta: {question}\nSe a pergunta pedir uma nota CAPES, a última linha da sua resposta deve ser APENAS: 'Predicted Grade: X' (número inteiro 1-7).\nResponda de forma concisa em Português:""", input_variables=["question"])

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def format_docs_chronological(docs, is_portuguese=False):
    if not docs: return "No evidence found."
    source_label = "Fonte" if is_portuguese else "Source"
    year_label   = "Ano"   if is_portuguese else "Year"
    sorted_docs  = sorted(docs, key=lambda d: str(d.metadata.get('year', '0')))
    formatted    = []
    for d in sorted_docs:
        year = str(d.metadata.get('year', 'N/A'))
        src  = str(d.metadata.get('source', 'Doc')).split('\\')[-1]
        formatted.append(f"--- [{source_label}: {src} | {year_label}: {year}] ---\n{d.page_content[:2000]}\n")
    return "\n".join(formatted)

def get_audit_data_string(program_id, target_years):
    try:
        d_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_docentes.csv"), sep=";")
        s_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_discentes.csv"), sep=";")
        p_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_producoes.csv"), sep=";", dtype={0: str}, low_memory=False)
        part_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_participantes.csv"), sep=";")

        prof_f = d_df[(d_df["programa"] == program_id) & (d_df["year"].astype(str).isin(target_years))]
        stud_f = s_df[(s_df["programa"] == program_id) & (s_df["year"].astype(str).isin(target_years))]
        pub_f  = p_df[(p_df["programa"] == program_id) & (p_df["year"].astype(str).isin(target_years))]
        ext_f  = part_df[(part_df["programa"] == program_id) & (part_df["year"].astype(str).isin(target_years))]

        total_prof = len(prof_f) / len(target_years) if not prof_f.empty and len(target_years) > 0 else 0
        total_pub = len(pub_f)
        impact_per_prof = round(total_pub / max(1, total_prof), 2)

        english_ratio = 0.0
        if not pub_f.empty:
            titles = pub_f["titulo"].dropna().unique().tolist()
            eng_count = sum(1 for t in titles if len(set(re.sub(r'[^\w\s]', '', str(t).lower()).split()).intersection(ENG_STOPS)) >= 2)
            english_ratio = round((eng_count / len(titles) * 100), 1) if titles else 0.0

        grad_efficiency, phd_ratio = 0.0, 0.0
        if not stud_f.empty:
            tits = stud_f["situacao"].str.contains("TITULADO", case=False, na=False).sum()
            desls = stud_f["situacao"].str.contains("DESLIGADO", case=False, na=False).sum()
            if (tits + desls) > 0: grad_efficiency = round((tits / (tits + desls) * 100), 1)
            phd_ratio = round((stud_f["nivel"].str.contains("Doutorado", case=False, na=False).sum() / len(stud_f) * 100), 1)

        return f"- Faculty Size: {round(total_prof, 1)}\n- Student Efficiency: {grad_efficiency}%\n- PhD Training: {phd_ratio}%\n- Total Output: {total_pub}\n- Productivity: {impact_per_prof}\n- INTERNATIONAL QUALITY: {english_ratio}%\n- External Network: {len(ext_f)}"
    except Exception as e:
        return f"Metrics unavailable: {e}"

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
def run_inference(model_label, model_id, is_baseline=False, vectorstore=None):
    output_file = os.path.join(OUTPUT_DIR, f"FINAL_{model_label}_RESULTS_150.csv")

    if os.path.exists(output_file):
        print(f"⏭️  Skipping — output already exists: {output_file}")
        return output_file

    if not os.path.exists(MASTER_INPUT_PATH):
        print(f"❌ CRITICAL ERROR: Master dataset not found at {MASTER_INPUT_PATH}")
        return None

    df = pd.read_csv(MASTER_INPUT_PATH)
    df['Generated_Report'] = ""
    df['Retrieved_Context'] = ""

    llm = ChatOpenAI(model=model_id, api_key=or_key, base_url="https://openrouter.ai/api/v1", temperature=0.1)

    print(f"\n{'='*60}\n🤖 Starting inference for : {model_label}\n{'='*60}")
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating [{model_label}]", colour="green"):
        q, pid, task = row['Question'], str(row['Program_ID']), row['Task_Type']
        period = str(row.get('Year_or_Cycle', '2024'))

        try:
            if is_baseline:
                df.at[idx, 'Generated_Report'] = (PROMPT_BASELINE | llm | StrOutputParser()).invoke({"question": q})
            else:
                if task == 'Semantic':
                    docs = vectorstore.similarity_search(q, k=10)
                    context = format_docs_chronological(docs, is_portuguese=True)
                    df.at[idx, 'Retrieved_Context'] = context
                    df.at[idx, 'Generated_Report'] = (PROMPT_SEMANTIC_PT | llm | StrOutputParser()).invoke({"context": context, "question": q})
                elif task == 'Symbolic':
                    docs = vectorstore.similarity_search(q, k=10, filter={"program_id": pid})
                    context = format_docs_chronological(docs, is_portuguese=True)
                    real_audit_data = get_audit_data_string(pid, [period])
                    df.at[idx, 'Retrieved_Context'] = context
                    df.at[idx, 'Generated_Report'] = (PROMPT_SYMBOLIC | llm | StrOutputParser()).invoke({"context": context, "audit_data": real_audit_data, "question": q})
                else:
                    docs = vectorstore.similarity_search(q, k=10, filter={"program_id": pid})
                    context = format_docs_chronological(docs, is_portuguese=True)
                    target_years = ['2017', '2018', '2019', '2020'] if "2021" in period else [period]
                    real_audit_data = get_audit_data_string(pid, target_years)
                    is_quad = ("Evaluation" in period or "-" in period or len(period) > 4 or task == 'Prediction')
                    prompt = PROMPT_QUADRENNIAL_PT if is_quad else PROMPT_ANNUAL_PT
                    df.at[idx, 'Retrieved_Context'] = context
                    df.at[idx, 'Generated_Report'] = (prompt | llm | StrOutputParser()).invoke({"program_name": f"Program {pid}", "period": period, "audit_data": real_audit_data, "real_titles": "Context", "top_venues": "Context", "context": context, "end_year": period[-4:]})
        except Exception as api_err:
            errors += 1
            df.at[idx, 'Generated_Report'] = "API ERROR - SKIPPED"

        time.sleep(0.8)

    df.to_csv(output_file, index=False)
    print(f"✅ Inference complete — {model_label} (Errors: {errors})")
    return output_file

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    if or_key:
        run_inference("Gemini_2.5", "google/gemini-2.5-flash", vectorstore=vectorstore)
        run_inference("GPT-4o", "openai/gpt-4o", vectorstore=vectorstore)
        run_inference("DeepSeek", "deepseek/deepseek-chat", vectorstore=vectorstore)
        run_inference("Gemini_Baseline", "google/gemini-2.5-flash", is_baseline=True)
        run_inference("GPT-4o_Baseline", "openai/gpt-4o", is_baseline=True)
        run_inference("DeepSeek_Baseline", "deepseek/deepseek-chat", is_baseline=True)
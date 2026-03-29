import os
import re
import pandas as pd
import gradio as gr
import google.generativeai as genai
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Resolve project root and output directory based on current working directory
current_dir = os.getcwd()
if os.path.exists(os.path.join(current_dir, "parsed")):
    PROJECT_ROOT = current_dir
    OUTPUT_DIR = os.path.join(current_dir, "AI-Powered PEPG 2.0 Evaluator", "output")
elif os.path.exists(os.path.join(current_dir, "..", "parsed")):
    PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, ".."))
    OUTPUT_DIR = os.path.join(current_dir, "output")
else:
    PROJECT_ROOT = r"C:\GINFO LAB\data-parsing-private"
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "AI-Powered PEPG 2.0 Evaluator", "output")

DB_PATH = os.path.join(OUTPUT_DIR, "db", "knowledgebase_faiss")
UI_META_PATH = os.path.join(OUTPUT_DIR, "ui_metadata.csv")

# Load API key from .env and inject into environment for LangChain compatibility
ENV_PATH = os.path.join(PROJECT_ROOT, ".env") if os.path.exists(os.path.join(PROJECT_ROOT, ".env")) else r"C:\GINFO LAB\data-parsing-private\.env"
load_dotenv(dotenv_path=ENV_PATH)

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    genai.configure(api_key=google_api_key)
    print("✅ Gemini API connected.")
else:
    print(f"❌ Google API Key not found. Looked in: {ENV_PATH}")

# Maps each CAPES quadrennial evaluation label to its constituent years
CYCLE_MAPPING = {
    "Quadrennial Evaluation 2017": ["2013", "2014", "2015", "2016"],
    "Quadrennial Evaluation 2021": ["2017", "2018", "2019", "2020"],
    "Quadrennial Evaluation 2025": ["2021", "2022", "2023", "2024"]
}

PROMPT_ANNUAL_EN = PromptTemplate(
    template="""You are a CAPES Auditor performing a **YEARLY DIAGNOSTIC MONITORING**.
Focus: Analyze strictly the single year provided: **{period}**.

INPUT DATA:
* METRICS: {audit_data}
* EVIDENCE: {real_titles}
* CONTEXT: {context}

INSTRUCTIONS:
1. Heading: "## Predictive CAPES Evaluation: {program_name} | {period} 📄"
2. Program Profile (Identity & Structure): Synthesize the program's identity based on the Proposal. 
   - Combine its **Mission/Objectives** with its **Concentration Areas** or **Research Lines**.
   - If specific goals are not explicitly stated, infer the profile from the described Research Areas.
   - Cite strictly: (Source: Context {end_year}).
3. DIMENSION ANALYSIS: Analyze Faculty Stability, Student Efficiency, and International Quality (English Ratio).
4. STRATEGIC DIAGNOSIS:
   - List 3 Strengths under the exact header: ### STRENGTHS
     * Format: "**Subject**: Detailed explanation of why this is a strength based on the metrics."
   - List 3 Weaknesses under the exact header: ### WEAKNESSES
     * Format: "**Subject**: Detailed explanation of the deficiency and its potential impact."
5. RECOMMENDATIONS:
   - List 3 Actions under the exact header: ### RECOMMENDATIONS
6. Trending Research: List 5 titles and venues.
(Constraint: Do NOT use bolding or markdown on the ### Headers. Keep them plain.)
7. EVIDENCE CITATION: Every analytical claim made in the Profile, Strengths, and Weaknesses sections MUST be followed by its exact source tag from the context. 
   - Format requirement: "...claim text... (Source: [Filename] | Year: [Year])."
   - Do not hallucinate sources. If a claim comes from the METRICS, cite it as (Source: Quantitative Metrics).
Begin Assessment Report:""", 
    input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"]
)

PROMPT_ANNUAL_PT = PromptTemplate(
    template="""Você é um Auditor da CAPES realizando um **MONITORAMENTO DIAGNÓSTICO ANUAL**.
Foco: Analise estritamente o ano único fornecido: **{period}**.

DADOS DE ENTRADA:
* MÉTRICAS: {audit_data}
* EVIDÊNCIAS: {real_titles}
* CONTEXTO: {context}

INSTRUÇÕES:
1. Título: "## Avaliação CAPES Preditiva: {program_name} | {period} 📄"
2. Perfil do Programa (Identidade e Estrutura): Sintetize a identidade do programa com base na Proposta.
   - Combine sua **Missão/Objetivos** com suas **Áreas de Concentração** ou **Linhas de Pesquisa**.
   - Se os objetivos não estiverem explícitos, infira o perfil a partir das Áreas de Pesquisa descritas.
   - Cite estritamente: (Fonte: Contexto {end_year}).
3. ANÁLISE DAS DIMENSÕES: Analise Estabilidade Docente, Eficiência Discente e Qualidade Internacional (Inglês).
4. DIAGNÓSTICO ESTRATÉGICO:
   - Liste 3 Fortalezas sob o título exato: ### STRENGTHS
     * Formato: "**Tópico**: Explicação detalhada do porquê isso é uma força com base nas métricas."
   - Liste 3 Fragilidades sob o título exato: ### WEAKNESSES
     * Formato: "**Tópico**: Explicação detalhada da deficiência e seu impacto potencial."
5. RECOMENDAÇÕES:
   - Liste 3 Ações sob o título exato: ### RECOMMENDATIONS
6. Pesquisa em Tendência: Liste 5 títulos e locais.
(Restrição: NÃO use negrito ou markdown nos títulos ###. Mantenha-os simples.)
7. CITAÇÃO DE EVIDÊNCIAS: Toda afirmação analítica feita nas seções de Perfil, Fortalezas e Fragilidades DEVE ser seguida pela sua respectiva fonte exata extraída do contexto. 
   - Requisito de formato: "...texto da afirmação... (Fonte: [Nome_do_Arquivo] | Ano: [Ano])."
   - Não invente (alucine) fontes. Se uma afirmação derivar das MÉTRICAS, cite-a estritamente como (Fonte: Métricas Quantitativas).
Início do Relatório:""", 
    input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"]
)

PROMPT_QUADRENNIAL_EN = PromptTemplate(
    template="""You are a CAPES Auditor performing a **FULL QUADRENNIAL CYCLE EVALUATION**.
Focus: Analyze the evolution over the cycle: **{period}**.

INPUT DATA:
* METRICS: {audit_data}
* EVIDENCE: {real_titles}
* CONTEXT: {context}

INSTRUCTIONS:
1. Heading: "## Predictive CAPES Evaluation: {program_name} | {period} 📄"
2. Program Profile (The Chronicle): Structure this section as a DETAILED EVOLUTION (Year 1 to Year 4).
   - Integrate the **Mission/Objectives** with the **Structure (Concentration Areas)**.
   - Describe how the program adheres to its proposed area (Adherence).
3. DIMENSION ANALYSIS: Analyze Faculty Stability, Student Efficiency, Production Quality, and Social Insertion.
4. COMPARATIVE ANALYSIS:
   - List 3 Cycle Strengths under the exact header: ### STRENGTHS
     * Format: "**Subject**: Detailed explanation of why this is a strength based on the cycle evolution."
   - List 3 Cycle Weaknesses under the exact header: ### WEAKNESSES
     * Format: "**Subject**: Detailed explanation of the deficiency and its impact on the final grade."
5. RECOMMENDATIONS:
   - List 3 Strategic Actions under the exact header: ### RECOMMENDATIONS
6. Trending Research:
   - List 5 representative titles from the cycle (Source: Evidence).
7. EVIDENCE CITATION: Every analytical claim made in the Profile, Strengths, and Weaknesses sections MUST be followed by its exact source tag from the context. 
   - Format requirement: "...claim text... (Source: [Filename] | Year: [Year])."
   - Do not hallucinate sources. If a claim comes from the METRICS, cite it as (Source: Quantitative Metrics).
8. Predicted Grade: Based on the metrics, assign a final grade (1 to 7).
   - RULE 1: If "PhD Training" is 0%, the maximum possible grade is 4.
   - RULE 2: Grades 6 and 7 require highly significant "International Quality" (high English Ratio) and strong "Social Insertion".
   - RULE 3: Grade 4 requires strong productivity and high student success. If productivity is moderate or low (< 5 items/professor), strictly assign Grade 3.
   The final line must be ONLY: 'Predicted Grade: X' (Whole Number 1-7).
(Constraint: Do NOT use bolding or markdown on the ### Headers. Keep them plain.)
Begin Assessment Report:""", 
    input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"]
)

PROMPT_QUADRENNIAL_PT = PromptTemplate(
    template="""Você é um Auditor da CAPES realizando uma **AVALIAÇÃO DE CICLO QUADRIENAL**.
Foco: Analise a evolução ao longo do ciclo: **{period}**.

DADOS DE ENTRADA:
* MÉTRICAS: {audit_data}
* EVIDÊNCIAS: {real_titles}
* CONTEXTO: {context}

INSTRUÇÕES:
1. Título: "## Avaliação CAPES Preditiva: {program_name} | {period} 📄"
2. Perfil e Contexto (A Crônica): Estruture esta seção como uma EVOLUÇÃO DETALHADA (Ano 1 a Ano 4).
   - Integre a **Missão/Objetivos** com a **Estrutura (Áreas de Concentração)**.
   - Descreva a aderência do programa à área proposta.
3. ANÁLISE DAS DIMENSÕES: Analise Estabilidade Docente, Eficiência Discente, Qualidade da Produção e Inserção Social.
4. ANÁLISE COMPARATIVA:
   - Liste 3 Fortalezas sob o título exato: ### STRENGTHS
     * Formato: "**Tópico**: Explicação detalhada do porquê isso é uma força com base na evolução do ciclo."
   - Liste 3 Fragilidades sob o título exato: ### WEAKNESSES
     * Formato: "**Tópico**: Explicação detalhada da deficiência e seu impacto na nota final."
5. RECOMENDAÇÕES:
   - Liste 3 Ações sob o título exato: ### RECOMMENDATIONS
6. Pesquisa em Tendência:
   - Liste 5 títulos representativos do ciclo (Fonte: Evidências).
7. CITAÇÃO DE EVIDÊNCIAS: Toda afirmação analítica feita nas seções de Perfil, Fortalezas e Fragilidades DEVE ser seguida pela sua respectiva fonte exata extraída do contexto. 
   - Requisito de formato: "...texto da afirmação... (Fonte: [Nome_do_Arquivo] | Ano: [Ano])."
   - Não invente (alucine) fontes. Se uma afirmação derivar das MÉTRICAS, cite-a estritamente como (Fonte: Métricas Quantitativas).
8. Nota Prevista: Com base nas métricas, atribua uma nota final (1 a 7).
   - REGRA 1: Se o "Treinamento de Doutorado" (PhD Training) for 0%, a nota máxima possível é 4.
   - REGRA 2: Notas 6 e 7 exigem "Qualidade Internacional" altamente significativa (alta proporção de inglês) e forte "Inserção Social".
   - REGRA 3: Nota 4 exige forte produtividade e alto sucesso discente. Se a produtividade for moderada ou baixa (< 5 itens/professor), atribua estritamente Nota 3.
   A última linha deve ser APENAS: 'Predicted Grade: X' (Inteiro 1-7).
(Restrição: NÃO use negrito ou markdown nos títulos ###. Mantenha-os simples.)
Início do Relatório:""", 
    input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"]
)

# Adaptive CSS for light and dark mode UI theming
ADAPTIVE_CSS = """
:root, .light {
    --card-bg: #ffffff; --text-primary: #1E293B; --text-faded: #475569; --accent-color: #00796B; --border-color: rgba(0,0,0,0.08); --roster-bg: #F8FAFC;
    --str-bg: #F0FDF4; --str-border: #4ADE80; --str-text: #166534;
    --weak-bg: #FEF2F2; --weak-border: #F87171; --weak-text: #991B1B;
    --rec-bg: #EFF6FF; --rec-border: #60A5FA; --rec-text: #1E40AF;
}
.dark {
    --card-bg: rgba(255,255,255,0.02); --text-primary: #DCE3F0; --text-faded: #94a3b8; --accent-color: #00BFA6; --border-color: rgba(255,255,255,0.12); --roster-bg: rgba(0,0,0,0.2);
    --str-bg: rgba(74, 222, 128, 0.1); --str-border: #4ADE80; --str-text: #86EFAC;
    --weak-bg: rgba(248, 113, 113, 0.1); --weak-border: #F87171; --weak-text: #FCA5A5;
    --rec-bg: rgba(96, 165, 250, 0.1); --rec-border: #60A5FA; --rec-text: #93C5FD;
}
.metric-compact-wrapper { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; }
.metric-compact { flex: 1; min-width: 220px; padding: 20px; border-radius: 12px; background: var(--card-bg); border: 1px solid var(--border-color); transition: all 0.25s ease; }
.metric-compact h4 { margin: 0 0 12px 0; font-size: 0.9em; color: var(--text-faded); text-transform: uppercase; }
.metric-compact .metric-value { font-weight: 700; font-size: 1.8em; margin-right: 5px; }
.roster-wrapper { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; width: 100%; }
.roster-card { flex: 1; min-width: 350px; padding: 20px; border-radius: 12px; background: var(--roster-bg); border: 1px solid var(--border-color); }
.roster-card h4 { margin: 0 0 10px 0; color: var(--accent-color); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
.roster-content { font-size: 0.85em; line-height: 1.5; }
.analysis-wrapper { display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; width: 100%; }
.strength-box, .weakness-box, .rec-box { flex: 1; min-width: 300px; padding: 20px; border-radius: 8px; border-left: 5px solid; }
.strength-box { background: var(--str-bg); border-color: var(--str-border); color: var(--str-text); }
.weakness-box { background: var(--weak-bg); border-color: var(--weak-border); color: var(--weak-text); }
.rec-box { background: var(--rec-bg); border-color: var(--rec-border); color: var(--rec-text); width: 100%; margin-top: 15px; }
.strength-box h3, .weakness-box h3, .rec-box h3 { margin-top: 0; display: flex; align-items: center; gap: 8px; }
"""

# Builds the HTML metrics and faculty roster dashboard for a given program and period
def generate_html_dashboard(program_name, program_id, selection, metrics, faculty_str="N/A", partners_str="N/A"):
    metrics_html = f"""
    <div class='metric-compact-wrapper'>
      <div class='metric-compact'><h4>👩‍🏫 Professors</h4><p><span class='metric-value'>{metrics['total_prof']}</span> <span class='metric-label'>Avg/Year</span></p><p><span class='metric-value'>{metrics['stability']}%</span> <span class='metric-label'>Stability</span></p></div>
      <div class='metric-compact'><h4>🎓 Students</h4><p><span class='metric-value'>{metrics['total_stud']}</span> <span class='metric-label'>Avg/Year</span></p><p><span class='metric-value'>{metrics['graduation']}%</span> <span class='metric-label'>Success Rate</span></p></div>
      <div class='metric-compact'><h4>📚 Publications</h4><p><span class='metric-value'>{metrics['total_pub']}</span> <span class='metric-label'>Total Period</span></p><p><span class='metric-value'>{metrics['visibility']}%</span> <span class='metric-label'>Visibility</span></p></div>
      <div class='metric-compact'><h4>🌍 Internationalization</h4><p><span class='metric-value'>{metrics['external']}</span> <span class='metric-label'>Ext. Participants</span></p><p><span class='metric-value'>{metrics['intl_rate']}</span> <span class='metric-label'>Collab Ratio</span></p></div>
    </div>"""
    roster_html = f"""
    <div class='roster-wrapper'>
        <div class='roster-card'><h4>👥 Core Faculty (Permanent)</h4><div class='roster-content'>{faculty_str}</div></div>
        <div class='roster-card'><h4>🌍 External Collaborators (Network)</h4><div class='roster-content'>{partners_str}</div></div>
    </div>"""
    return metrics_html + roster_html

# Sorts retrieved documents by year and formats them with source and year labels for the LLM context
def format_docs_chronological(docs, is_portuguese=False):
    if not docs: 
        return "Nenhuma evidência textual específica encontrada para este período." if is_portuguese else "No specific textual evidence found for this period."
    
    source_label = "Fonte" if is_portuguese else "Source"
    year_label = "Ano" if is_portuguese else "Year"
    
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get('year', '0'))
    formatted, seen = [], []
    for d in sorted_docs:
        year = str(d.metadata.get('year', 'N/A'))
        source = str(d.metadata.get('source', 'Unknown Document')).split('\\')[-1].split('/')[-1] 
        
        doc_chunk = f"--- [{source_label}: {source} | {year_label}: {year}] ---\n{d.page_content[:2000]}\n"
        
        if year not in seen:
            seen.append(year)
            context_header = f"### CONTEXTO DO ANO {year} ###" if is_portuguese else f"### CONTEXT FOR YEAR {year} ###"
            formatted.append(f"\n{context_header}\n{doc_chunk}")
        else:
            formatted.append(doc_chunk)
    return "\n".join(formatted)

# Estimates the percentage of publication titles written in English using stopword matching
def detect_language_ratio(titles_list):
    if not titles_list: return 0.0
    english_stops = {'the', 'of', 'and', 'in', 'to', 'a', 'for', 'with', 'by', 'an', 'analysis', 'system', 'based'}
    english_count = 0
    for t in titles_list:
        words = set(re.sub(r'[^\w\s]', '', str(t).lower()).split())
        if len(words.intersection(english_stops)) >= 2: english_count += 1
    return round((english_count / len(titles_list) * 100), 1)

# Wraps LLM-generated STRENGTHS, WEAKNESSES, and RECOMMENDATIONS sections in styled HTML boxes
def inject_visual_boxes(text, is_portuguese=False):
    h_str = "Fortalezas" if is_portuguese else "Strengths"
    h_weak = "Fragilidades" if is_portuguese else "Weaknesses"
    h_rec = "Recomendações Estratégicas" if is_portuguese else "Strategic Recommendations"
    
    text = text.replace("### ✅ STRENGTHS", f"<div class='analysis-wrapper'><div class='strength-box'><h3>✅ {h_str}</h3>")
    text = text.replace("### ⚠️ WEAKNESSES", f"</div><div class='weakness-box'><h3>⚠️ {h_weak}</h3>")
    text = text.replace("### 🚀 RECOMMENDATIONS", f"</div></div><div class='rec-box'><h3>🚀 {h_rec}</h3>")
    
    if "Predicted Grade:" in text: text = text.replace("Predicted Grade:", "</div>\n\n**Predicted Grade:**")
    elif "Nota Prevista:" in text: text = text.replace("Nota Prevista:", "</div>\n\n**Nota Prevista:**")
    elif "### Trending Research" in text: text = text.replace("### Trending Research", "</div>\n\n### Trending Research")
    elif "### Pesquisa em Tendência" in text: text = text.replace("### Pesquisa em Tendência", "</div>\n\n### Pesquisa em Tendência")
    else: text += "</div>"
    return text

# Core RAG evaluation pipeline: loads metrics from CSVs, retrieves FAISS context, and runs the LLM chain
def run_evaluation(selection, program_with_id, language_toggle, is_quadrennial=False, custom_llm=None):
    if not all([selection, program_with_id]): return "Please select all fields.", "N/A", "N/A", "### No selection"

    active_llm = custom_llm if custom_llm else llm

    try:
        program_id = str(program_with_id.split(' (ID:')[1].replace(')', '').strip())
        program_name = program_with_id.split(' (ID:')[0].replace(':', '').strip()
    except: return "Error parsing program ID.", "Error", "N/A", "### Error"
    
    use_pt = "Português" in str(language_toggle)
    sel_str = str(selection).strip()
    
    if is_quadrennial:
        if "2017" in sel_str: cycle_years = ['2013','2014','2015','2016']
        elif "2021" in sel_str: cycle_years = ['2017','2018','2019','2020']
        elif "2025" in sel_str: cycle_years = ['2021','2022','2023','2024']
        else: cycle_years = ['2021','2022','2023','2024']
        
        target_years = cycle_years
        benchmark_year = cycle_years[0] 
        period_label = f"Quadrennial Cycle ({target_years[0]} - {target_years[-1]})"
    else:
        try: sel_year = int(sel_str.split()[-1])
        except: return f"Error parsing year: {selection}", "Error", "N/A", "### Error"
        target_years = [str(sel_year)]
        cycle_years = target_years
        benchmark_year = None
        period_label = f"Annual Monitor ({sel_year})"

    metrics = {'total_prof': 0, 'stability': 0, 'perm_count': 0, 'total_stud': 0, 'matriculado_count': 0, 
               'total_pub': 0, 'visibility': 0, 'external': 0, 'intl_rate': 0, 'english_ratio': 0}
    faculty_names_str, partners_str = "Data not available", "Data not available"
    real_titles_str, top_venues_str = "No specific titles found.", "No specific venues found."

    try:
        d_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_docentes.csv"), sep=";")
        s_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_discentes.csv"), sep=";")
        p_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_producoes.csv"), sep=";", dtype={0: str}, low_memory=False)
        part_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_participantes.csv"), sep=";")
        
        prof_f = d_df[(d_df["programa"] == program_id) & (d_df["year"].astype(str).isin(target_years))]
        stud_f = s_df[(s_df["programa"] == program_id) & (s_df["year"].astype(str).isin(target_years))]
        pub_f = p_df[(p_df["programa"] == program_id) & (p_df["year"].astype(str).isin(target_years))]
        ext_f = part_df[(part_df["programa"] == program_id) & (part_df["year"].astype(str).isin(target_years))]

        unique_years = len(target_years)
        
        if not prof_f.empty:
            metrics['total_prof'] = round(len(prof_f) / unique_years, 1)
            perm_mask = prof_f["categoria"].str.contains("PERMANENTE", case=False, na=False)
            metrics['perm_count'] = int(perm_mask.sum()) 
            metrics['stability'] = round((perm_mask.sum() / len(prof_f) * 100), 1)
            perm_names = prof_f[perm_mask]['nome'].dropna().unique().tolist()
            if perm_names:
                display_names = sorted(perm_names)[:10]
                faculty_names_str = ", ".join([n.title() for n in display_names])
                if len(perm_names) > 10: faculty_names_str += f", and {len(perm_names)-10} others."
        
        if not ext_f.empty:
            metrics['external'] = len(ext_f) 
            if metrics['total_prof'] > 0: metrics['intl_rate'] = round(len(ext_f) / metrics['total_prof'], 2)
            name_col = next((c for c in ext_f.columns if 'nome' in c.lower()), None)
            if name_col:
                vals = [str(p).title() for p in ext_f[name_col].dropna().unique().tolist() if len(str(p)) > 3]
                if vals:
                    partners_str = ", ".join(sorted(vals)[:10])
                    if len(vals) > 10: partners_str += f", and {len(vals)-10} others."

        metrics['total_pub'] = len(pub_f) 
        if not pub_f.empty:
            metrics['visibility'] = round((pub_f["titulo"].nunique() / len(pub_f) * 100), 1)
            titles_list = pub_f["titulo"].dropna().unique().tolist()
            metrics['english_ratio'] = detect_language_ratio(titles_list)
            
            sample_size = min(5, len(pub_f))
            sampled_rows = pub_f.sample(n=sample_size, random_state=42)
            formatted_titles = []
            for _, row in sampled_rows.iterrows():
                title, venue = str(row['titulo']).strip(), str(row['periodico/conferencia']).strip()
                if venue and venue.lower() != 'nan': formatted_titles.append(f"- {title} (Venue: {venue})")
                else: formatted_titles.append(f"- {title}")
            real_titles_str = "\n".join(formatted_titles)
            if "periodico/conferencia" in pub_f.columns:
                top_venues_str = "\n".join([f"- {v}" for v in pub_f["periodico/conferencia"].dropna().value_counts().head(5).index.tolist()])
        
        grad_efficiency = 0; phd_ratio = 0
        if not stud_f.empty:
            metrics['total_stud'] = round(len(stud_f) / len(target_years), 1)
            metrics['matriculado_count'] = stud_f["situacao"].str.contains("MATRICULADO", case=False, na=False).sum()
            titulados = stud_f["situacao"].str.contains("TITULADO", case=False, na=False).sum()
            desligados = stud_f["situacao"].str.contains("DESLIGADO", case=False, na=False).sum()
            if (titulados + desligados) > 0:
                success_rate = round((titulados / (titulados + desligados) * 100), 1)
                metrics['graduation'] = success_rate
                grad_efficiency = success_rate
                phd_ratio = round((stud_f["nivel"].str.contains("Doutorado", case=False, na=False).sum() / len(stud_f) * 100), 1)

        metrics['impact_per_prof'] = round(metrics['total_pub'] / max(1, metrics['total_prof']), 2)

        audit_data = f"""
        --- 1. PROGRAM & FORMATION ---
        - Faculty Size: {metrics['total_prof']} (Stability: {metrics['stability']}%)
        - Student Efficiency: {grad_efficiency}% (Success Rate)
        - PhD Training: {phd_ratio}% of students are Doctoral candidates
        --- 2. RESEARCH IMPACT & QUALITY ---
        - Total Output: {metrics['total_pub']} items
        - Productivity: {metrics['impact_per_prof']} items/professor
        - INTERNATIONAL QUALITY: {metrics['english_ratio']}% of titles are in English
        --- 3. SOCIAL RELEVANCE & NETWORK ---
        - External Network: {metrics['external']} participants
        - Collaboration Intensity: {metrics['intl_rate']} per prof
        """
        dashboard_html = generate_html_dashboard(program_name, program_id, selection, metrics, faculty_names_str, partners_str)
        
        dashboard_html += f"""
        <div style='display:none;' id='evaluator-probes'>
            <span class='metric-value'>{metrics['perm_count']}</span>
            <span class='metric-value'>{metrics['matriculado_count']}</span>
            <span class='metric-value'>{metrics['total_pub']}</span>
            <span class='metric-value'>{metrics['external']}</span>
        </div>
        """
    except Exception as e:
        dashboard_html = f"<div style='color:red;'>Error reading local data: {str(e)}</div>"

    official_grade = "N/A"
    try:
        if is_quadrennial and benchmark_year:
            grade_row = meta_df[(meta_df["program_id"] == program_id) & (meta_df["year"] == benchmark_year)]
            if not grade_row.empty:
                val = grade_row.iloc[0]['grade']
                if pd.notna(val) and str(val).lower().strip() != 'nan':
                    official_grade = str(int(float(val))) if str(val).replace('.','',1).isdigit() else str(val)
    except: pass

    search_query = f"{program_name} objetivos metas autoavaliação"
    k_val = 50 if is_quadrennial else 20
    
    raw_docs = vectorstore.similarity_search(search_query, k=k_val, filter={"program_id": program_id})
    
    if len(raw_docs) == 0:
        raw_docs = vectorstore.similarity_search(program_id, k=k_val, filter={"program_id": program_id})
    if len(raw_docs) == 0:
        raw_docs = vectorstore.similarity_search(program_name, k=k_val, filter={"program_id": program_id})
    
    allowed_context_years = set(target_years)
    allowed_context_years.add(cycle_years[0])
    filtered_docs = [d for d in raw_docs if str(d.metadata.get('year', 'N/A')) in allowed_context_years]
    
    if len(filtered_docs) == 0:
        filtered_docs = raw_docs
    
    formatted_context = format_docs_chronological(filtered_docs, is_portuguese=use_pt)
    
    prompt = (PROMPT_QUADRENNIAL_PT if use_pt else PROMPT_QUADRENNIAL_EN) if is_quadrennial else (PROMPT_ANNUAL_PT if use_pt else PROMPT_ANNUAL_EN)
    chain = (prompt | active_llm | StrOutputParser())

    try:
        report = chain.invoke({
            "context": formatted_context, "audit_data": audit_data, "real_titles": real_titles_str,
            "top_venues": top_venues_str, "program_name": program_name, "period": period_label, "end_year": target_years[-1]
        })
        
        disclaimer = "\n\n*Nota: As publicações são recuperadas com base na relevância semântica...*" if use_pt else "\n\n*Note: Publications are retrieved based on semantic relevance...*"
        report = report + disclaimer

        if is_quadrennial:
            match = re.search(r'(?:Predicted Grade|Nota Prevista)[*:\s]*([0-7])', report, re.IGNORECASE)
            predicted_grade = match.group(1) if match else "?"
        else:
            report = re.sub(r'(?:Predicted Grade|Nota Prevista).*(\n|$)', '', report, flags=re.IGNORECASE)
            predicted_grade = "Monitor"
            official_grade = "Last: " + official_grade
            
        report = inject_visual_boxes(report, is_portuguese=use_pt)
        return report, predicted_grade, official_grade, dashboard_html
    except Exception as e:
        return f"System Error: {str(e)}", "Error", official_grade, dashboard_html


if __name__ == "__main__":
    
    LLM_MODEL = "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS Knowledgebase Connected.")
    else:
        print("❌ WARNING: FAISS Index not found. RAG won't work.")

    if os.path.exists(UI_META_PATH):
        meta_df = pd.read_csv(UI_META_PATH)
        meta_df["program_name"] = meta_df["program_name"].astype(str).str.replace(r'^[:\s]+', '', regex=True).str.strip()
        meta_df["year"] = meta_df["year"].astype(str).str.replace(r'\.0$', '', regex=True)
        meta_df["program_id"] = meta_df["program_id"].astype(str)
        print(f"✅ Metadata Loaded: {len(meta_df)} rows.")

        def get_inst_annual(year):
            df_year = meta_df[meta_df['year'] == str(year)]
            opts = sorted([f"{r['instituicao']} ({r['ies_sigla']})" for _, r in df_year.iterrows()])
            return gr.Dropdown(choices=sorted(list(set(opts))), value=None, interactive=True)

        def get_inst_quad(cycle_name):
            ref_year = "2016" if "2017" in str(cycle_name) else "2020"
            df_year = meta_df[meta_df['year'] == ref_year]
            opts = sorted([f"{r['instituicao']} ({r['ies_sigla']})" for _, r in df_year.iterrows()])
            return gr.Dropdown(choices=sorted(list(set(opts))), value=None, interactive=True)

        def get_prog(year_or_cycle, inst_str):
            if not inst_str: return gr.Dropdown(choices=[])
            if "Quadrennial" in str(year_or_cycle): target_year = "2016" if "2017" in str(year_or_cycle) else "2020"
            else: target_year = str(year_or_cycle)
            inst_name = inst_str.split(" (")[0]
            rows = meta_df[(meta_df['year'] == target_year) & (meta_df['instituicao'] == inst_name)]
            opts = [f"{r['program_name']} (ID: {r['program_id']})" for _, r in rows.iterrows()]
            return gr.Dropdown(choices=sorted(list(set(opts))), value=None, interactive=True)

        with gr.Blocks(css=ADAPTIVE_CSS, theme="soft") as demo:
            gr.Markdown("# 📈 AI-Powered PEPG 2.0 Evaluator")
            gr.Markdown("### *A Hybrid⚡RAG Prescriptive Audit & Predictive System*")
            
            lang = gr.Radio(["🇧🇷 Português", "🇺🇸 English"], value="🇧🇷 Português", label="Language Selection")

            with gr.Tabs():
                with gr.TabItem("Annual Progress Monitoring"):
                    with gr.Row():
                        y1 = gr.Dropdown(choices=sorted(meta_df["year"].unique().tolist(), reverse=True), label="1. Select Audit Year")
                        i1 = gr.Dropdown(label="2. Select Institution", interactive=False)
                        p1 = gr.Dropdown(label="3. Select Program", interactive=False)
                    
                    btn1 = gr.Button("Generate Monitoring Report", variant="primary")
                    dash1 = gr.HTML(value="<p>Quantitative overview pending selection...</p>")
                    out_r1 = gr.Markdown()
                    
                    y1.change(fn=get_inst_annual, inputs=y1, outputs=i1)
                    i1.change(fn=get_prog, inputs=[y1, i1], outputs=p1)
                    btn1.click(lambda y, p, l: run_evaluation(y, p, l, False), [y1, p1, lang], [out_r1, gr.State(), gr.State(), dash1])

                with gr.TabItem("Quadrennial Cycle Evaluation"):
                    with gr.Row():
                        y2 = gr.Dropdown(choices=list(CYCLE_MAPPING.keys()), label="* Select Evaluation Period")
                        i2 = gr.Dropdown(label="Institution", interactive=False)
                        p2 = gr.Dropdown(label="Program Selection", interactive=False)
                    
                    btn2 = gr.Button("Calculate Cycle Assessment", variant="primary")
                    dash2 = gr.HTML(value="<p>Cycle overview pending selection...</p>")
                    
                    with gr.Row():
                        out_p2 = gr.Textbox(label="AI Predicted Cycle Grade (1-7)")
                        out_a2 = gr.Textbox(label="Official CAPES Grade")
                    
                    out_r2 = gr.Markdown()
                    
                    y2.change(fn=get_inst_quad, inputs=y2, outputs=i2)
                    i2.change(fn=get_prog, inputs=[y2, i2], outputs=p2)
                    btn2.click(lambda c, p, l: run_evaluation(c, p, l, True), [y2, p2, lang], [out_r2, out_p2, out_a2, dash2])

        demo.launch(share=True)
    else:
        print("❌ Metadata missing. Run your ETL script first.")
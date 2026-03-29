# Annual Monitoring Prompts
PROMPT_ANNUAL_EN = PromptTemplate(
    template="""
You are a CAPES Auditor performing a **YEARLY DIAGNOSTIC MONITORING**.
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

Begin Assessment Report:
""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])

PROMPT_ANNUAL_PT = PromptTemplate(
    template="""
Você é um Auditor da CAPES realizando um **MONITORAMENTO DIAGNÓSTICO ANUAL**.
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

Início do Relatório:
""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])

# Cycle Evaluation Prompts
PROMPT_QUADRENNIAL_EN = PromptTemplate(
    template="""
You are a CAPES Auditor performing a **FULL QUADRENNIAL CYCLE EVALUATION**.
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
7. Predicted Grade: The final line must be ONLY: 'Predicted Grade: X' (Whole Number 3-7).

(Constraint: Do NOT use bolding or markdown on the ### Headers. Keep them plain.)

Begin Assessment Report:
""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])

PROMPT_QUADRENNIAL_PT = PromptTemplate(
    template="""
Você é um Auditor da CAPES realizando uma **AVALIAÇÃO DE CICLO QUADRIENAL**.
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
7. Nota Prevista: A última linha deve ser APENAS: 'Predicted Grade: X' (Inteiro 3-7).

(Restrição: NÃO use negrito ou markdown nos títulos ###. Mantenha-os simples.)

Início do Relatório:
""", input_variables=["program_name", "period", "audit_data", "real_titles", "top_venues", "context", "end_year"])

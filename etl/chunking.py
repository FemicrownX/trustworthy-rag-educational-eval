from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def create_rag_documents(master_docentes, master_discentes, master_producoes, joint_proposals):
    """Builds the RAG document corpus from summaries, profiles, and proposals."""
    print("\n--- 🧩 PHASE 3: RAG CORPUS CREATION ---")
    all_documents = []

    # 1. Structured summary per program/year
    if not master_docentes.empty:
        for (pid, year), group in master_docentes.groupby(["program_id", "year"]):
            inst = group["instituicao"].iloc[0] if "instituicao" in group.columns else "Unknown"
            stud_count = len(master_discentes[(master_discentes['program_id']==pid) & (master_discentes['year']==year)])
            pub_count = len(master_producoes[(master_producoes['program_id']==pid) & (master_producoes['year']==year)])
            
            text = (f"Structured summary for program {pid} at {inst} in {year}:\n"
                    f"- Faculty: {group['nome'].nunique()}\n- Students: {stud_count}\n- Publications: {pub_count}")
            all_documents.append(Document(page_content=text, metadata={"source":"summary", "program_id":pid, "year":year}))

    # 2. Professor profiles with recent publications
    if not master_docentes.empty:
        unique_profs = master_docentes.sort_values('year', ascending=False).drop_duplicates('nome')
        for _, row in tqdm(unique_profs.iterrows(), total=len(unique_profs), desc="Profiling Faculty"):
            name = row['nome']
            text = f"Profile for Professor {name}. Involved in program {row['program_id']}."
            pubs = master_producoes[master_producoes['docentes'].astype(str).str.contains(name, na=False, regex=False)]
            
            if not pubs.empty:
                text += "\nRecent Titles: " + "; ".join(pubs['titulo'].head(3).tolist())
            all_documents.append(Document(page_content=text, metadata={"source":"profile", "program_id":row['program_id']}))

    # 3. Unstructured proposals
    for key, content in joint_proposals.items():
        pid, year = key.split('_')
        all_documents.append(Document(page_content=f"PROPOSAL:\n{content}", metadata={"source":"proposta.txt", "program_id":pid, "year":year}))

    print(f"✅ Pre-chunking RAG Documents: {len(all_documents)}")
    return all_documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Applies the RecursiveCharacterTextSplitter to the document list."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"✅ Generated {len(chunks)} text chunks.")
    return chunks
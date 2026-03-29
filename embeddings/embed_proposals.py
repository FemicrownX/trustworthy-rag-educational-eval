import os
import re
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_faiss_index(chunks, db_path):
    """Embeds chunks using HuggingFace and saves the FAISS index."""
    print("\n--- 🧠 PHASE 4: FAISS INDEXING ---")
    
    if os.path.exists(db_path):
        print(f"⏭️ Skipping Embeddings: Database already exists at {db_path}")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)
    print("✅ FAISS Index successfully built and saved.")

def generate_ui_metadata(master_docentes, joint_proposals, sucupira_master_path, ui_meta_path):
    """Extracts program names and builds dropdown metadata for the Gradio UI."""
    if os.path.exists(ui_meta_path) or master_docentes.empty:
        return

    prog_names = {}
    for key, text in joint_proposals.items():
        pid = key.split('_')[0]
        match = re.search(r"(?:NOME DO PROGRAMA|PROGRAMA DE PÓS-GRADUAÇÃO EM|PROGRAMA:)\s*([\s\S]*?)(?=\n)", text, re.IGNORECASE)
        prog_names[pid] = match.group(1).strip() if match else f"Program {pid}"
    
    req_cols = ['year', 'program_id', 'instituicao', 'ies_sigla']
    ui_df = master_docentes[[c for c in req_cols if c in master_docentes.columns]].drop_duplicates().astype(str)
    ui_df['program_name'] = ui_df['program_id'].map(prog_names).fillna("Unknown Program")
    
    if os.path.exists(sucupira_master_path):
        grades = pd.read_csv(sucupira_master_path)[['program_id','year','grade']].astype(str)
        ui_df = ui_df.merge(grades, on=['program_id','year'], how='left')
    
    ui_df.to_csv(ui_meta_path, index=False)
    print(f"✅ UI Metadata Saved: {len(ui_df)} Program-Year entries.")
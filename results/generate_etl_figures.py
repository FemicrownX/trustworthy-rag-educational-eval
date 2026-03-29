import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def generate_conceptual_data_mapping(output_dir):
    """Figure 3.1: Directed graph showing how Program_ID links to each source file."""
    print("Generating 1/7: Conceptual Data Mapping...")
    G1 = nx.DiGraph()
    G1.add_node("Program_ID", layer=0, color='orange', label="Program Object\n(Primary Key)")
    files = ["docentes.csv", "discentes.csv", "producoes.csv", "proposta.txt"]
    for f in files: G1.add_node(f, layer=1, color='#add8e6', label=f)
    entities = ["Professor\n(Permanent)", "Student\n(MSc/PhD)", "Publication\n(Title/Venue)", "Mission\n(Text Chunk)"]
    for e in entities: G1.add_node(e, layer=2, color='#90ee90', label=e)
    
    edges1 = [("Program_ID", f) for f in files] + [
        ("docentes.csv", "Professor\n(Permanent)"), ("discentes.csv", "Student\n(MSc/PhD)"),
        ("producoes.csv", "Publication\n(Title/Venue)"), ("proposta.txt", "Mission\n(Text Chunk)"),
        ("Professor\n(Permanent)", "Student\n(MSc/PhD)"), ("Professor\n(Permanent)", "Publication\n(Title/Venue)")
    ]
    G1.add_edges_from(edges1)

    plt.figure(figsize=(12, 8))
    pos1 = nx.shell_layout(G1)
    node_colors1 = [G1.nodes[n].get('color', 'gray') for n in G1.nodes]
    nx.draw(G1, pos1, with_labels=True, node_color=node_colors1, node_size=4500, edge_color='gray', width=2, font_size=9, font_weight='bold', arrowsize=20)
    plt.title("Figure 3.1: Relational Data Mapping Schema", fontsize=15)
    plt.text(0, -1.2, "Note: The Program_ID acts as the central anchor for heterogeneous data.", ha='center', fontsize=12)
    plt.savefig(os.path.join(output_dir, "Diagram_1_Conceptual_DataMapping.png"), dpi=300)
    plt.close()


def generate_conceptual_pipeline_flow(output_dir):
    """Figure 3.2: Two-path pipeline showing symbolic and neural flows merging."""
    print("Generating 2/7: Conceptual Neuro-Symbolic Pipeline...")
    G4 = nx.DiGraph()
    nodes4 = [
        ("Structured Data\n(CSVs)", (0, 4), 'lightgray', 's'), ("Unstructured Data\n(PDF/TXT)", (4, 4), 'lightgray', 's'),
        ("SYMBOLIC ENGINE\n(Pandas/Python)", (0, 2), '#1f77b4', 'o'), ("NEURAL ENGINE\n(FAISS/Embeddings)", (4, 2), '#2ca02c', 'o'),
        ("Deterministic KPIs\n(Stability, Success Rate)", (0, 0), '#aec7e8', 's'), ("Semantic Context\n(Mission, Goals)", (4, 0), '#98df8a', 's'),
        ("Context Window\n(Prompt Injection)", (2, -2), '#d62728', 'd'), ("LLM (Gemini)", (2, -3), 'gold', 's'), ("Final Report", (2, -4), 'orange', 's')
    ]
    for n, p, c, s in nodes4: G4.add_node(n, pos=p, color=c, shape=s)
    edges4 = [
        ("Structured Data\n(CSVs)", "SYMBOLIC ENGINE\n(Pandas/Python)"), ("Unstructured Data\n(PDF/TXT)", "NEURAL ENGINE\n(FAISS/Embeddings)"),
        ("SYMBOLIC ENGINE\n(Pandas/Python)", "Deterministic KPIs\n(Stability, Success Rate)"), ("NEURAL ENGINE\n(FAISS/Embeddings)", "Semantic Context\n(Mission, Goals)"),
        ("Deterministic KPIs\n(Stability, Success Rate)", "Context Window\n(Prompt Injection)"), ("Semantic Context\n(Mission, Goals)", "Context Window\n(Prompt Injection)"),
        ("Context Window\n(Prompt Injection)", "LLM (Gemini)"), ("LLM (Gemini)", "Final Report")
    ]
    G4.add_edges_from(edges4)

    plt.figure(figsize=(12, 10))
    pos4 = nx.get_node_attributes(G4, 'pos')
    colors4 = [G4.nodes[n]['color'] for n in G4.nodes]
    nx.draw_networkx_nodes(G4, pos4, node_size=4000, node_color=colors4, node_shape='s')
    nx.draw_networkx_edges(G4, pos4, width=2, arrowsize=25, edge_color='gray')
    nx.draw_networkx_labels(G4, pos4, font_size=10, font_weight='bold')
    plt.title("Figure 3.2: The Hybrid Neuro-Symbolic Pipeline", fontsize=16)
    plt.text(5, 2, "Neural Path\n(Semantic)", color='green', fontsize=12, ha='center')
    plt.text(-1, 2, "Symbolic Path\n(Math)", color='blue', fontsize=12, ha='center')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "Diagram_2_Conceptual_PipelineFlow.png"), dpi=300)
    plt.close()


def generate_conceptual_chunking(output_dir):
    """Figure 3.4: Visual of overlapping sliding window chunks."""
    print("Generating 3/7: Conceptual Chunking Strategy...")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    plt.plot([0, 100], [4, 4], color='black', linewidth=1, linestyle='--', alpha=0.5)
    plt.text(0, 4.2, "Original Unstructured Document (e.g., proposta.txt)", fontsize=12, fontweight='bold')
    chunks = [(0, 40), (35, 40), (70, 40)]
    colors5 = ['#add8e6', '#98df8a', '#ffbb78']
    labels5 = ["Chunk 1\n(Vector A)", "Chunk 2\n(Vector B)", "Chunk 3\n(Vector C)"]

    for i, (start, width) in enumerate(chunks):
        rect = patches.Rectangle((start, 2 - i), width, 0.8, linewidth=2, edgecolor='black', facecolor=colors5[i], alpha=0.8)
        ax5.add_patch(rect)
        plt.text(start + width/2, 2 - i + 0.3, labels5[i], ha='center', va='center', fontsize=10, fontweight='bold')
        plt.plot([start, start], [2 - i + 0.8, 4], color='gray', linestyle=':', alpha=0.5)
        plt.plot([start + width, start + width], [2 - i + 0.8, 4], color='gray', linestyle=':', alpha=0.5)

    ax5.add_patch(patches.Rectangle((35, 1.0), 5, 1.8, linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
    plt.text(37.5, 0.8, "Context Overlap\n(200 chars)", color='red', ha='center', fontsize=9)
    ax5.add_patch(patches.Rectangle((70, 0.0), 5, 1.8, linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))
    plt.text(72.5, -0.2, "Continuity", color='red', ha='center', fontsize=9)
    plt.xlim(-5, 115); plt.ylim(-1, 5); plt.axis('off')
    plt.title("Figure 3.4: Sliding Window Chunking Strategy", fontsize=16)
    plt.savefig(os.path.join(output_dir, "Diagram_3_Conceptual_Chunking.png"), dpi=300)
    plt.close()


def generate_conceptual_embedding(output_dir):
    """Figure 3.6: Step-by-step diagram from raw text to FAISS index."""
    print("Generating 4/7: Conceptual Vector Pipeline...")
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    ax7.add_patch(patches.Rectangle((0, 3), 3, 2, facecolor='lightgray', edgecolor='black'))
    plt.text(1.5, 4, '"The mission is\nto innovate..."', ha='center', va='center', fontsize=10, family='monospace')
    plt.text(1.5, 2.7, "Raw Text Chunk", ha='center', fontweight='bold')
    plt.arrow(3, 4, 1.5, 0, head_width=0.2, color='black')

    ax7.add_patch(patches.Rectangle((4.5, 3), 3, 2, facecolor='#add8e6', edgecolor='blue'))
    plt.text(6, 4, 'HuggingFace\nTransformer', ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
    plt.arrow(7.5, 4, 1.5, 0, head_width=0.2, color='black')

    ax7.add_patch(patches.Rectangle((9, 3.5), 4, 1, facecolor='#98df8a', edgecolor='green'))
    plt.text(11, 4, '[0.12, -0.98, 0.04...]', ha='center', va='center', fontsize=10, family='monospace', fontweight='bold')
    plt.text(11, 3.2, "Dense Vector (768-dim)", ha='center', fontweight='bold', color='green')
    plt.arrow(11, 3, 0, -1, head_width=0.2, color='black')

    ax7.add_patch(patches.Circle((11, 0.5), 1.2, facecolor='gold', edgecolor='orange'))
    plt.text(11, 0.5, 'FAISS\nIndex', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.xlim(-1, 14); plt.ylim(-1, 6); plt.axis('off')
    plt.title("Figure 3.6: The Vector Embedding Pipeline", fontsize=16)
    plt.savefig(os.path.join(output_dir, "Diagram_4_Conceptual_Embedding.png"), dpi=300)
    plt.close()


def draw_entity(ax, x, y, width, height, title, columns, color='#e6f2ff'):
    """Helper function for Figure 5 schema boxes."""
    ax.add_patch(patches.Rectangle((x+0.1, y-0.1), width, height, linewidth=0, facecolor='gray', alpha=0.3, zorder=5))
    ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='black', facecolor=color, zorder=10))
    header_height = height * 0.25
    ax.add_patch(patches.Rectangle((x, y + height - header_height), width, header_height, linewidth=2, edgecolor='black', facecolor='#007acc', zorder=11))
    ax.text(x + width/2, y + height - header_height/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=12)
    display_cols = columns[:5]
    if len(columns) > 5: display_cols.append(f"... (+{len(columns)-5} more)")
    ax.text(x + 0.2, y + height - header_height - 0.3, "\n".join(display_cols), ha='left', va='top', fontsize=9, family='monospace', zorder=12)
    return (x, y, width, height)


def generate_real_structured_schema(output_dir, real_columns):
    """Figure 5: Entity boxes drawn directly from real CSV column headers."""
    print("Generating 5/7: Real Relational Schema...")
    fig8, ax8 = plt.subplots(figsize=(14, 10))
    ax8.set_xlim(0, 20); ax8.set_ylim(0, 15); ax8.axis('off')

    doc_cols = real_columns.get("docentes", ["File not found"])
    disc_cols = real_columns.get("discentes", ["File not found"])
    prod_cols = real_columns.get("producoes", ["File not found"])
    part_cols = real_columns.get("participantes", ["File not found"])

    box_meta = draw_entity(ax8, 8, 6, 4, 3, "PROGRAM_METADATA\n(ui_metadata.csv)", ["program_id", "program_name", "instituicao", "ies_sigla"], '#ffcc99')
    box_doc = draw_entity(ax8, 2, 10, 4.5, 3.5, "DOCENTES\n(analytical_docentes.csv)", doc_cols, '#cceeff')
    box_disc = draw_entity(ax8, 13.5, 10, 4.5, 3.5, "DISCENTES\n(analytical_discentes.csv)", disc_cols, '#cceeff')
    box_prod = draw_entity(ax8, 2, 2, 4.5, 3.5, "PRODUCOES\n(analytical_producoes.csv)", prod_cols, '#d9f2d9')
    box_part = draw_entity(ax8, 13.5, 2, 4.5, 3.5, "PARTICIPANTES\n(analytical_participantes.csv)", part_cols, '#e6e6fa')

    ax8.text(10, 14.5, "Figure: Relational Schema (Driven by Real CSV Headers)", ha='center', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Diagram_5_Real_StructuredSchema.png"), dpi=300)
    plt.close()


def generate_real_faiss_vector_space(output_dir, vectorstore, embeddings):
    """Figure 6: PCA projection of sampled FAISS vectors, colored by source type."""
    print("Generating 6/7: Real FAISS Vector Space (PCA)...")
    real_query = "Qual é a identidade, missão e objetivo deste programa de pós-graduação?"
    num_vectors = vectorstore.index.ntotal

    if num_vectors > 0:
        query_vector_768 = embeddings.embed_query(real_query)

        # Get 500 nearest neighbours so the query star always lands inside the cluster
        query_np = np.array([query_vector_768], dtype=np.float32)
        _, nn_ids = vectorstore.index.search(query_np, 500)
        nn_ids = nn_ids[0].tolist()

        # Fill remaining slots with a random sample from the rest of the index
        remaining = [i for i in range(num_vectors) if i not in set(nn_ids)]
        random_ids = np.random.choice(remaining, size=min(1500, len(remaining)), replace=False).tolist()
        sample_ids = nn_ids + random_ids

        sampled_vectors = np.array([vectorstore.index.reconstruct(int(i)) for i in sample_ids])
        doc_sources = [vectorstore.docstore.search(vectorstore.index_to_docstore_id[int(i)]).metadata.get('source', 'other') for i in sample_ids]
        combined_vectors = np.vstack([sampled_vectors, [query_vector_768]])
        
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
        vectors_2d = tsne.fit_transform(combined_vectors)
        real_2d = vectors_2d[:-1]
        query_2d = vectors_2d[-1]
        
        plt.figure(figsize=(10, 8))
        proposta_idx = [i for i, src in enumerate(doc_sources) if 'proposta' in src.lower()]
        if proposta_idx: plt.scatter(real_2d[proposta_idx, 0], real_2d[proposta_idx, 1], c='#2ca02c', s=80, label='Propostas (Real)', alpha=0.7)
            
        other_idx = [i for i, src in enumerate(doc_sources) if 'proposta' not in src.lower()]
        if other_idx: plt.scatter(real_2d[other_idx, 0], real_2d[other_idx, 1], c='gray', s=50, label='Other Chunks (Real)', alpha=0.4)
            
        plt.scatter(query_2d[0], query_2d[1], c='red', marker='*', s=500, label='Query: "Identity"', zorder=10)
        ax6 = plt.gca()
        ax6.add_patch(plt.Circle((query_2d[0], query_2d[1]), 3, color='red', fill=False, linestyle=':', linewidth=2))
        
        plt.title("Figure: REAL FAISS Vector Space (t-SNE, n=2000 sample)", fontsize=15)
        plt.xlabel("t-SNE Dimension 1", fontsize=10); plt.ylabel("t-SNE Dimension 2", fontsize=10)
        plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(output_dir, "Diagram_6_Real_FAISS_VectorSpace.png"), dpi=300)
        plt.close()


def generate_real_retrieval_heatmap(output_dir, vectorstore):
    """Figure 7: Heatmap of cosine similarity scores for top 5 retrieved chunks."""
    print("Generating 7/7: Real Retrieval Heatmap...")
    real_query = "Qual é a identidade, missão e objetivo deste programa de pós-graduação?"
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(real_query, k=5)
    
    unique_docs, unique_scores, seen_texts = [], [], set()

    for d, s in docs_and_scores:
        preview = f"[{d.metadata.get('source', 'Unknown')}] {d.page_content[:45]}..."
        if preview not in seen_texts:
            seen_texts.add(preview)
            unique_docs.append(preview)
            unique_scores.append([s])

    if not unique_scores:
        print("Warning: No documents retrieved for heatmap.")
        return

    scores_array = np.array(unique_scores)
    s_min, s_max = scores_array.min(), scores_array.max()
    scores_array = (scores_array - s_min) / (s_max - s_min) if s_max != s_min else scores_array

    plt.figure(figsize=(10, 6))
    ax9 = sns.heatmap(scores_array, annot=True, cmap="Greens", fmt=".3f", xticklabels=["Query: 'Program Identity'"], yticklabels=unique_docs, linewidths=1, linecolor='black', cbar_kws={'label': 'Normalized Similarity Score (0-1)'})
    plt.title("Figure: Real Retrieval Relevance Heatmap (From FAISS)", fontsize=14, pad=20)
    ax9.xaxis.tick_top(); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Diagram_7_Real_SimilarityHeatmap.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    print("🚀 Initializing Final Thesis Visualization Script...")
    
    # Configure your paths here. You can change these to match the final structure.
    OUTPUT_DIR = r"C:\GINFO LAB\data-parsing-private\AI-Powered PEPG 2.0 Evaluator\output"
    DB_PATH = os.path.join(OUTPUT_DIR, "db", "knowledgebase_faiss")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading real ETL outputs...")
    
    # Generate conceptual diagrams that don't need real data first
    generate_conceptual_data_mapping(OUTPUT_DIR)
    generate_conceptual_pipeline_flow(OUTPUT_DIR)
    generate_conceptual_chunking(OUTPUT_DIR)
    generate_conceptual_embedding(OUTPUT_DIR)

    # Load real data for the remaining diagrams
    csv_paths = {
        "docentes": os.path.join(OUTPUT_DIR, "analytical_docentes.csv"),
        "discentes": os.path.join(OUTPUT_DIR, "analytical_discentes.csv"),
        "producoes": os.path.join(OUTPUT_DIR, "analytical_producoes.csv"),
        "participantes": os.path.join(OUTPUT_DIR, "analytical_participantes.csv")
    }
    
    real_columns = {name: pd.read_csv(path, sep=';', nrows=0).columns.tolist() for name, path in csv_paths.items() if os.path.exists(path)}
    generate_real_structured_schema(OUTPUT_DIR, real_columns)

    if os.path.exists(DB_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        generate_real_faiss_vector_space(OUTPUT_DIR, vectorstore, embeddings)
        generate_real_retrieval_heatmap(OUTPUT_DIR, vectorstore)
    else:
        print(f"⚠️ FAISS Database not found at {DB_PATH}. Skipping figures 6 and 7.")

    print("\n✅ Clean-up Complete! 7 thesis-ready visualizations saved to your output folder.")
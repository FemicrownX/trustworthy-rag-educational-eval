import os
import random
import pandas as pd

# ==========================================
# 1. SETUP & PATHS
# ==========================================
PROJECT_ROOT = r"C:\GINFO LAB\data-parsing-private"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "AI-Powered PEPG 2.0 Evaluator", "output")
GENERATED_100_PATH = os.path.join(OUTPUT_DIR, "Automated_Predicitive_and_Symbolic_100_Datasets.csv")
HUMAN_VERIFIED_SEMANTIC_PATH = os.path.join(OUTPUT_DIR, "Human_Verified_Semantic_Questions_50_Datasets.csv")
MASTER_DATASET_PATH = os.path.join(OUTPUT_DIR, "Master_Evaluation_Dataset_150.csv")

def build_master_dataset():
    random.seed(42)

    if os.path.exists(MASTER_DATASET_PATH):
        print(f"✅ Master Evaluation Dataset already exists, Skipping All Generation, Please wait!")
        df_master = pd.read_csv(MASTER_DATASET_PATH)
        print(f"📁 Loaded Existing Master File with {len(df_master)} total rows: {MASTER_DATASET_PATH}")
        return df_master

    print("Now Mapping IDs to Academic Institutions, Please wait!")
    df_lookup = pd.read_csv(os.path.join(OUTPUT_DIR, "ui_metadata.csv"))
    inst_map = dict(zip(df_lookup['program_id'].astype(str), df_lookup['instituicao']))

    columns_data = {
        "Case_ID": [], "Task_Type": [], "Program_ID": [], "Institution": [], 
        "Year_or_Cycle": [], "Question": [], "Ground_Truth_Answer": [], 
        "Source_File": [], "KPI_Subtype": [], "Official_Grade": []
    }

    case_counters = {"Symbolic": 1, "Prediction": 1}

    def add_row(task_type, prog_id, year, q, gt, src, kpi_sub, official_grade=""):
        case_id = f"{task_type[:3].upper()}_{case_counters[task_type]:03d}"
        case_counters[task_type] += 1
        real_inst = inst_map.get(str(prog_id), "Unknown Institution")
        columns_data["Case_ID"].append(case_id)
        columns_data["Task_Type"].append(task_type)
        columns_data["Program_ID"].append(prog_id)
        columns_data["Institution"].append(real_inst)
        columns_data["Year_or_Cycle"].append(year)
        columns_data["Question"].append(q)
        columns_data["Ground_Truth_Answer"].append(gt)
        columns_data["Source_File"].append(src)
        columns_data["KPI_Subtype"].append(kpi_sub)
        columns_data["Official_Grade"].append(official_grade)

    df_base_100 = None

    if os.path.exists(GENERATED_100_PATH):
        print("✅ 100-Row Dataset already exists, Skipping Generation and Loading Existing File, Please wait!")
        df_base_100 = pd.read_csv(GENERATED_100_PATH)
    else:
        print("Now Generating 100-Row Base Dataset with Academic Phrasing, Please wait")
        try:
            df_meta = pd.read_csv(os.path.join(OUTPUT_DIR, "ui_metadata.csv"))
            df_meta_graded = df_meta.dropna(subset=['grade'])
            for grade in [3.0, 4.0, 5.0, 6.0, 7.0]:
                grade_pool = df_meta_graded[df_meta_graded['grade'].astype(float) == grade]
                if not grade_pool.empty:
                    sample = grade_pool.sample(n=10, replace=True, random_state=42)
                    for i, (_, row) in enumerate(sample.iterrows()):
                        q = f"Considerando os indicadores de desempenho, qual a nota CAPES inferida para o Programa {row['program_id']} em {row['year']}?" if i >= 6 else \
                            f"Based on the provided performance indicators, what is the inferred CAPES Grade for Program {row['program_id']} for the {row['year']} cycle?"
                        add_row("Prediction", row['program_id'], row['year'], q, str(int(row['grade'])), "ui_metadata.csv", "Full_Inference", str(int(row['grade'])))

            def batch_sym(path, q_en, q_pt, subtype, n, n_pt):
                if not os.path.exists(path): return
                df = pd.read_csv(path, sep=';', encoding='utf-8')
                sample_group = df.groupby(['program_id', 'year']).size().reset_index(name='count')
                sample = sample_group.sample(n=min(n, len(sample_group)), random_state=42)
                for i, (_, row) in enumerate(sample.iterrows()):
                    q = q_pt.format(id=row['program_id'], y=row['year']) if i >= (n-n_pt) else q_en.format(id=row['program_id'], y=row['year'])
                    add_row("Symbolic", row['program_id'], row['year'], q, str(row['count']), os.path.basename(path), subtype)

            batch_sym(os.path.join(OUTPUT_DIR, "analytical_docentes.csv"), 
                      "What is the total number of permanent faculty members affiliated with Program {id} for the {y} academic year?", 
                      "Qual o total de docentes permanentes vinculados ao Programa {id} no ano de {y}?", 
                      "Faculty", 13, 3)

            batch_sym(os.path.join(OUTPUT_DIR, "analytical_discentes.csv"), 
                      "According to official records, what is the total student enrollment for Program {id} during the {y} cycle?", 
                      "De acordo com os registros oficiais, qual o total de discentes matriculados no Programa {id} durante o ciclo de {y}?", 
                      "Students", 13, 3)

            batch_sym(os.path.join(OUTPUT_DIR, "analytical_producoes.csv"), 
                      "How many scientific and intellectual publications were registered for Program {id} in the year {y}?", 
                      "Quantas produções científicas e intelectuais foram registradas para o Programa {id} no ano de {y}?", 
                      "Production", 12, 2)

            batch_sym(os.path.join(OUTPUT_DIR, "analytical_participantes.csv"), 
                      "What is the total count of external participants and international collaborators associated with Program {id} in {y}?", 
                      "Qual o total de participantes externos e colaboradores internacionais associados ao Programa {id} em {y}?", 
                      "International", 12, 2)

            df_base_100 = pd.DataFrame(columns_data)
            df_base_100.to_csv(GENERATED_100_PATH, index=False)
            print(f" Successfully Generated 100-row automated dataset: {GENERATED_100_PATH}")

        except FileNotFoundError as e:
            print(f"Oh X, File Error! A required CSV file was not found: {e}\nPlease check that all source files exist in the output directory.")
            raise
        except KeyError as e:
            print(f"Oh X, Column Error! An expected column is missing from one of the CSVs: {e}\nCheck that 'program_id', 'year', and 'grade' columns exist in your source files.")
            raise
        except ValueError as e:
            print(f"Oh X, Value Error! A data type issue occurred: {e}\nThis may be caused by unexpected values in the 'grade' column.")
            raise
        except Exception as e:
            print(f"Oh X, Generation Error: {e}")
            raise

    if df_base_100 is not None:
        print("\n Now Merging with Human-in-the-Loop Semantic Data, Please wait")
        if os.path.exists(HUMAN_VERIFIED_SEMANTIC_PATH):
            df_semantic = pd.read_csv(HUMAN_VERIFIED_SEMANTIC_PATH)

            sem_cols = ['Case_ID', 'Task_Type', 'Program_ID', 'Institution', 'Year_or_Cycle', 'Question', 'Ground_Truth_Answer', 'Reference_Context']
            base_cols = ['Case_ID', 'Task_Type', 'Program_ID', 'Institution', 'Year_or_Cycle', 'Question', 'Ground_Truth_Answer']

            df_semantic_clean = df_semantic[[col for col in sem_cols if col in df_semantic.columns]]
            df_base_clean = df_base_100[[col for col in base_cols if col in df_base_100.columns]]

            df_master = pd.concat([df_base_clean, df_semantic_clean], ignore_index=True)
            df_master['Reference_Context'] = df_master['Reference_Context'].fillna('')
            df_master.to_csv(MASTER_DATASET_PATH, index=False)
            print(f"✅ SUCCESS! Created Master Evaluation Dataset with {len(df_master)} total rows.\n📁 Master File: {MASTER_DATASET_PATH}")
            return df_master
        else:
            print(f"❌ Error: Could not find human-verified file at {HUMAN_VERIFIED_SEMANTIC_PATH}")

if __name__ == "__main__":
    build_master_dataset()
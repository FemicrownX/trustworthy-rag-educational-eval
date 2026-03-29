import os
import pandas as pd
from tqdm import tqdm

def safe_read_and_tag(file_path, year, program_id, separator=';'):
    """Reads a CSV, tags it with year and program ID, handles bad lines."""
    try:
        df = pd.read_csv(file_path, sep=separator, on_bad_lines='warn', encoding='utf-8')
        df['year'] = str(year)
        df['program_id'] = str(program_id)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def consolidate_analytical_data(base_data_dir, output_dir, programs_csv_path):
    """Parses raw CSVs and outputs the 4 master analytical CSVs."""
    joint_docentes, joint_discentes, joint_participantes, joint_producoes = [], [], [], []
    
    if not os.path.isdir(base_data_dir):
        raise FileNotFoundError(f"ERROR: Data directory '{base_data_dir}' not found.")

    print("\n--- 🔄 PHASE 1: DATA COLLATION ---")
    years = [d for d in os.listdir(base_data_dir) if d.isdigit()]
    
    for year in tqdm(years, desc="Processing Years"):
        year_dir = os.path.join(base_data_dir, year)
        for program_folder in [f.path for f in os.scandir(year_dir) if f.is_dir()]:
            pid = os.path.basename(program_folder)
            
            files_map = {
                'docentes.csv': joint_docentes, 
                'discentes.csv': joint_discentes,
                'participantes_externos.csv': joint_participantes, 
                'producoes.csv': joint_producoes
            }
            
            for fname, target_list in files_map.items():
                fpath = os.path.join(program_folder, fname)
                if os.path.exists(fpath):
                    df = safe_read_and_tag(fpath, year, pid)
                    if df is not None: 
                        target_list.append(df)

    print("\n--- 🔄 PHASE 2: MERGING INSTITUTIONAL METADATA ---")
    master_docentes = pd.concat(joint_docentes, ignore_index=True) if joint_docentes else pd.DataFrame()
    master_discentes = pd.concat(joint_discentes, ignore_index=True) if joint_discentes else pd.DataFrame()
    master_producoes = pd.concat(joint_producoes, ignore_index=True) if joint_producoes else pd.DataFrame()
    master_participantes = pd.concat(joint_participantes, ignore_index=True) if joint_participantes else pd.DataFrame()

    if os.path.exists(programs_csv_path):
        programs_df = pd.read_csv(programs_csv_path, sep=';')
        programs_df.rename(columns={'codigo': 'program_id', 'ies_nome': 'instituicao'}, inplace=True)
        programs_df['program_id'] = programs_df['program_id'].astype(str)
        merge_cols = programs_df[['program_id', 'ies_sigla', 'instituicao']]
        
        if not master_docentes.empty: master_docentes = master_docentes.merge(merge_cols, on='program_id', how='left')
        if not master_discentes.empty: master_discentes = master_discentes.merge(merge_cols, on='program_id', how='left')
        if not master_producoes.empty: master_producoes = master_producoes.merge(merge_cols, on='program_id', how='left')
        if not master_participantes.empty: master_participantes = master_participantes.merge(merge_cols, on='program_id', how='left')

    os.makedirs(output_dir, exist_ok=True)
    files_to_save = {
        "analytical_docentes.csv": master_docentes, 
        "analytical_discentes.csv": master_discentes,
        "analytical_producoes.csv": master_producoes, 
        "analytical_participantes.csv": master_participantes
    }

    for fname, df in files_to_save.items():
        fout = os.path.join(output_dir, fname)
        if not df.empty:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv(fout, index=False, sep=';')
            print(f"💾 Saved {fname} ({len(df)} rows)")

    return master_docentes, master_discentes, master_producoes, master_participantes
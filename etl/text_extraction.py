import os
from tqdm import tqdm

def extract_proposals(base_data_dir):
    """Extracts all proposta.txt files into a dictionary mapping pid_year -> text."""
    joint_proposals = {}
    
    if not os.path.isdir(base_data_dir):
        return joint_proposals

    years = [d for d in os.listdir(base_data_dir) if d.isdigit()]
    
    for year in tqdm(years, desc="Extracting Proposals"):
        year_dir = os.path.join(base_data_dir, year)
        for program_folder in [f.path for f in os.scandir(year_dir) if f.is_dir()]:
            pid = os.path.basename(program_folder)
            ppath = os.path.join(program_folder, 'proposta.txt')
            
            if os.path.exists(ppath):
                with open(ppath, 'r', encoding='utf-8') as f:
                    joint_proposals[f"{pid}_{year}"] = f.read()
                    
    print(f"✅ Loaded {len(joint_proposals)} Proposal TXTs.")
    return joint_proposals
import os
import re
import pandas as pd
from tqdm import tqdm
from evaluation.retrieval_metrics import compute_retrieval_at_k
from evaluation.generation_metrics import calculate_bert_scores, calculate_rouge_l

OUTPUT_DIR = os.path.join(r"C:\GINFO LAB\data-parsing-private", "AI-Powered PEPG 2.0 Evaluator", "output")

def score_logic(ans, gt, task):
    try:
        if task == 'Symbolic':
            gt_clean = re.sub(r'\D', '', str(gt)).strip()
            return 1.0 if gt_clean and bool(re.search(rf'\b{re.escape(gt_clean)}\b', str(ans))) else 0.0

        if task == 'Prediction':
            match = re.search(r'(?:Predicted Grade|Nota Prevista)[*:\s]*([0-7])', str(ans), re.IGNORECASE)
            pred = float(match.group(1)) if match else -1.0
            if pred == -1.0: return 0.0 
            return 1.0 if abs(pred - float(gt)) <= 1.0 else 0.0
    except:
        return 0.0
    return 0.0

def run_scoring(model_label):
    output_file = os.path.join(OUTPUT_DIR, f"FINAL_{model_label}_RESULTS_150.csv")
    is_baseline = "Baseline" in model_label

    if not os.path.exists(output_file): return None
    df_res = pd.read_csv(output_file)

    for col in ['Context_Recall', 'Context_Precision', 'Retrieval_F1', 'Groundedness', 'ROUGE_L', 'BERT']: df_res[col] = 0.0
    df_res['Generated_Report'] = df_res['Generated_Report'].fillna("No response").astype(str)
    df_res['Ground_Truth_Answer'] = df_res['Ground_Truth_Answer'].fillna("No ground truth").astype(str)
    df_res['Reference_Context'] = df_res.get('Reference_Context', pd.Series([""] * len(df_res))).fillna("").astype(str)
    df_res['Retrieved_Context'] = df_res.get('Retrieved_Context', pd.Series([""] * len(df_res))).fillna("").astype(str)

    sem_mask = df_res['Task_Type'] == 'Semantic'

    # BERTScore
    if sem_mask.sum() > 0:
        df_res.loc[sem_mask, 'BERT'] = calculate_bert_scores(df_res.loc[sem_mask, 'Generated_Report'].tolist(), df_res.loc[sem_mask, 'Ground_Truth_Answer'].tolist())

    # ROUGE-L
    for real_idx, row in df_res[sem_mask].iterrows():
        df_res.at[real_idx, 'ROUGE_L'] = calculate_rouge_l(row['Generated_Report'], row['Ground_Truth_Answer'])

    if not is_baseline:
        # Retrieval Metrics
        for real_idx, row in df_res[sem_mask].iterrows():
            gold = str(row['Reference_Context']).strip() or str(row['Ground_Truth_Answer'])
            ctx = str(row['Retrieved_Context'])
            gold_words = set(re.findall(r'\b\w{4,}\b', gold.lower()))
            ctx_words  = set(re.findall(r'\b\w{4,}\b', ctx.lower()))
            overlap    = gold_words.intersection(ctx_words)

            df_res.at[real_idx, 'Context_Recall'] = len(overlap) / len(gold_words) if gold_words else 1.0
            df_res.at[real_idx, 'Context_Precision'] = len(overlap) / len(ctx_words) if ctx_words else 0.0

        df_res['Retrieval_F1'] = df_res.apply(lambda r: (2 * r['Context_Recall'] * r['Context_Precision'] / (r['Context_Recall'] + r['Context_Precision'])) if (r['Context_Recall'] + r['Context_Precision']) > 0 else 0, axis=1)

        # Groundedness
        df_res.loc[sem_mask, 'Groundedness'] = calculate_bert_scores(df_res.loc[sem_mask, 'Generated_Report'].tolist(), df_res.loc[sem_mask, 'Retrieved_Context'].tolist())

    df_res.to_csv(output_file, index=False)

    pred_rows = df_res[df_res['Task_Type'] == 'Prediction']
    fmt_fail  = pred_rows['Generated_Report'].apply(lambda a: not bool(re.search(r'(?:Predicted Grade|Nota Prevista)[*:\s]*([0-7])', str(a), re.IGNORECASE))).sum()

    sym_acc = df_res[df_res['Task_Type'] == 'Symbolic'].apply(lambda r: score_logic(r['Generated_Report'], r['Ground_Truth_Answer'], 'Symbolic'), axis=1).mean() * 100
    pred_acc = df_res[df_res['Task_Type'] == 'Prediction'].apply(lambda r: score_logic(r['Generated_Report'], r['Ground_Truth_Answer'], 'Prediction'), axis=1).mean() * 100

    at_k_df = None
    if not is_baseline:
        at_k_df = compute_retrieval_at_k(df_res[sem_mask], k_values=[1, 3, 5, 10])

    return {
        "Model": model_label,
        "Symbolic_Acc (%)": round(sym_acc, 2),
        "Pred_Format_Fail": int(fmt_fail),
        "Prediction_Acc (%)": round(pred_acc, 2),
        "BERTScore": round(df_res[sem_mask]['BERT'].mean(), 4),
        "ROUGE-L": round(df_res[sem_mask]['ROUGE_L'].mean(), 4),
        "Recall": round(df_res.loc[sem_mask, 'Context_Recall'].mean(), 4) if not is_baseline else "N/A",
        "Precision": round(df_res.loc[sem_mask, 'Context_Precision'].mean(), 4) if not is_baseline else "N/A",
        "Retrieval_F1": round(df_res.loc[sem_mask, 'Retrieval_F1'].mean(), 4) if not is_baseline else "N/A",
        "Groundedness": round(df_res[sem_mask]['Groundedness'].mean(), 4) if not is_baseline else "N/A",
    }, at_k_df

if __name__ == "__main__":
    models_to_score = ["DeepSeek", "DeepSeek_Baseline", "Gemini_2.5", "Gemini_Baseline", "GPT-4o", "GPT-4o_Baseline"]
    all_summaries = []
    
    for label in models_to_score:
        result = run_scoring(label)
        if result: all_summaries.append(result[0])

    if all_summaries:
        df_final = pd.DataFrame(all_summaries)
        df_final.to_csv(os.path.join(OUTPUT_DIR, "FINAL_BENCHMARK_SUMMARY.csv"), index=False)
        print("✅ Summary saved successfully.")
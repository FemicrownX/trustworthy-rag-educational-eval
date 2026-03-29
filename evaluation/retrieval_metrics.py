import re
import pandas as pd

def compute_retrieval_at_k(df_sem, k_values=[1, 3, 5, 10]):
    results = []
    for k in k_values:
        recalls, precisions = [], []
        for _, row in df_sem.iterrows():
            gold = str(row['Reference_Context']).strip()
            if not gold: gold = str(row['Ground_Truth_Answer'])
            ctx_k = str(row['Retrieved_Context'])[:k * 1000]

            gold_words  = set(re.findall(r'\b\w{4,}\b', gold.lower()))
            ctx_k_words = set(re.findall(r'\b\w{4,}\b', ctx_k.lower()))
            overlap     = gold_words.intersection(ctx_k_words)

            recall    = len(overlap) / len(gold_words)  if gold_words  else 1.0
            precision = len(overlap) / len(ctx_k_words) if ctx_k_words else 0.0
            recalls.append(recall)
            precisions.append(precision)

        avg_r = sum(recalls) / len(recalls)
        avg_p = sum(precisions) / len(precisions)
        avg_f = sum([2 * r * p / (r + p) if (r + p) > 0 else 0 for r, p in zip(recalls, precisions)]) / len(recalls)

        results.append({"k": k, "Recall@k": round(avg_r, 4), "Precision@k": round(avg_p, 4), "F1@k": round(avg_f, 4)})
    return pd.DataFrame(results)
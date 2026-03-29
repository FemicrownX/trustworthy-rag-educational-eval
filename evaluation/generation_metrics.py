from bert_score import score as bert_score
from rouge_score import rouge_scorer

r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_bert_scores(preds, refs):
    _, _, F1 = bert_score(preds, refs, lang="pt", model_type="bert-base-multilingual-cased", verbose=False)
    return F1.tolist()

def calculate_rouge_l(pred, ref):
    return r_scorer.score(str(ref), str(pred))['rougeL'].fmeasure
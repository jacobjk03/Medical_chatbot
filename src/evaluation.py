# evaluation.py
from rouge_score import rouge_scorer
from bert_score import score

def compute_rouge(reference: str, candidate: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def compute_bertscore(reference: str, candidate: str):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item()
    }

def evaluate_response(reference: str, candidate: str):
    rouge = compute_rouge(reference, candidate)
    bert = compute_bertscore(reference, candidate)
    return {**rouge, **bert}

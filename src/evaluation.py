import evaluate

rouge = evaluate.load("rouge")

def compute_rouge(preds, refs):
    return rouge.compute(predictions=preds, references=refs, use_stemmer=True)

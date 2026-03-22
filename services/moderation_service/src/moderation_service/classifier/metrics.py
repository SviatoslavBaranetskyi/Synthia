import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(predictions, labels):
    probs = 1 / (1 + np.exp(-predictions))

    preds = probs > 0.5

    f1 = f1_score(labels, preds, average="micro")

    roc = roc_auc_score(labels, probs, average="micro")

    return {
        "f1": f1,
        "roc_auc": roc,
    }

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(probs, labels):
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(0.05, 0.95, 50)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1),
    }

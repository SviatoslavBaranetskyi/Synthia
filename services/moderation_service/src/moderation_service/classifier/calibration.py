import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from moderation_service.classifier.threshold import find_best_threshold
from moderation_service.classifier.model import ToxicClassifier


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"].cpu().numpy().squeeze()

            all_logits.append(logits)
            all_labels.append(labels.squeeze())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    probs = sigmoid(logits)

    return probs, labels


def compute_full_metrics(probs, labels):
    roc = roc_auc_score(labels, probs)

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }


def save_calibration(save_dir, threshold_data, metrics):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "threshold.json"), "w") as f:
        json.dump(threshold_data, f, indent=2)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def run_calibration(model_path, dataloader, save_dir, device="cpu"):
    model = ToxicClassifier.from_pretrained(model_path)

    probs, labels = evaluate_model(model, dataloader, device)

    metrics = compute_full_metrics(probs, labels)

    threshold_data = find_best_threshold(probs, labels)

    save_calibration(save_dir, threshold_data, metrics)

    return {
        "metrics": metrics,
        "threshold": threshold_data,
    }
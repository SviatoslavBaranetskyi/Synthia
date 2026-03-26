import json
import os

import torch
from moderation_service.classifier.model import ToxicClassifier
from transformers import DistilBertTokenizer


class ToxicityEngine:
    def __init__(self, model_path: str):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = ToxicClassifier.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        threshold_path = os.path.join(model_path, "threshold.json")

        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                self.threshold = json.load(f)["threshold"]
        else:
            self.threshold = 0.5

    def predict(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)

        prob = probs.squeeze().item()
        label = int(prob >= self.threshold)

        if label == 1:
            reason = "ml: high toxicity probability"
        else:
            reason = "ml: low toxicity probability"

        return {
            "probability": prob,
            "label": label,
            "threshold": self.threshold,
            "reason": reason,
        }

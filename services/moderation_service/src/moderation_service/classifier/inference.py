import torch
from moderation_service.classifier.model import ToxicClassifier
from transformers import DistilBertTokenizer


class ToxicityEngine:
    def __init__(self, model_path: str):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.model = ToxicClassifier(num_labels=6)

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.model.eval()

    def predict(self, text: str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**tokens)

            probs = torch.sigmoid(logits)

        return probs.tolist()

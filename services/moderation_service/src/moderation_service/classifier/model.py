import os
import torch
import torch.nn as nn
from transformers import DistilBertModel


class ToxicClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()

        self.num_labels = num_labels

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Linear(768, num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {
            "loss": loss,
            "logits": logits,
        }

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))

        config = {
            "num_labels": self.num_labels,
            "model_name": "distilbert-base-uncased",
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        config = torch.load(
            os.path.join(load_directory, "config.pt"),
            weights_only=True,
        )

        model = cls(num_labels=config["num_labels"])

        state_dict = torch.load(
            os.path.join(load_directory, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)

        return model

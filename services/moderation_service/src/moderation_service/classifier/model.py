import torch.nn as nn
from transformers import DistilBertModel


class ToxicClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier(pooled_output)

        return logits

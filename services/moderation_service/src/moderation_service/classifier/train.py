from transformers import (
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

from moderation_service.ml.dataset import load_toxic_dataset
from moderation_service.ml.model import ToxicClassifier


def main():

    dataset = load_toxic_dataset()

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = ToxicClassifier(num_labels=6)

    training_args = TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
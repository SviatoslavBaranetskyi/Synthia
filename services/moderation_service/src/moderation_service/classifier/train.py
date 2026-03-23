import torch
from datasets import DatasetDict
from moderation_service.classifier.dataset import load_toxic_dataset, preprocess_dataset
from moderation_service.classifier.model import ToxicClassifier
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, Trainer, TrainingArguments


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()

    preds = (preds > 0).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")

    return {"accuracy": acc, "f1": f1}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_toxic_dataset()

    if "eval" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(
            {
                "train": split["train"],
                "eval": split["test"],
            }
        )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = preprocess_dataset(dataset, tokenizer)

    model = ToxicClassifier(num_labels=1)
    model.to(device)

    training_args = TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_model_path = "models/final"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model and tokenizer saved to {final_model_path}")


if __name__ == "__main__":
    print("CUDA available inside train:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)
    main()

import torch
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer

from moderation_service.classifier.dataset import (
    load_toxic_dataset,
    preprocess_dataset,
)
from moderation_service.classifier.calibration import run_calibration


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_toxic_dataset()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    dataset = preprocess_dataset(dataset, tokenizer)

    eval_loader = DataLoader(
        dataset["test"],
        batch_size=32,
        shuffle=False,
    )

    result = run_calibration(
        model_path="models/final",
        dataloader=eval_loader,
        save_dir="models/final",
        device=device,
    )

    print("Calibration results:")
    print(result)


if __name__ == "__main__":
    main()
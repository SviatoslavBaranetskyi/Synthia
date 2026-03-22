from datasets import load_dataset


def load_toxic_dataset():
    dataset = load_dataset("SetFit/toxic_conversations_50k")

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    return dataset


def preprocess_dataset(dataset, tokenizer):
    def preprocess(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

        tokens["labels"] = [[float(label)] for label in batch["label"]]

        return tokens

    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return dataset
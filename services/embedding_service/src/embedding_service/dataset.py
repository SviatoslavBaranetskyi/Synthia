from datasets import load_dataset


def load_toxic_texts(limit: int = 10000):
    dataset = load_dataset("SetFit/toxic_conversations_50k")

    texts = []

    for item in dataset["train"]:
        if item["label"] == 1:
            texts.append(item["text"])

        if len(texts) >= limit:
            break

    return texts
from datasets import load_dataset


def load_toxic_dataset():
    dataset = load_dataset("jigsaw_toxicity_pred")

    return dataset
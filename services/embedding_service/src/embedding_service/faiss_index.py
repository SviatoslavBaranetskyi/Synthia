import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build(self, texts):
        self.texts = texts

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "texts.pkl"), "rb") as f:
            self.texts = pickle.load(f)

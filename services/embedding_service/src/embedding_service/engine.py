import numpy as np
import torch
from embedding_service.faiss_index import EmbeddingIndex
from embedding_service.schemas import SimilarExample
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, index_path: str):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.index = EmbeddingIndex()
        self.index.load(index_path)

    def search(self, text: str, top_k: int = 5):
        query_emb = self.model.encode([text], convert_to_numpy=True)

        import faiss

        faiss.normalize_L2(query_emb)

        scores, indices = self.index.index.search(query_emb, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            results.append(
                SimilarExample(
                    text=self.index.texts[idx],
                    score=float(score),
                )
            )

        return results

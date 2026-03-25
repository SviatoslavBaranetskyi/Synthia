from embedding_service.dataset import load_toxic_texts
from embedding_service.faiss_index import EmbeddingIndex


def main():
    texts = load_toxic_texts(limit=20000)

    index = EmbeddingIndex()
    index.build(texts)

    index.save("models/embedding_index")

    print("Index built and saved")


if __name__ == "__main__":
    main()

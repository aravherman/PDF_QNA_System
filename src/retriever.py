import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index from document embeddings.
    """

    embeddings = np.asarray(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_faiss(index, question_embedding, chunks, top_k=3):
    """
    Search for the most relevant PDF chunks.
    """

    question_embedding = np.asarray(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        chunk = chunks[index_number].copy()

        chunk["distance"] = float(distance)

        results.append(chunk)

    return results
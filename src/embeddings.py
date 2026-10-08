import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    """
    Load the sentence transformer model only once.
    """
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, chunks):
    """
    Create embeddings for all text chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def create_question_embedding(model, question):
    """
    Create an embedding for the user's question.
    """

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    return embedding
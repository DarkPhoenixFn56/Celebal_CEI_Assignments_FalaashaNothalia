"""
vector_store.py
----------------
Handles embedding generation and the local FAISS vector database.

Pipeline stages covered by this file:

    Generate Embeddings -> Store in FAISS
    Question Embedding -> Similarity Search -> Top-K Chunks

Three public functions are exposed:
    - get_embedding_model()
    - create_vector_store(chunks)
    - retrieve_documents(vector_store, question)
"""

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import EMBEDDING_MODEL_NAME, TOP_K_RESULTS


class VectorStoreError(Exception):
    """Raised when embedding generation or FAISS operations fail."""


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence-embedding model used to convert
    text into numerical vectors.

    The model runs entirely locally (CPU), so no external API call
    or API key is needed for this step.

    Returns:
        A configured HuggingFaceEmbeddings instance.

    Raises:
        VectorStoreError: If the embedding model fails to load
            (e.g. no internet connection on first-time download).
    """
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as error:
        raise VectorStoreError(
            "Failed to load the embedding model. Check your internet "
            f"connection (required on first run to download the model). "
            f"Details: {error}"
        ) from error


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Generate embeddings for all text chunks and store them in a
    local FAISS vector index.

    Args:
        chunks: List of chunked Document objects to embed and store.

    Returns:
        A FAISS vector store containing all chunk embeddings.

    Raises:
        VectorStoreError: If chunks are empty or FAISS index creation
            fails for any reason.
    """
    if not chunks:
        raise VectorStoreError("Cannot create a vector store from zero chunks.")

    try:
        embedding_model = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embedding_model)
    except VectorStoreError:
        raise
    except Exception as error:
        raise VectorStoreError(
            f"Failed to build the FAISS vector store. Details: {error}"
        ) from error

    return vector_store


def retrieve_documents(
    vector_store: FAISS, question: str
) -> List[Tuple[Document, float]]:
    """
    Convert the user's question into an embedding and run a
    similarity search against the FAISS index to find the most
    relevant chunks.

    Args:
        vector_store: The FAISS vector store built from the
            uploaded document(s).
        question: The user's natural-language question.

    Returns:
        A list of (Document, similarity_score) tuples for the
        top-K most relevant chunks, ordered by relevance.

    Raises:
        VectorStoreError: If the question is empty or the similarity
            search fails.
    """
    if not question or not question.strip():
        raise VectorStoreError("Question cannot be empty.")

    if vector_store is None:
        raise VectorStoreError(
            "No document has been indexed yet. Please upload a PDF first."
        )

    try:
        results = vector_store.similarity_search_with_score(
            question, k=TOP_K_RESULTS
        )
    except Exception as error:
        raise VectorStoreError(
            f"Similarity search failed. Details: {error}"
        ) from error

    if not results:
        raise VectorStoreError("No relevant chunks were found for this question.")

    return results

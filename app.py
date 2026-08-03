"""
app.py
------
Streamlit UI for the Document Question Answering System.

This file orchestrates the full RAG pipeline end-to-end:

    Upload PDF -> Extract Text -> Split into Chunks -> Generate
    Embeddings -> Store in FAISS -> User Question -> Question
    Embedding -> Similarity Search -> Top-3 Chunks -> Gemini ->
    Final Answer

It intentionally contains no business logic of its own -- every
pipeline step is delegated to document_loader.py, vector_store.py,
and rag_pipeline.py. This file's only job is UI and orchestration.
"""

import os
import tempfile

import streamlit as st

from config import NO_ANSWER_MESSAGE, TOP_K_RESULTS, GEMINI_API_KEY
from document_loader import load_documents, split_documents, DocumentLoadError
from vector_store import create_vector_store, retrieve_documents, VectorStoreError
from rag_pipeline import generate_answer, RAGPipelineError


def initialize_session_state() -> None:
    """Set up Streamlit session state variables on first run."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "indexed_file_names" not in st.session_state:
        st.session_state.indexed_file_names = []
    if "answer" not in st.session_state:
        st.session_state.answer = None
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []


def save_uploaded_files_to_disk(uploaded_files) -> list[str]:
    """
    Persist Streamlit's in-memory uploaded files to a temporary
    directory on disk, since PyPDFLoader requires a file path.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        List of absolute file paths where the PDFs were saved.
    """
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def process_uploaded_pdfs(uploaded_files) -> None:
    """
    Run the full indexing pipeline (load -> split -> embed -> store)
    on newly uploaded PDF files and save the result in session state.
    """
    with st.spinner("Reading and indexing your document(s)... this may take a moment."):
        try:
            pdf_paths = save_uploaded_files_to_disk(uploaded_files)
            documents = load_documents(pdf_paths)
            chunks = split_documents(documents)
            vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.indexed_file_names = [f.name for f in uploaded_files]
            st.success(
                f"Indexed {len(uploaded_files)} document(s) into "
                f"{len(chunks)} chunks. You can now ask questions."
            )
        except DocumentLoadError as error:
            st.error(f"Document error: {error}")
        except VectorStoreError as error:
            st.error(f"Indexing error: {error}")
        except Exception as error:
            st.error(f"An unexpected error occurred while indexing: {error}")


def handle_question(question: str) -> None:
    """
    Run the full retrieval + generation pipeline for a user question
    and store the result in session state for display.
    """
    if st.session_state.vector_store is None:
        st.warning("Please upload and index a PDF document before asking a question.")
        return

    if not question or not question.strip():
        st.warning("Please enter a question before clicking Ask.")
        return

    with st.spinner("Searching document and generating answer..."):
        try:
            retrieved_chunks = retrieve_documents(
                st.session_state.vector_store, question
            )
            answer = generate_answer(question, retrieved_chunks)

            st.session_state.answer = answer
            st.session_state.retrieved_chunks = retrieved_chunks
        except VectorStoreError as error:
            st.error(f"Retrieval error: {error}")
        except RAGPipelineError as error:
            st.error(f"Answer generation error: {error}")
        except Exception as error:
            st.error(f"An unexpected error occurred: {error}")


def render_sidebar() -> None:
    """Render the sidebar containing upload controls and app info."""
    with st.sidebar:
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Index Document(s)", type="primary"):
                process_uploaded_pdfs(uploaded_files)

        if st.session_state.indexed_file_names:
            st.markdown("**Currently indexed:**")
            for name in st.session_state.indexed_file_names:
                st.markdown(f"- {name}")

        st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.answer = None
            st.session_state.retrieved_chunks = []
            st.rerun()

        st.divider()
        st.caption(
            "This app answers questions using ONLY the content of the "
            "documents you upload. It will not use outside knowledge."
        )

        if not GEMINI_API_KEY:
            st.warning(
                "GEMINI_API_KEY environment variable is not set. "
                "Answer generation will fail until it is configured."
            )


def render_answer_section() -> None:
    """Render the final answer, if one has been generated."""
    st.subheader("💬 Answer")
    if st.session_state.answer:
        st.write(st.session_state.answer)
    else:
        st.info("Ask a question to see the answer here.")


def render_retrieved_chunks_section() -> None:
    """Render the retrieved chunks and their source page numbers."""
    st.subheader(f"🔍 Retrieved Chunks (Top {TOP_K_RESULTS})")

    if not st.session_state.retrieved_chunks:
        st.caption("Retrieved chunks will appear here after you ask a question.")
        return

    for i, (chunk, score) in enumerate(st.session_state.retrieved_chunks, start=1):
        page_number = chunk.metadata.get("page", "unknown")
        source_file = chunk.metadata.get("source_file", "document")
        with st.expander(
            f"Chunk {i} — {source_file}, Page {page_number} "
            f"(similarity score: {score:.4f})"
        ):
            st.write(chunk.page_content)


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Document QA - RAG System",
        page_icon="📄",
        layout="wide",
    )

    initialize_session_state()

    st.title("📄 Document Question Answering System")
    st.caption("Retrieval-Augmented Generation (RAG) over your own PDF documents")

    render_sidebar()

    st.divider()

    question = st.text_input(
        "Ask a question about your uploaded document(s):",
        placeholder="e.g. What is the main conclusion of this report?",
    )

    if st.button("Ask", type="primary"):
        handle_question(question)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_answer_section()
    with col2:
        render_retrieved_chunks_section()


if __name__ == "__main__":
    main()

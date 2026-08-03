"""
document_loader.py
-------------------
Handles everything related to turning a raw uploaded PDF file into
clean, chunked text ready for embedding.

Pipeline stage covered by this file:

    Upload PDF -> Extract Text -> Split into Chunks

Two public functions are exposed:
    - load_documents(pdf_paths)
    - split_documents(documents)
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoadError(Exception):
    """Raised when a PDF cannot be loaded or contains no usable text."""


def load_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Load one or more PDF files from disk and extract their raw text.

    Each page of each PDF becomes a separate LangChain `Document`
    object, with page number metadata attached so we can later show
    users exactly where an answer came from.

    Args:
        pdf_paths: List of absolute file paths to PDF files on disk.

    Returns:
        A list of Document objects, one per extracted page, across
        all provided PDFs.

    Raises:
        DocumentLoadError: If a file is missing, corrupted, not a
            valid PDF, or contains no extractable text.
    """
    if not pdf_paths:
        raise DocumentLoadError("No PDF files were provided.")

    all_documents: List[Document] = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise DocumentLoadError(f"File not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise DocumentLoadError(
                f"Unsupported file type for '{os.path.basename(pdf_path)}'. "
                "Only PDF files are supported."
            )

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
        except Exception as error:
            # Covers corrupted files, unreadable/encrypted PDFs, etc.
            raise DocumentLoadError(
                f"Could not read '{os.path.basename(pdf_path)}'. "
                f"The file may be corrupted or invalid. Details: {error}"
            ) from error

        if not pages:
            raise DocumentLoadError(
                f"'{os.path.basename(pdf_path)}' appears to be empty."
            )

        # Attach a clean source filename to each page's metadata so
        # later stages (retrieval, UI) can display it to the user.
        for page in pages:
            page.metadata["source_file"] = os.path.basename(pdf_path)

        all_documents.extend(pages)

    # Check that at least some real text was extracted across all
    # pages. Scanned/image-only PDFs will pass the load step above
    # but yield empty text here.
    total_characters = sum(len(doc.page_content.strip()) for doc in all_documents)
    if total_characters == 0:
        raise DocumentLoadError(
            "No extractable text was found in the uploaded PDF(s). "
            "The file may be a scanned image without a text layer."
        )

    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into smaller overlapping text chunks.

    Large pages are broken into chunks of roughly CHUNK_SIZE
    characters, with CHUNK_OVERLAP characters shared between
    consecutive chunks so that context near chunk boundaries is not
    lost. This is required because embedding models and LLM context
    windows work best with reasonably sized, focused text blocks.

    Args:
        documents: List of Document objects (typically one per page).

    Returns:
        A list of smaller Document chunks, each retaining the
        original page's metadata (page number, source file).
    """
    if not documents:
        raise DocumentLoadError("No documents available to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise DocumentLoadError("Document splitting produced no chunks.")

    return chunks

"""
config.py
---------
Central configuration file for the Document QA RAG system.

Every tunable value used across the project (file paths, model names,
chunking parameters, retrieval settings) lives here. This keeps the
rest of the codebase free of "magic numbers" and hardcoded strings,
and makes the pipeline easy to tune from a single location.
"""

import os

# --------------------------------------------------------------------
# Directory paths
# --------------------------------------------------------------------

# Root directory of the project (folder this file lives in)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Folder where uploaded PDFs are temporarily stored before processing
DATA_DIR: str = os.path.join(BASE_DIR, "data")

# Folder where the FAISS index is persisted to disk
VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "vector_store")

# Ensure these folders exist at import time so the app never fails
# because a directory is missing.
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Text splitting configuration
# --------------------------------------------------------------------

# Maximum number of characters in a single text chunk
CHUNK_SIZE: int = 1000

# Number of overlapping characters between consecutive chunks.
# Overlap helps preserve context that would otherwise be cut in half
# at a chunk boundary.
CHUNK_OVERLAP: int = 150

# --------------------------------------------------------------------
# Embedding model configuration
# --------------------------------------------------------------------

# A small, fast, well-regarded sentence-embedding model from
# HuggingFace. Runs locally on CPU, no API calls required.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------
# Retrieval configuration
# --------------------------------------------------------------------

# Number of most-similar chunks to retrieve for a given question
TOP_K_RESULTS: int = 3

# --------------------------------------------------------------------
# Gemini LLM configuration
# --------------------------------------------------------------------

# The Gemini API key must be set as an environment variable.
# Never hardcode API keys directly into source code.
GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")

# Gemini model used for final answer generation
GEMINI_MODEL_NAME: str = "gemini-3.6-flash"

# Maximum tokens Gemini is allowed to generate in a single answer
GEMINI_MAX_OUTPUT_TOKENS: int = 1024

# Temperature controls randomness. Kept low because we want factual,
# grounded answers rather than creative ones.
GEMINI_TEMPERATURE: float = 0.2

# --------------------------------------------------------------------
# Fallback message
# --------------------------------------------------------------------

# Message shown when the retrieved context does not contain enough
# information to answer the user's question.
NO_ANSWER_MESSAGE: str = (
    "The uploaded document does not contain enough information "
    "to answer this question."
)

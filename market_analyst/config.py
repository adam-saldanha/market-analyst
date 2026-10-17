"""Central settings loaded from .env so every module reads config the same way."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CHUNK_SIZE = 1200      # characters
CHUNK_OVERLAP = 200
TOP_K = 8              # chunks handed to each agent

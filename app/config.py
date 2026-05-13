from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Chunking settings (tune later)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval settings (tune later)
TOP_K = 4

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Safety limits
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB per file
MAX_CHUNKS_PER_INGEST = 5000          # cap embedding spend per ingest

DEFAULT_COLLECTION = "rag_docs"

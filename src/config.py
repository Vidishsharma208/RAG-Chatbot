import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "data"
CHUNKS_DIR = "chunks"
VECTORDB_DIR = "vectordb"

PDF_PATH = os.path.join(DATA_DIR, "AI Training Document.pdf")
CHUNKS_PATH = os.path.join(CHUNKS_DIR, "chunks.json")
FAISS_INDEX_PATH = os.path.join(VECTORDB_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTORDB_DIR, "metadata.json")

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3-8b-8192"

# Chunking settings
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40

# Retrieval
TOP_K = 4

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
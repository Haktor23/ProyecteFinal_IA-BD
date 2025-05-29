# config_loader.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
    JSON_OUTPUT = os.getenv("JSON_OUTPUT", "./chunks.json")
    
    # Elasticsearch Configuration
    ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
    ES_INDEX = os.getenv("ES_INDEX", "contaminacion_ollama")
    ES_USERNAME = os.getenv("ES_USERNAME", "")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "")
    
    # Ollama Configuration
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    VISION_MODEL = os.getenv("VISION_MODEL", "llava:latest")
    EMBED_MODEL_OLLAMA = os.getenv("EMBED_MODEL_OLLAMA", "all-minilm:latest")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
    OLLAMA_NUM_RETRIES = int(os.getenv("OLLAMA_NUM_RETRIES", "2"))
    OLLAMA_RETRY_BACKOFF = float(os.getenv("OLLAMA_RETRY_BACKOFF", "0.5"))
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "3")) # For sliding_window of pages
    OVERLAP = int(os.getenv("OVERLAP", "1")) # For sliding_window of pages
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024")) # For SimpleTextSplitter
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "256")) # For SimpleTextSplitter
    
    EMBEDDING_DIM = None  # Will be detected automatically from Ollama
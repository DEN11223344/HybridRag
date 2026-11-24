import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False

if HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        app_host: str = "0.0.0.0"
        app_port: int = 8000
        debug: bool = False
        cpu_num_threads: int = 4

        # Storage paths
        upload_dir: Path = Path("data/uploads")
        extracted_dir: Path = Path("data/extracted")
        processed_dir: Path = Path("data/processed")

        # Qdrant Vector Database
        qdrant_host: str = "localhost"
        qdrant_port: int = 6333
        qdrant_collection: str = "pdf_hybrid_rag"

        # Models (Local CPU Inference)
        dense_model: str = "BAAI/bge-small-en-v1.5"
        dense_dim: int = 384
        splade_model: str = "prithivida/Splade_PP_en_v1"
        reranker_model: str = "BAAI/bge-reranker-base"

        # Ingestion & Chunking parameters
        target_chunk_size: int = 512
        chunk_overlap: int = 100
        min_chunk_size: int = 100
        max_chunk_size: int = 700
        ocr_char_threshold: int = 40
        batch_size: int = 16

        # Retrieval & Reranker parameters
        bm25_top_k: int = 20
        splade_top_k: int = 20
        dense_top_k: int = 20
        rrf_k: int = 60
        rerank_top_k: int = 5

        # Generation / LLM Configuration
        llm_provider: str = "mock"
        llm_model: str = "mock-llm"
        llm_base_url: Optional[str] = None
        llm_api_key: Optional[str] = None

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

    settings = Settings()
else:
    class FallbackSettings:
        def __init__(self):
            self.app_host = os.getenv("APP_HOST", "0.0.0.0")
            self.app_port = int(os.getenv("APP_PORT", "8000"))
            self.debug = os.getenv("DEBUG", "false").lower() == "true"
            self.cpu_num_threads = int(os.getenv("CPU_NUM_THREADS", "4"))
            self.upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
            self.extracted_dir = Path(os.getenv("EXTRACTED_DIR", "data/extracted"))
            self.processed_dir = Path(os.getenv("PROCESSED_DIR", "data/processed"))
            self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "pdf_hybrid_rag")
            self.dense_model = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
            self.dense_dim = int(os.getenv("DENSE_DIM", "384"))
            self.splade_model = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
            self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
            self.target_chunk_size = int(os.getenv("TARGET_CHUNK_SIZE", "512"))
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
            self.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "100"))
            self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "700"))
            self.ocr_char_threshold = int(os.getenv("OCR_CHAR_THRESHOLD", "40"))
            self.batch_size = int(os.getenv("BATCH_SIZE", "16"))
            self.bm25_top_k = int(os.getenv("BM25_TOP_K", "20"))
            self.splade_top_k = int(os.getenv("SPLADE_TOP_K", "20"))
            self.dense_top_k = int(os.getenv("DENSE_TOP_K", "20"))
            self.rrf_k = int(os.getenv("RRF_K", "60"))
            self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))
            self.llm_provider = os.getenv("LLM_PROVIDER", "mock")
            self.llm_model = os.getenv("LLM_MODEL", "mock-llm")
            self.llm_base_url = os.getenv("LLM_BASE_URL")
            self.llm_api_key = os.getenv("LLM_API_KEY")

    settings = FallbackSettings()

for p in [settings.upload_dir, settings.extracted_dir, settings.processed_dir]:
    p.mkdir(parents=True, exist_ok=True)

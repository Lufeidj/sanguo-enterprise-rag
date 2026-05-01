from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "sanguo_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8008"))
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() == "true"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip() or LLM_BASE_URL
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip() or LLM_API_KEY
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "30"))
VECTOR_RRF_K = int(os.getenv("VECTOR_RRF_K", "60"))
KEYWORD_RRF_K = int(os.getenv("KEYWORD_RRF_K", "60"))
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
# 语义重排前最多对多少条候选做 embedding（控制成本与评测耗时）
RERANK_CANDIDATE_CAP = int(os.getenv("RERANK_CANDIDATE_CAP", "48"))

# Day08 文本默认路径
DEFAULT_SANGUO_PATH = (
    PROJECT_ROOT.parent.parent.parent
    / "Day08"
    / "Day08"
    / "wolin_learn-master-master"
    / "rag_examples"
    / "02_document_chunking"
    / "资料"
    / "《三国演义》.txt"
)

"""Cau hinh tap trung cho he thong BotAI.

Module nay gom cac hang so, duong dan va bien moi truong duoc app, retrieval, auth va export su dung.
"""

import os

LLM_MODEL = os.getenv("BOTAI_LLM_MODEL", "qwen2.5:latest")
EMBEDDING_MODEL_PATH = os.getenv("BOTAI_EMBEDDING_MODEL_PATH", "./models--BAAI--bge-m3")
RERANKER_MODEL_PATH = (
    "./models--BAAI--bge-reranker-large/"
    "snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312"
)
RERANKER_V2_M3_MODEL_PATH = (
    "./models--BAAI--bge-reranker-v2-m3/"
    "snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
DB_DIR = os.getenv("BOTAI_CHROMA_DIR", "chroma_db")
BM25_PATH = os.getenv("BOTAI_BM25_PATH", "bm25_retriever.pkl")
APP_DB_PATH = os.getenv("BOTAI_APP_DB_PATH", "chatbot_app.db")
DEFAULT_DOCUMENT_FILENAME = "tailieu.pdf"
DOCUMENTS_DIR = "storage/documents"
EXPORTS_DIR = "storage/exports"
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".docx", ".xlsx", ".pptx"}
MAX_UPLOAD_BYTES = int(os.getenv("BOTAI_MAX_UPLOAD_MB", "30")) * 1024 * 1024
MAX_URL_BYTES = int(os.getenv("BOTAI_MAX_URL_MB", "2")) * 1024 * 1024
RATE_LIMIT_RULES = {
    "chat": (60, 60),
    "stream": (60, 60),
    "login": (10, 300),
    "upload": (12, 300),
    "url": (12, 300),
    "export": (30, 300),
}

VECTOR_TOP_K = 10
BM25_TOP_K = 12
RERANK_TOP_N = int(os.getenv("BOTAI_RERANK_TOP_N", "6"))
FINAL_TOP_K = 3
FUSION_MODE = os.getenv("BOTAI_FUSION_MODE", "rrf").strip().lower()
FUSION_VECTOR_WEIGHT = float(os.getenv("BOTAI_FUSION_VECTOR_WEIGHT", "1.0"))
FUSION_BM25_WEIGHT = float(os.getenv("BOTAI_FUSION_BM25_WEIGHT", "1.2"))
MAX_CONTEXT_CHARS = 7000
NO_ANSWER_MESSAGE = "Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp."
EMPTY_QUESTION_MESSAGE = "Vui lòng nhập câu hỏi."
GREETING_MESSAGE = "Xin chào! Mình có thể hỗ trợ bạn tra cứu thông tin trong tài liệu."
OUT_OF_SCOPE_MESSAGE = "Mình chỉ hỗ trợ hỏi đáp dựa trên tài liệu đã được cung cấp."
PROMPT_INJECTION_MESSAGE = (
    "Mình không thể thực hiện yêu cầu bỏ qua luật hệ thống hoặc trả lời ngoài phạm vi tài liệu."
)
INDEX_NOT_READY_MESSAGE = (
    "Lỗi: Cơ sở dữ liệu tài liệu chưa được tạo. Hãy chạy `python ingest.py` trước."
)
TABLE_DOC_TYPES = {"table", "excel_table", "ocr_table"}
BOTAI_CACHE_ENABLED = os.getenv("BOTAI_CACHE_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
BOTAI_CITATION_VERIFIER_ENABLED = os.getenv("BOTAI_CITATION_VERIFIER_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
BOTAI_CITATION_VERIFIER_STRICT = os.getenv("BOTAI_CITATION_VERIFIER_STRICT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
DOCUMENT_COLLECTION_VERSION = os.getenv("BOTAI_DOCUMENT_COLLECTION_VERSION", "tailieu-v1")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
RERANKER_ENABLED = os.getenv("BOTAI_RERANKER_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RERANKER_BACKEND = os.getenv("BOTAI_RERANKER_BACKEND", "cross_encoder").strip().lower()
RERANKER_CONFIGURED_MODEL = os.getenv(
    "BOTAI_RERANKER_MODEL_PATH",
    RERANKER_V2_M3_MODEL_PATH if RERANKER_BACKEND in {"flag", "flagembedding", "bge-v2-m3"} else RERANKER_MODEL_PATH,
)

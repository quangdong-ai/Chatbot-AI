"""Khoi tao LLM, embedding, reranker va index runtime cho BotAI.

Tach rieng phan tai model de app.py chi con tap trung vao API/routes.
"""

import os
import pickle
import warnings
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

from botai.core.config import *

def _resolve_torch_device() -> str:
    configured = os.getenv("BOTAI_TORCH_DEVICE", "auto").strip().lower()
    if configured in {"cuda", "cpu"}:
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_hf_snapshot_path(path: str) -> str:
    model_path = Path(path)
    if (model_path / "config.json").exists():
        return str(model_path)
    refs_main = model_path / "refs" / "main"
    snapshots = model_path / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = snapshots / revision
        if (snapshot_path / "config.json").exists():
            return str(snapshot_path)
    if snapshots.exists():
        for snapshot_path in snapshots.iterdir():
            if (snapshot_path / "config.json").exists():
                return str(snapshot_path)
    return str(model_path)


RESOLVED_EMBEDDING_MODEL_PATH = _resolve_hf_snapshot_path(EMBEDDING_MODEL_PATH)


class FlagEmbeddingReranker:
    def __init__(self, model_path: str, device: str):
        from FlagEmbedding import FlagReranker

        use_fp16 = device == "cuda"
        self.model = FlagReranker(model_path, use_fp16=use_fp16)

    def predict(self, pairs: list[list[str]]):
        return self.model.compute_score(pairs)


def _load_reranker(model_path: str, device: str):
    resolved_path = _resolve_hf_snapshot_path(model_path)
    backend = RERANKER_BACKEND
    if backend in {"flag", "flagembedding", "bge-v2-m3"}:
        print(f"Loading FlagEmbedding reranker: {resolved_path}")
        return FlagEmbeddingReranker(resolved_path, device)
    print(f"Loading CrossEncoder reranker: {resolved_path}")
    return CrossEncoder(resolved_path, max_length=512, device=device)


TORCH_DEVICE = _resolve_torch_device()

print("Initializing AI models, please wait...")
print(f"Using torch device for embeddings/reranker: {TORCH_DEVICE}")
print(f"Reranker enabled: {RERANKER_ENABLED}")
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0,
    num_ctx=4096,
    num_predict=420,
)

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=RESOLVED_EMBEDDING_MODEL_PATH,
        model_kwargs={"device": TORCH_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )
except RuntimeError as exc:
    if TORCH_DEVICE != "cuda":
        raise
    print(f"CUDA embedding load failed, falling back to CPU: {exc}")
    TORCH_DEVICE = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=RESOLVED_EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )

reranker = None
if RERANKER_ENABLED:
    print(f"Loading Reranker Model ({RERANKER_BACKEND}): {RERANKER_CONFIGURED_MODEL}")
    try:
        reranker = _load_reranker(RERANKER_CONFIGURED_MODEL, TORCH_DEVICE)
    except RuntimeError as exc:
        if TORCH_DEVICE != "cuda":
            raise
        print(f"CUDA reranker load failed, falling back to CPU: {exc}")
        torch.cuda.empty_cache()
        reranker = _load_reranker(RERANKER_CONFIGURED_MODEL, "cpu")
else:
    print("Reranker skipped for fast LLM-focused mode.")

vector_db = None
vector_retriever = None
bm25_retriever = None
all_indexed_docs: list[Document] = []

if os.path.exists(DB_DIR) and os.path.exists(BM25_PATH):
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_TOP_K})

    with open(BM25_PATH, "rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = BM25_TOP_K
        all_indexed_docs = list(getattr(bm25_retriever, "docs", []) or [])


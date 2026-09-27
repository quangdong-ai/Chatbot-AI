"""Quan ly trang thai index, rebuild Chroma/BM25 va diagnostics."""

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

import gc
import os
import pickle
import warnings

from langchain_chroma import Chroma

__all__ = ['_ocr_runtime_status', '_index_runtime_stats', '_load_indexes', '_release_indexes', '_rebuild_indexes', '_update_document_status', '_run_rebuild_job']

def _ocr_runtime_status() -> dict:
    enabled = os.getenv("BOTAI_OCR_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
    pillow_installed = importlib.util.find_spec("PIL") is not None
    pytesseract_installed = importlib.util.find_spec("pytesseract") is not None
    tesseract_path = shutil.which("tesseract")
    version = ""
    error = ""
    requested_language = os.getenv("BOTAI_OCR_LANG", "vie+eng")
    requested_languages = [lang for lang in re.split(r"[+,]", requested_language) if lang]
    installed_languages: list[str] = []
    missing_languages: list[str] = []
    effective_languages: list[str] = []
    if pytesseract_installed:
        try:
            import pytesseract

            version = str(pytesseract.get_tesseract_version())
            installed_languages = sorted(pytesseract.get_languages(config=""))
            installed_set = set(installed_languages)
            effective_languages = [lang for lang in requested_languages if lang in installed_set]
            missing_languages = [lang for lang in requested_languages if lang not in installed_set]
            if not effective_languages:
                if "eng" in installed_set:
                    effective_languages = ["eng"]
                elif installed_languages:
                    effective_languages = [installed_languages[0]]
        except Exception as exc:
            error = str(exc)
    missing = []
    if not pillow_installed:
        missing.append("pillow")
    if not pytesseract_installed:
        missing.append("pytesseract")
    if not tesseract_path:
        missing.append("Tesseract OCR engine")
    missing.extend(f"OCR language: {lang}" for lang in missing_languages)
    return {
        "enabled": enabled,
        "ready": (
            enabled
            and pillow_installed
            and pytesseract_installed
            and bool(tesseract_path)
            and bool(effective_languages)
            and not error
        ),
        "pillow_installed": pillow_installed,
        "pytesseract_installed": pytesseract_installed,
        "tesseract_path": tesseract_path or "",
        "version": version,
        "language": requested_language,
        "effective_language": "+".join(effective_languages),
        "installed_languages": installed_languages,
        "missing": missing,
        "error": error,
    }


def _index_runtime_stats() -> dict:
    by_type: dict[str, int] = {}
    by_bot: dict[str, int] = {}
    ocr_needs_review = 0
    for doc in all_indexed_docs:
        metadata = doc.metadata or {}
        doc_type = str(metadata.get("type") or "unknown")
        bot_id = str(metadata.get("bot_id") or 1)
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
        by_bot[bot_id] = by_bot.get(bot_id, 0) + 1
        if metadata.get("ocr_needs_review"):
            ocr_needs_review += 1
    return {
        "total_chunks": len(all_indexed_docs),
        "by_type": by_type,
        "by_bot": by_bot,
        "has_ocr_chunks": by_type.get("ocr_text", 0) > 0,
        "ocr_needs_review": ocr_needs_review,
    }








_ensure_default_admin_user()
_ensure_default_bot()
_ensure_default_document()




def _load_indexes() -> None:
    global vector_db, vector_retriever, bm25_retriever, all_indexed_docs, response_cache
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
        response_cache.clear()


def _release_indexes() -> None:
    global vector_db, vector_retriever, bm25_retriever, all_indexed_docs
    vector_db = None
    vector_retriever = None
    bm25_retriever = None
    all_indexed_docs = []
    gc.collect()


def _rebuild_indexes() -> None:
    import ingest

    _release_indexes()
    ingest.ingest_document()
    _load_indexes()


def _update_document_status(document_id: int, status: str, error_message: str | None = None) -> None:
    with _db_connect() as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, _now_iso(), document_id),
        )
        conn.commit()


def _run_rebuild_job(document_id: int | None = None) -> None:
    with INGEST_LOCK:
        try:
            if document_id is not None:
                _update_document_status(document_id, "processing", "")
            _rebuild_indexes()
            if document_id is not None:
                _update_document_status(document_id, "done", "")
        except Exception as exc:
            if document_id is not None:
                _update_document_status(document_id, "error", str(exc))
            print(f"Background ingest failed: {exc}")

"""Debug retrieval va streaming Ollama helpers."""

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

import json
import urllib.request
from typing import Iterable

from botai.core.config import LLM_MODEL
from botai.answering.response_service import *
from botai.retrieval.retrieval_service import *
from botai.answering.rule_service import *

__all__ = ['_debug_retrieve', '_chunk_text_for_stream', '_ollama_stream']

def _debug_retrieve(question: str, top_k: int = 8, bot_id: int | None = None) -> dict:
    cleaned_question = question.strip()
    intent = classify_intent(cleaned_question)
    if not vector_retriever or not bm25_retriever:
        return {
            "error": "indexes_not_ready",
            "message": "Co so du lieu tai lieu chua duoc tao. Hay chay python ingest.py truoc.",
        }

    bot = _load_bot(bot_id)
    selected_bot_id = int(bot.get("id") or 1)
    v_docs, b_docs = _retrieve_documents(cleaned_question, selected_bot_id)
    channel_by_key = {}
    for doc in b_docs:
        channel_by_key.setdefault(_doc_key(doc), set()).add("bm25")
    for doc in v_docs:
        channel_by_key.setdefault(_doc_key(doc), set()).add("vector")

    combined_docs = _fuse_documents(v_docs, b_docs)
    rerank_scores: dict[tuple, float] = {}
    if reranker and combined_docs:
        try:
            rerank_docs = combined_docs[:RERANK_TOP_N]
            remaining_docs = combined_docs[RERANK_TOP_N:]
            pairs = [[cleaned_question, doc.page_content[:1200]] for doc in rerank_docs]
            scores = reranker.predict(pairs)
            rerank_scores = {
                _doc_key(doc): float(score)
                for score, doc in zip(scores, rerank_docs)
            }
            scored_docs = list(zip(scores, rerank_docs))
            scored_docs.sort(key=lambda item: item[0], reverse=True)
            combined_docs = [doc for _, doc in scored_docs] + remaining_docs
        except Exception as exc:
            print(f"Debug reranker failed, using hybrid order: {exc}")

    table_lookup_docs = _filter_docs_by_mentioned_source(
        cleaned_question,
        _combine_documents(
            _combine_documents(b_docs[:BM25_TOP_K], v_docs[:VECTOR_TOP_K]),
            _table_docs_for_bot(selected_bot_id),
        ),
    )
    direct_docs = _combine_documents(
        _combine_documents(b_docs[:BM25_TOP_K], v_docs[:VECTOR_TOP_K]),
        _metadata_candidate_docs(cleaned_question, selected_bot_id),
    )
    direct_answer = (
        _answer_total_training_time_fact(cleaned_question, selected_bot_id)
        or _answer_time_range_definition(cleaned_question, direct_docs)
        or _answer_definition(cleaned_question, direct_docs)
        or _answer_from_ocr_table(cleaned_question, table_lookup_docs)
        or _answer_from_table_summaries(cleaned_question, table_lookup_docs)
    )
    combined_docs = _rank_documents_for_question(cleaned_question, combined_docs)
    final_docs = combined_docs[: max(1, min(top_k, 20))]
    items = []
    for rank, doc in enumerate(final_docs, start=1):
        metadata = doc.metadata or {}
        key = _doc_key(doc)
        source_line = _table_source_line(doc, final_docs) if metadata.get("type") in TABLE_DOC_TYPES else _source_line_for_doc(doc)
        items.append(
            {
                "rank": rank,
                "source": metadata.get("source", "tailieu.pdf"),
                "pdf_page": metadata.get("pdf_page", ""),
                "type": metadata.get("type", "text"),
                "region": metadata.get("region", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "article": metadata.get("article", ""),
                "clause": metadata.get("clause", ""),
                "appendix": metadata.get("appendix", ""),
                "section_title": metadata.get("section_title", ""),
                "channel": "+".join(sorted(channel_by_key.get(key, {"hybrid"}))),
                "source_score": round(_source_score(cleaned_question, cleaned_question, doc), 4),
                "rerank_score": round(rerank_scores[key], 4) if key in rerank_scores else None,
                "source_line": source_line,
                "preview": _preview_text(doc.page_content, 900),
            }
        )

    return {
        "question": cleaned_question,
        "bot": {"id": selected_bot_id, "name": bot.get("name", "")},
        "intent": intent,
        "fusion_mode": FUSION_MODE,
        "retrieval_relevance": round(_retrieval_relevance_score(cleaned_question, final_docs), 4),
        "direct_answer": _preview_text(direct_answer or "", 500),
        "items": items,
    }


def _chunk_text_for_stream(text: str, size: int = 24) -> Iterable[str]:
    text = str(text or "")
    for start in range(0, len(text), size):
        yield text[start:start + size]


def _ollama_stream(prompt: str) -> Iterable[str]:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 420,
        },
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            data = json.loads(raw_line.decode("utf-8"))
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done"):
                break

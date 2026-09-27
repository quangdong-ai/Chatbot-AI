"""Public API and chat/stream routes."""

from fastapi import APIRouter

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

router = APIRouter()

@router.get("/api/v1/health")
async def api_v1_health():
    return {
        "ok": True,
        "model": LLM_MODEL,
        "torch_device": TORCH_DEVICE,
        "vector_ready": bool(core.vector_retriever),
        "bm25_ready": bool(core.bm25_retriever),
        "documents_indexed": len(core.all_indexed_docs),
        "cache_entries": len(core.response_cache),
        "index": _index_runtime_stats(),
        "ocr": _ocr_runtime_status(),
    }


@router.get("/api/v1/bots")
async def api_v1_bots():
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, is_active
            FROM bots
            WHERE is_active = 1
            ORDER BY id ASC
            """
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.get("/api/v1/documents")
async def api_v1_documents(bot_id: int | None = None):
    values = []
    where = ""
    if bot_id:
        bot = _load_bot(bot_id)
        selected_bot_id = int(bot.get("id") or 1)
        where = "WHERE documents.bot_id = ?"
        values.append(selected_bot_id)
    with _db_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT documents.id, documents.original_name, documents.file_type,
                   documents.bot_id, bots.name AS bot_name, documents.status,
                   documents.updated_at,
                   CASE WHEN documents.filename = ? THEN 1 ELSE 0 END AS protected
            FROM documents
            LEFT JOIN bots ON bots.id = documents.bot_id
            {where}
            ORDER BY documents.id DESC
            """,
            (DEFAULT_DOCUMENT_FILENAME, *values),
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.post("/api/v1/chat")
async def api_v1_chat(request: ApiChatRequest, http_request: Request):
    limited = _rate_limit_json(http_request, "chat")
    if limited:
        return limited
    started = time.perf_counter()
    bot = _load_bot(request.bot_id)
    selected_bot_id = int(bot.get("id") or 1)
    message = request.message.strip()
    if not message:
        return {
            "ok": False,
            "answer": EMPTY_QUESTION_MESSAGE,
            "sources": [],
            "session_id": request.session_id or "",
            "bot_id": selected_bot_id,
            "has_sources": False,
        }
    intent = classify_intent(message)
    if intent == "greeting":
        answer = GREETING_MESSAGE
        _save_chat_message(request.session_id or "", message, answer, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
        return {
            "ok": True,
            "answer": answer,
            "sources": [],
            "session_id": request.session_id or "",
            "bot_id": selected_bot_id,
            "intent": intent,
            "has_sources": False,
        }
    if intent == "out_of_scope":
        answer = OUT_OF_SCOPE_MESSAGE
        _save_chat_message(request.session_id or "", message, answer, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
        return {
            "ok": True,
            "answer": answer,
            "sources": [],
            "session_id": request.session_id or "",
            "bot_id": selected_bot_id,
            "intent": intent,
            "has_sources": False,
        }
    if intent == "prompt_injection":
        answer = PROMPT_INJECTION_MESSAGE
        _save_chat_message(request.session_id or "", message, answer, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
        return {
            "ok": True,
            "answer": answer,
            "sources": [],
            "session_id": request.session_id or "",
            "bot_id": selected_bot_id,
            "intent": intent,
            "has_sources": False,
        }
    response = await chat(request, http_request)
    if not isinstance(response, dict):
        return response
    raw_response = str(response.get("response", ""))
    answer, sources = _split_answer_sources(raw_response)
    payload = {
        "ok": True,
        "answer": answer,
        "sources": _parse_source_items(sources),
        "session_id": request.session_id or "",
        "bot_id": selected_bot_id,
        "intent": intent,
        "has_sources": bool(sources),
    }
    if request.include_raw_response:
        payload["raw_response"] = raw_response
    return payload


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    def generate() -> Iterable[str]:
        started = time.perf_counter()
        session_id = request.session_id or ""
        user_msg = request.message.strip()
        bot = _load_bot(request.bot_id)
        selected_bot_id = int(bot.get("id") or 1)
        intent = "document_qa"

        def finish(response: str, save_intent: str = intent) -> Iterable[str]:
            for chunk in _chunk_text_for_stream(response):
                yield chunk
            _save_chat_message(
                session_id,
                user_msg,
                response,
                save_intent,
                int((time.perf_counter() - started) * 1000),
                selected_bot_id,
            )

        try:
            ok, rate_message = _rate_limit(http_request, "stream")
            if not ok:
                yield from finish(rate_message, "rate_limited")
                return

            if not user_msg:
                yield from finish(EMPTY_QUESTION_MESSAGE, intent)
                return

            intent = classify_intent(user_msg)
            if intent == "prompt_injection":
                yield from finish(
                    PROMPT_INJECTION_MESSAGE,
                    intent,
                )
                return

            if intent == "greeting":
                yield from finish(
                    GREETING_MESSAGE,
                    intent,
                )
                return

            if intent == "out_of_scope":
                yield from finish(
                    OUT_OF_SCOPE_MESSAGE,
                    intent,
                )
                return

            if not core.vector_retriever or not core.bm25_retriever:
                yield from finish(
                    INDEX_NOT_READY_MESSAGE,
                    intent,
                )
                return

            retrieval_question = rewrite_question(user_msg, session_id)
            cache_key = f"bot:{selected_bot_id}:{_normalize_cache_key(retrieval_question)}"
            if BOTAI_CACHE_ENABLED and cache_key in core.response_cache:
                yield from finish(core.response_cache[cache_key], f"{intent}:cache")
                return

            v_docs, b_docs = _retrieve_documents(retrieval_question, selected_bot_id)
            direct_docs = _combine_documents(
                _combine_documents(b_docs[:BM25_TOP_K], v_docs[:VECTOR_TOP_K]),
                _metadata_candidate_docs(retrieval_question, selected_bot_id),
            )
            table_lookup_docs = _filter_docs_by_mentioned_source(
                retrieval_question,
                _combine_documents(direct_docs, _table_docs_for_bot(selected_bot_id)),
            )
            direct_response = (
                _answer_total_training_time_fact(retrieval_question, selected_bot_id)
                or _answer_scope_or_coverage(retrieval_question, direct_docs, selected_bot_id)
                or _answer_time_range_definition(retrieval_question, direct_docs)
                or _answer_definition(retrieval_question, direct_docs)
                or _answer_from_ocr_table(retrieval_question, table_lookup_docs)
                or _answer_from_table_summaries(retrieval_question, table_lookup_docs)
                or _answer_training_form(retrieval_question, direct_docs, selected_bot_id)
                or _answer_policy_rules(retrieval_question, direct_docs)
                or _answer_completion_requirement_scenario(retrieval_question, direct_docs)
                or _answer_from_high_confidence_sentence(retrieval_question, direct_docs)
            )
            if direct_response:
                if BOTAI_CACHE_ENABLED:
                    core.response_cache[cache_key] = direct_response
                yield from finish(direct_response, intent)
                return

            combined_docs = _fuse_documents(v_docs, b_docs)

            if reranker and combined_docs:
                try:
                    rerank_docs = combined_docs[:RERANK_TOP_N]
                    remaining_docs = combined_docs[RERANK_TOP_N:]
                    pairs = [[user_msg, doc.page_content[:1200]] for doc in rerank_docs]
                    scores = reranker.predict(pairs)
                    scored_docs = list(zip(scores, rerank_docs))
                    scored_docs.sort(key=lambda item: item[0], reverse=True)
                    combined_docs = [doc for _, doc in scored_docs] + remaining_docs
                except Exception as exc:
                    print(f"Reranker failed, using hybrid order: {exc}")

            combined_docs = _rank_documents_for_question(retrieval_question, combined_docs)
            final_docs = combined_docs[:FINAL_TOP_K]
            if _retrieval_relevance_score(retrieval_question, final_docs) < 0.12:
                yield from finish(NO_ANSWER_MESSAGE, intent)
                return

            table_answer = (
                _answer_from_table_summaries(retrieval_question, _combine_documents(final_docs, table_lookup_docs))
                or _answer_from_ocr_table(retrieval_question, _combine_documents(final_docs, table_lookup_docs))
                or _answer_training_form(
                    retrieval_question,
                    _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
                    selected_bot_id,
                )
                or _answer_policy_rules(
                    retrieval_question,
                    _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
                )
                or _answer_completion_requirement_scenario(
                    retrieval_question,
                    _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
                )
                or _answer_from_high_confidence_sentence(
                    retrieval_question,
                    _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
                )
            )
            if table_answer:
                if BOTAI_CACHE_ENABLED:
                    core.response_cache[cache_key] = table_answer
                yield from finish(table_answer, intent)
                return

            context_text = _format_context(final_docs)
            if not context_text:
                yield from finish(NO_ANSWER_MESSAGE, intent)
                return

            final_prompt = _format_prompt(context_text, retrieval_question, bot)
            raw_answer = "".join(_ollama_stream(final_prompt))
            formatted_response = _format_final_response(retrieval_question, raw_answer, final_docs)
            if "-----------------------------------" not in formatted_response:
                formatted_response = NO_ANSWER_MESSAGE
            for chunk in _chunk_text_for_stream(formatted_response):
                yield chunk

            if BOTAI_CACHE_ENABLED and NO_ANSWER_MESSAGE not in formatted_response:
                core.response_cache[cache_key] = formatted_response
            _save_chat_message(
                session_id,
                user_msg,
                formatted_response,
                intent,
                int((time.perf_counter() - started) * 1000),
                selected_bot_id,
            )
        except Exception as exc:
            error_response = f"Lỗi hệ thống: {exc}"
            yield error_response

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@router.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    started = time.perf_counter()
    session_id = request.session_id or ""
    bot = _load_bot(request.bot_id)
    selected_bot_id = int(bot.get("id") or 1)
    intent = "document_qa"
    try:
        limited = _rate_limit_json(http_request, "chat")
        if limited:
            return limited
        user_msg = request.message.strip()
        if not user_msg:
            return {"response": EMPTY_QUESTION_MESSAGE}

        intent = classify_intent(user_msg)
        if intent == "prompt_injection":
            response = PROMPT_INJECTION_MESSAGE
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        if intent == "greeting":
            response = GREETING_MESSAGE
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        if intent == "out_of_scope":
            response = OUT_OF_SCOPE_MESSAGE
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        if not core.vector_retriever or not core.bm25_retriever:
            return {
                "response": (
                    INDEX_NOT_READY_MESSAGE
                )
            }

        retrieval_question = rewrite_question(user_msg, session_id)
        cache_key = f"bot:{selected_bot_id}:{_normalize_cache_key(retrieval_question)}"
        if BOTAI_CACHE_ENABLED and cache_key in core.response_cache:
            response = core.response_cache[cache_key]
            _save_chat_message(session_id, user_msg, response, f"{intent}:cache", int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        v_docs, b_docs = _retrieve_documents(retrieval_question, selected_bot_id)
        direct_docs = _combine_documents(
            _combine_documents(b_docs[:BM25_TOP_K], v_docs[:VECTOR_TOP_K]),
            _metadata_candidate_docs(retrieval_question, selected_bot_id),
        )

        total_training_answer = _answer_total_training_time_fact(retrieval_question, selected_bot_id)
        if total_training_answer:
            response = total_training_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        scope_answer = _answer_scope_or_coverage(retrieval_question, direct_docs, selected_bot_id)
        if scope_answer:
            response = scope_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        definition_answer = _answer_definition(retrieval_question, direct_docs)
        time_range_answer = _answer_time_range_definition(retrieval_question, direct_docs)
        if time_range_answer:
            response = time_range_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        if definition_answer:
            response = definition_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        table_lookup_docs = _filter_docs_by_mentioned_source(
            retrieval_question,
            _combine_documents(direct_docs, _table_docs_for_bot(selected_bot_id)),
        )
        table_answer = (
            _answer_from_table_summaries(retrieval_question, table_lookup_docs)
            or _answer_from_ocr_table(retrieval_question, table_lookup_docs)
            or _answer_training_form(retrieval_question, direct_docs, selected_bot_id)
            or _answer_policy_rules(retrieval_question, direct_docs)
            or _answer_completion_requirement_scenario(retrieval_question, direct_docs)
            or _answer_from_high_confidence_sentence(retrieval_question, direct_docs)
        )
        if table_answer:
            response = table_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        combined_docs = _fuse_documents(v_docs, b_docs)

        if reranker and combined_docs:
            try:
                rerank_docs = combined_docs[:RERANK_TOP_N]
                remaining_docs = combined_docs[RERANK_TOP_N:]
                pairs = [[user_msg, doc.page_content[:1200]] for doc in rerank_docs]
                scores = reranker.predict(pairs)
                scored_docs = list(zip(scores, rerank_docs))
                scored_docs.sort(key=lambda item: item[0], reverse=True)
                combined_docs = [doc for _, doc in scored_docs] + remaining_docs
            except Exception as exc:
                print(f"Reranker failed, using hybrid order: {exc}")

        combined_docs = _rank_documents_for_question(retrieval_question, combined_docs)
        final_docs = combined_docs[:FINAL_TOP_K]
        if _retrieval_relevance_score(retrieval_question, final_docs) < 0.12:
            response = NO_ANSWER_MESSAGE
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        table_answer = (
            _answer_from_table_summaries(retrieval_question, _combine_documents(final_docs, table_lookup_docs))
            or _answer_from_ocr_table(retrieval_question, _combine_documents(final_docs, table_lookup_docs))
            or _answer_training_form(
                retrieval_question,
                _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
                selected_bot_id,
            )
            or _answer_policy_rules(
                retrieval_question,
                _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
            )
            or _answer_completion_requirement_scenario(
                retrieval_question,
                _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
            )
            or _answer_from_high_confidence_sentence(
                retrieval_question,
                _combine_documents(final_docs, b_docs[:BM25_TOP_K]),
            )
        )
        if table_answer:
            response = table_answer
            if BOTAI_CACHE_ENABLED:
                core.response_cache[cache_key] = response
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        context_text = _format_context(final_docs)
        if not context_text:
            response = NO_ANSWER_MESSAGE
            _save_chat_message(session_id, user_msg, response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
            return {"response": response}

        final_prompt = _format_prompt(context_text, retrieval_question, bot)
        response = llm.invoke(final_prompt)
        formatted_response = _format_final_response(retrieval_question, response, final_docs)
        if "-----------------------------------" not in formatted_response:
            formatted_response = NO_ANSWER_MESSAGE
        if BOTAI_CACHE_ENABLED and NO_ANSWER_MESSAGE not in formatted_response:
            core.response_cache[cache_key] = formatted_response
        _save_chat_message(session_id, user_msg, formatted_response, intent, int((time.perf_counter() - started) * 1000), selected_bot_id)
        return {"response": formatted_response}

    except Exception as e:
        return {"response": f"Lỗi hệ thống: {str(e)}"}

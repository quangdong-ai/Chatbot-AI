"""Evaluation case management and benchmark routes."""

from fastapi import APIRouter

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

router = APIRouter()

def _eval_case_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "expected_contains": json.loads(row["expected_contains_json"] or "[]"),
        "expected_source_contains": json.loads(row["expected_source_contains_json"] or "[]"),
        "should_have_source": bool(row["should_have_source"]),
        "created_at": row["created_at"],
    }


def _load_eval_cases() -> list[dict]:
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, question, expected_contains_json, expected_source_contains_json,
                   should_have_source, created_at
            FROM eval_cases
            ORDER BY id ASC
            """
        ).fetchall()
    return [_eval_case_from_row(row) for row in rows]


def _save_eval_run(bot_id: int, result: dict) -> int:
    now = _now_iso()
    with _db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO eval_runs
                (bot_id, total, passed, failed, response_time_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                bot_id,
                int(result.get("total") or 0),
                int(result.get("passed") or 0),
                int(result.get("failed") or 0),
                int(result.get("response_time_ms") or 0),
                now,
            ),
        )
        run_id = int(cursor.lastrowid)
        for item in result.get("items", []):
            conn.execute(
                """
                INSERT INTO eval_run_items
                    (run_id, case_id, question, ok, missing_answer_json,
                     missing_source_json, has_source, response_time_ms, response_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item.get("case_id"),
                    item.get("question", ""),
                    1 if item.get("ok") else 0,
                    json.dumps(item.get("missing_answer") or [], ensure_ascii=False),
                    json.dumps(item.get("missing_source") or [], ensure_ascii=False),
                    1 if item.get("has_source") else 0,
                    int(item.get("response_time_ms") or 0),
                    item.get("response_preview", ""),
                ),
            )
        conn.commit()
    return run_id


def _chat_for_eval(question: str, bot_id: int | None = None) -> str:
    bot = _load_bot(bot_id)
    selected_bot_id = int(bot.get("id") or 1)
    intent = classify_intent(question)
    if intent == "prompt_injection":
        return PROMPT_INJECTION_MESSAGE
    if intent == "greeting":
        return GREETING_MESSAGE
    if intent == "out_of_scope":
        return OUT_OF_SCOPE_MESSAGE
    if not core.vector_retriever or not core.bm25_retriever:
        return INDEX_NOT_READY_MESSAGE

    retrieval_question = question
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
        return direct_response

    combined_docs = _fuse_documents(v_docs, b_docs)
    combined_docs = _rank_documents_for_question(retrieval_question, combined_docs)
    final_docs = combined_docs[:FINAL_TOP_K]
    if _retrieval_relevance_score(retrieval_question, final_docs) < 0.12:
        return NO_ANSWER_MESSAGE

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
        return table_answer

    context_text = _format_context(final_docs)
    if not context_text:
        return NO_ANSWER_MESSAGE

    final_prompt = _format_prompt(context_text, retrieval_question, bot)
    response = llm.invoke(final_prompt)
    formatted_response = _format_final_response(retrieval_question, response, final_docs)
    if "-----------------------------------" not in formatted_response:
        return NO_ANSWER_MESSAGE
    return formatted_response


@router.get("/api/admin/eval/cases")
async def admin_eval_cases(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    return {"items": _load_eval_cases()}


@router.post("/api/admin/eval/cases")
async def admin_create_eval_case(
    request: EvalCaseCreate,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    if not request.question.strip():
        return {"error": "empty_question"}
    with _db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO eval_cases
                (question, expected_contains_json, expected_source_contains_json,
                 should_have_source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request.question.strip(),
                json.dumps([item.strip() for item in request.expected_contains if item.strip()], ensure_ascii=False),
                json.dumps([item.strip() for item in request.expected_source_contains if item.strip()], ensure_ascii=False),
                1 if request.should_have_source else 0,
                _now_iso(),
            ),
        )
        conn.commit()
    return {"ok": True, "id": cursor.lastrowid}


@router.put("/api/admin/eval/cases/{case_id}")
async def admin_update_eval_case(
    case_id: int,
    request: EvalCaseUpdate,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    updates = []
    values = []
    if request.question is not None:
        question = request.question.strip()
        if not question:
            return {"error": "empty_question"}
        updates.append("question = ?")
        values.append(question)
    if request.expected_contains is not None:
        updates.append("expected_contains_json = ?")
        values.append(json.dumps([item.strip() for item in request.expected_contains if item.strip()], ensure_ascii=False))
    if request.expected_source_contains is not None:
        updates.append("expected_source_contains_json = ?")
        values.append(json.dumps([item.strip() for item in request.expected_source_contains if item.strip()], ensure_ascii=False))
    if request.should_have_source is not None:
        updates.append("should_have_source = ?")
        values.append(1 if request.should_have_source else 0)
    if not updates:
        return {"ok": True}
    values.append(case_id)
    with _db_connect() as conn:
        cursor = conn.execute(
            f"UPDATE eval_cases SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        conn.commit()
    if cursor.rowcount <= 0:
        return {"error": "not_found"}
    return {"ok": True}


@router.delete("/api/admin/eval/cases/{case_id}")
async def admin_delete_eval_case(
    case_id: int,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        conn.execute("DELETE FROM eval_cases WHERE id = ?", (case_id,))
        conn.commit()
    return {"ok": True}


@router.get("/api/admin/eval/history")
async def admin_eval_history(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT eval_runs.id, eval_runs.bot_id, bots.name AS bot_name,
                   eval_runs.total, eval_runs.passed, eval_runs.failed,
                   eval_runs.response_time_ms, eval_runs.created_at
            FROM eval_runs
            LEFT JOIN bots ON bots.id = eval_runs.bot_id
            ORDER BY eval_runs.id DESC
            LIMIT 20
            """
        ).fetchall()
        items = []
        for row in rows:
            item_rows = conn.execute(
                """
                SELECT case_id, question, ok, missing_answer_json, missing_source_json,
                       has_source, response_time_ms, response_preview
                FROM eval_run_items
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (row["id"],),
            ).fetchall()
            item = _row_to_dict(row)
            item["items"] = [
                {
                    **_row_to_dict(item_row),
                    "ok": bool(item_row["ok"]),
                    "has_source": bool(item_row["has_source"]),
                    "missing_answer": json.loads(item_row["missing_answer_json"] or "[]"),
                    "missing_source": json.loads(item_row["missing_source_json"] or "[]"),
                }
                for item_row in item_rows
            ]
            items.append(item)
    return {"items": items}


@router.post("/api/admin/eval/run")
async def admin_eval_run(
    request: EvalRunRequest | None = None,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}

    saved_case_ids: list[int | None] = []
    if request and request.cases:
        cases = request.cases
        saved_case_ids = [None for _ in cases]
    else:
        stored_cases = _load_eval_cases()
        cases = [
            EvalCase(
                question=item["question"],
                expected_contains=item["expected_contains"],
                expected_source_contains=item["expected_source_contains"],
                should_have_source=item["should_have_source"],
            )
            for item in stored_cases
        ]
        saved_case_ids = [int(item["id"]) for item in stored_cases]
    results = []
    passed = 0
    started = time.perf_counter()
    requested_bot_id = request.bot_id if request and request.bot_id else None
    selected_bot = _load_bot(requested_bot_id)
    selected_bot_id = int(selected_bot.get("id") or 1)
    for index, case in enumerate(cases):
        case_started = time.perf_counter()
        response = _chat_for_eval(case.question, selected_bot_id)
        folded_response = _fold_vietnamese(response)
        missing_answer = [
            item for item in case.expected_contains
            if _fold_vietnamese(item) not in folded_response
        ]
        missing_source = [
            item for item in case.expected_source_contains
            if _fold_vietnamese(item) not in folded_response
        ]
        has_source = "-----------------------------------" in response
        ok = (
            not missing_answer
            and not missing_source
            and (has_source if case.should_have_source else True)
        )
        if ok:
            passed += 1
        results.append(
            {
                "case_id": saved_case_ids[index] if index < len(saved_case_ids) else None,
                "question": case.question,
                "ok": ok,
                "missing_answer": missing_answer,
                "missing_source": missing_source,
                "has_source": has_source,
                "response_time_ms": int((time.perf_counter() - case_started) * 1000),
                "response_preview": response[:500],
            }
        )

    result = {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "response_time_ms": int((time.perf_counter() - started) * 1000),
        "items": results,
    }
    result["run_id"] = _save_eval_run(selected_bot_id, result)
    result["bot"] = {"id": selected_bot_id, "name": selected_bot.get("name", "")}
    return result

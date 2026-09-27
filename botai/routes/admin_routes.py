"""Authentication, admin, bot, feedback and diagnostics routes."""

from fastapi import APIRouter

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

router = APIRouter()

def _check_admin_token(x_admin_token: str | None) -> bool:
    token = (x_admin_token or "").strip()
    if ADMIN_TOKEN and token == ADMIN_TOKEN:
        return True
    user = _user_from_token(token)
    if user and user.get("role") == "admin":
        return True
    return not ADMIN_TOKEN and not token


def _require_admin_user(x_admin_token: str | None) -> dict | None:
    token = (x_admin_token or "").strip()
    if ADMIN_TOKEN and token == ADMIN_TOKEN:
        return {"id": 0, "username": "legacy-admin-token", "role": "admin"}
    user = _user_from_token(token)
    if user and user.get("role") == "admin":
        return user
    if not ADMIN_TOKEN and not token:
        return {"id": 0, "username": "development-admin", "role": "admin"}
    return None


@router.post("/api/auth/login")
async def auth_login(request: LoginRequest, http_request: Request):
    limited = _rate_limit_json(http_request, "login")
    if limited:
        return limited
    username = request.username.strip()
    password = request.password
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row or not _verify_password(password, row["password_hash"]):
        return {"ok": False, "error": "Sai tài khoản hoặc mật khẩu."}
    token = _create_auth_session(int(row["id"]))
    return {
        "ok": True,
        "token": token,
        "user": {"id": row["id"], "username": row["username"], "role": row["role"]},
    }


@router.get("/api/auth/me")
async def auth_me(x_admin_token: str | None = Header(default=None)):
    user = _user_from_token(x_admin_token)
    if user:
        return {"ok": True, "user": user}
    if ADMIN_TOKEN and x_admin_token == ADMIN_TOKEN:
        return {"ok": True, "user": {"id": 0, "username": "legacy-admin-token", "role": "admin"}}
    return {"ok": False, "user": None}


@router.post("/api/auth/logout")
async def auth_logout(x_admin_token: str | None = Header(default=None)):
    token = (x_admin_token or "").strip()
    if token:
        with _db_connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
            conn.commit()
    return {"ok": True}


@router.get("/api/admin/users")
async def admin_users(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id DESC"
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.post("/api/admin/users")
async def admin_create_user(
    request: UserCreateRequest,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    username = request.username.strip()
    role = request.role.strip().lower()
    if not username or len(request.password) < 6:
        return {"ok": False, "error": "Tên đăng nhập không được trống và mật khẩu tối thiểu 6 ký tự."}
    if role not in {"admin", "user"}:
        return {"ok": False, "error": "Role chỉ hỗ trợ admin hoặc user."}
    try:
        with _db_connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, _hash_password(request.password), role, _now_iso()),
            )
            conn.commit()
        return {"ok": True, "id": cursor.lastrowid}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Tên đăng nhập đã tồn tại."}


@router.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: int,
    x_admin_token: str | None = Header(default=None),
):
    current_user = _require_admin_user(x_admin_token)
    if not current_user:
        return {"ok": False, "error": "unauthorized"}
    if current_user.get("id") == user_id:
        return {"ok": False, "error": "Không thể xóa chính tài khoản đang đăng nhập."}
    with _db_connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    return {"ok": True}


@router.get("/api/bots")
async def list_public_bots():
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


@router.get("/api/admin/bots")
async def admin_bots(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, system_prompt, is_active, created_at, updated_at
            FROM bots
            ORDER BY id ASC
            """
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.post("/api/admin/bots")
async def admin_create_bot(
    request: BotCreateRequest,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    name = request.name.strip()
    if not name:
        return {"ok": False, "error": "Tên bot không được trống."}
    now = _now_iso()
    with _db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO bots (name, description, system_prompt, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                (request.description or "").strip(),
                (request.system_prompt or "").strip(),
                1,
                now,
                now,
            ),
        )
        conn.commit()
    return {"ok": True, "id": cursor.lastrowid}


@router.put("/api/admin/bots/{bot_id}")
async def admin_update_bot(
    bot_id: int,
    request: BotUpdateRequest,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    updates = []
    values = []
    if request.name is not None:
        name = request.name.strip()
        if not name:
            return {"ok": False, "error": "Tên bot không được trống."}
        updates.append("name = ?")
        values.append(name)
    if request.description is not None:
        updates.append("description = ?")
        values.append(request.description.strip())
    if request.system_prompt is not None:
        updates.append("system_prompt = ?")
        values.append(request.system_prompt.strip())
    if request.is_active is not None:
        updates.append("is_active = ?")
        values.append(1 if request.is_active else 0)
    if not updates:
        return {"ok": True}
    updates.append("updated_at = ?")
    values.append(_now_iso())
    values.append(bot_id)
    with _db_connect() as conn:
        conn.execute(f"UPDATE bots SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()
    return {"ok": True}


@router.delete("/api/admin/bots/{bot_id}")
async def admin_delete_bot(
    bot_id: int,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    with _db_connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM bots").fetchone()["count"]
        if count <= 1:
            return {"ok": False, "error": "Không thể xóa bot cuối cùng."}
        document_count = conn.execute(
            "SELECT COUNT(*) AS count FROM documents WHERE bot_id = ?",
            (bot_id,),
        ).fetchone()["count"]
        if document_count:
            return {"ok": False, "error": "Bot này đang có tài liệu. Hãy xóa hoặc chuyển tài liệu trước."}
        conn.execute("DELETE FROM bots WHERE id = ?", (bot_id,))
        conn.commit()
    return {"ok": True}


@router.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    rating = request.rating.strip().lower()
    if rating not in {"helpful", "wrong", "bad_source"}:
        return {"ok": False, "error": "rating không hợp lệ"}

    bot = _load_bot(request.bot_id)
    bot_id = int(bot.get("id") or 1)

    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback
                (session_id, bot_id, question, answer, sources, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.session_id or "",
                bot_id,
                request.question.strip(),
                request.answer.strip(),
                request.sources or "",
                rating,
                request.comment or "",
                _now_iso(),
            ),
        )
        conn.commit()
    return {"ok": True}


@router.get("/api/admin/stats")
async def admin_stats(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        total_questions = conn.execute("SELECT COUNT(*) AS count FROM chat_messages").fetchone()["count"]
        no_answer = conn.execute("SELECT COUNT(*) AS count FROM chat_messages WHERE no_answer = 1").fetchone()["count"]
        avg_time = conn.execute("SELECT AVG(response_time_ms) AS avg_time FROM chat_messages").fetchone()["avg_time"] or 0
        bad_feedback = conn.execute(
            "SELECT COUNT(*) AS count FROM feedback WHERE rating IN ('wrong', 'bad_source')"
        ).fetchone()["count"]
        top_intents = [
            _row_to_dict(row)
            for row in conn.execute(
                """
                SELECT intent, COUNT(*) AS count
                FROM chat_messages
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()
        ]
        source_rows = conn.execute(
            """
            SELECT sources
            FROM chat_messages
            WHERE sources IS NOT NULL AND sources != ''
            """
        ).fetchall()
        source_counts: dict[str, int] = {}
        for row in source_rows:
            for source in re.findall(r"(?:tài liệu|Excel)\s+([^,\n]+)", _repair_mojibake(row["sources"]), flags=re.I):
                source = source.strip()
                if source:
                    source_counts[source] = source_counts.get(source, 0) + 1
        top_sources = [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        bot_stats = [
            _row_to_dict(row)
            for row in conn.execute(
                """
                SELECT
                    bots.id AS bot_id,
                    bots.name AS bot_name,
                    COUNT(chat_messages.id) AS total_questions,
                    COALESCE(SUM(chat_messages.no_answer), 0) AS no_answer,
                    COALESCE(ROUND(AVG(chat_messages.response_time_ms)), 0) AS average_response_time_ms
                FROM bots
                LEFT JOIN chat_messages ON chat_messages.bot_id = bots.id
                GROUP BY bots.id, bots.name
                ORDER BY total_questions DESC, bots.id ASC
                """
            ).fetchall()
        ]
        document_stats_by_bot = [
            _row_to_dict(row)
            for row in conn.execute(
                """
                SELECT
                    bots.id AS bot_id,
                    bots.name AS bot_name,
                    COUNT(documents.id) AS document_count
                FROM bots
                LEFT JOIN documents ON documents.bot_id = bots.id
                GROUP BY bots.id, bots.name
                ORDER BY bots.id ASC
                """
            ).fetchall()
        ]
    return {
        "total_questions": total_questions,
        "no_answer": no_answer,
        "average_response_time_ms": round(avg_time),
        "bad_feedback": bad_feedback,
        "cache_entries": len(core.response_cache),
        "top_intents": top_intents,
        "top_sources": top_sources,
        "bot_stats": bot_stats,
        "document_stats_by_bot": document_stats_by_bot,
        "index": _index_runtime_stats(),
        "ocr": _ocr_runtime_status(),
    }


@router.get("/api/admin/questions")
async def admin_questions(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT chat_messages.id, chat_messages.session_id, chat_messages.bot_id,
                   bots.name AS bot_name, chat_messages.question, chat_messages.answer,
                   chat_messages.sources, chat_messages.intent, chat_messages.response_time_ms,
                   chat_messages.no_answer, chat_messages.created_at
            FROM chat_messages
            LEFT JOIN bots ON bots.id = chat_messages.bot_id
            ORDER BY chat_messages.id DESC
            LIMIT 20
            """
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.get("/api/admin/feedback")
async def admin_feedback(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT feedback.id, feedback.session_id, feedback.bot_id, bots.name AS bot_name,
                   feedback.question, feedback.answer, feedback.sources, feedback.rating,
                   feedback.comment, feedback.created_at
            FROM feedback
            LEFT JOIN bots ON bots.id = feedback.bot_id
            WHERE rating IN ('wrong', 'bad_source')
            ORDER BY feedback.id DESC
            LIMIT 50
            """
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.get("/api/chat/history")
async def chat_history(session_id: str, bot_id: int | None = None):
    session_id = (session_id or "").strip()
    if not session_id:
        return {"items": []}
    selected_bot_id = None
    if bot_id:
        bot = _load_bot(bot_id)
        selected_bot_id = int(bot.get("id") or 1)
    with _db_connect() as conn:
        if selected_bot_id:
            rows = conn.execute(
                """
                SELECT id, session_id, bot_id, question, answer, sources,
                       intent, response_time_ms, created_at
                FROM chat_messages
                WHERE session_id = ? AND bot_id = ?
                ORDER BY id ASC
                LIMIT 100
                """,
                (session_id, selected_bot_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, bot_id, question, answer, sources,
                       intent, response_time_ms, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT 100
                """,
                (session_id,),
            ).fetchall()
    items = []
    for row in rows:
        record = _row_to_dict(row)
        response = record["answer"]
        if record.get("sources"):
            response = f"{response}\n\n-----------------------------------\n{record['sources']}"
        items.append(
            {
                **record,
                "response": response,
            }
        )
    return {"items": items}


@router.post("/api/admin/retrieve/debug")
async def admin_retrieve_debug(
    request: RetrieveDebugRequest,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    if not request.question.strip():
        return {"error": "empty_question"}
    return _debug_retrieve(request.question, request.top_k, request.bot_id)


@router.get("/api/admin/system/diagnostics")
async def admin_system_diagnostics(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
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

"""Document management, upload, replace, move and export routes."""

from fastapi import APIRouter

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

router = APIRouter()

def _docs_for_source(filename: str, bot_id: int | None = None, limit: int = 18) -> list[Document]:
    source_name = Path(filename).name
    docs = [
        doc for doc in core.all_indexed_docs
        if Path(str(doc.metadata.get("source", ""))).name == source_name
        and (bot_id is None or _doc_matches_bot(doc, bot_id))
    ]
    docs.sort(
        key=lambda doc: (
            int(doc.metadata.get("pdf_page") or doc.metadata.get("page") or 0),
            int(doc.metadata.get("chunk_index") or 0),
        )
    )
    table_docs = [doc for doc in docs if doc.metadata.get("type") in TABLE_DOC_TYPES]
    text_docs = [doc for doc in docs if doc not in table_docs]
    selected = (text_docs[: max(limit - 5, 1)] + table_docs[:5])[:limit]
    return selected


def _source_list_for_docs(docs: list[Document]) -> str:
    lines = []
    seen = set()
    for doc in docs:
        line = _table_source_line(doc, docs) if doc.metadata.get("type") in TABLE_DOC_TYPES else _source_line_for_doc(doc)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines[:10])


def _generate_document_report(original_name: str, docs: list[Document], mode: str) -> tuple[str, str, str]:
    context_parts = []
    for index, doc in enumerate(docs, start=1):
        source_line = _table_source_line(doc, docs) if doc.metadata.get("type") in TABLE_DOC_TYPES else _source_line_for_doc(doc)
        context_parts.append(
            f"[{index}] {source_line}\n{_preview_text(_strip_context_markers(doc.page_content), 900)}"
        )
    context = "\n\n".join(context_parts)
    mode = mode.strip().lower()
    if mode not in {"summary", "checklist"}:
        mode = "summary"
    if mode == "checklist":
        title = f"Checklist từ tài liệu {original_name}"
        instruction = (
            "Tạo checklist thực hiện/kiểm tra dựa trên tài liệu. "
            "Nhóm các mục theo chủ đề, mỗi mục bắt đầu bằng '- [ ]'. "
            "Chỉ dùng thông tin trong trích đoạn, không bịa thêm."
        )
    else:
        title = f"Tóm tắt tài liệu {original_name}"
        instruction = (
            "Tóm tắt tài liệu theo các mục: Chủ đề chính, Điểm quan trọng, "
            "Số liệu hoặc bảng đáng chú ý, Lưu ý khi sử dụng. "
            "Chỉ dùng thông tin trong trích đoạn, không bịa thêm."
        )
    prompt = f"""Bạn là trợ lý tóm tắt tài liệu bằng tiếng Việt.

Yêu cầu:
- {instruction}
- Viết ngắn gọn, rõ ý, ưu tiên dữ kiện cụ thể.
- Nếu trích đoạn không đủ để kết luận, hãy nói rõ là tài liệu trích xuất chưa đủ.

Tài liệu: {original_name}

TRÍCH ĐOẠN:
{context}

KẾT QUẢ:
"""
    body = str(llm.invoke(prompt)).strip()
    sources = _source_list_for_docs(docs)
    return title, body, sources




@router.post("/api/export")
async def export_answer(request: ExportRequest, http_request: Request):
    limited = _rate_limit_json(http_request, "export")
    if limited:
        return limited
    file_format = request.format.strip().lower()
    if file_format not in {"docx", "pdf"}:
        return {"ok": False, "error": "format chỉ hỗ trợ docx hoặc pdf"}

    filename = f"answer_{int(time.time())}.{file_format}"
    path = Path(EXPORTS_DIR) / filename
    if file_format == "docx":
        _create_docx_export(path, request.question, request.answer, request.sources or "")
    else:
        _create_pdf_export(path, request.question, request.answer, request.sources or "")
    return {"ok": True, "url": f"/exports/{filename}"}


@router.post("/api/documents/{document_id}/report")
async def export_document_report(
    document_id: int,
    request: DocumentReportRequest,
    http_request: Request,
    x_admin_token: str | None = Header(default=None),
):
    limited = _rate_limit_json(http_request, "export")
    if limited:
        return limited
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    file_format = request.format.strip().lower()
    if file_format not in {"docx", "pdf"}:
        return {"ok": False, "error": "format chỉ hỗ trợ docx hoặc pdf"}
    mode = request.mode.strip().lower()
    if mode not in {"summary", "checklist"}:
        return {"ok": False, "error": "mode chỉ hỗ trợ summary hoặc checklist"}

    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT id, filename, original_name, bot_id, status
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
    if row is None:
        return {"ok": False, "error": "Không tìm thấy tài liệu."}
    if row["status"] != "done":
        return {"ok": False, "error": "Tài liệu chưa ingest xong, hãy thử lại sau."}

    docs = _docs_for_source(row["filename"], int(row["bot_id"] or 1))
    if not docs:
        return {"ok": False, "error": "Không tìm thấy chunk đã ingest cho tài liệu này."}

    title, body, sources = _generate_document_report(row["original_name"], docs, mode)
    filename = f"document_{document_id}_{mode}_{int(time.time())}.{file_format}"
    path = Path(EXPORTS_DIR) / filename
    if file_format == "docx":
        _create_document_report_docx(path, title, body, sources)
    else:
        _create_document_report_pdf(path, title, body, sources)
    return {"ok": True, "url": f"/exports/{filename}", "title": title}


@router.get("/api/documents")
async def list_documents(x_admin_token: str | None = Header(default=None)):
    if not _check_admin_token(x_admin_token):
        return {"error": "unauthorized"}
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT documents.id, documents.filename, documents.original_name, documents.file_type,
                   documents.bot_id, bots.name AS bot_name, documents.status,
                   documents.created_at, documents.updated_at, documents.error_message,
                   CASE WHEN documents.filename = ? THEN 1 ELSE 0 END AS protected
            FROM documents
            LEFT JOIN bots ON bots.id = documents.bot_id
            ORDER BY documents.id DESC
            """
            ,
            (DEFAULT_DOCUMENT_FILENAME,),
        ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@router.post("/api/documents/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    bot_id: int = Form(default=1),
    x_admin_token: str | None = Header(default=None),
):
    limited = _rate_limit_json(request, "upload")
    if limited:
        return limited
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    original_name = file.filename or "document"
    safe_name = _safe_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        return {
            "ok": False,
            "error": "Định dạng chưa hỗ trợ. Hỗ trợ: PDF, TXT, DOCX, XLSX, PPTX.",
        }
    bot = _load_bot(bot_id)
    bot_id = int(bot.get("id") or 1)

    stored_name = f"{int(time.time())}_{safe_name}"
    file_path = Path(DOCUMENTS_DIR) / stored_name
    content = await file.read()
    validation_error = _validate_upload_bytes(safe_name, content)
    if validation_error:
        return {"ok": False, "error": validation_error}
    file_path.write_bytes(content)

    with _db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents
                (filename, original_name, file_type, bot_id, status, created_at, updated_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (stored_name, original_name, suffix.lstrip("."), bot_id, "processing", _now_iso(), _now_iso(), ""),
        )
        document_id = int(cursor.lastrowid)
        conn.commit()

    background_tasks.add_task(_run_rebuild_job, document_id)
    return {
        "ok": True,
        "document_id": document_id,
        "filename": stored_name,
        "status": "processing",
    }


@router.post("/api/documents/url")
async def add_url_document(
    http_request: Request,
    request: UrlDocumentRequest,
    background_tasks: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
):
    limited = _rate_limit_json(http_request, "url")
    if limited:
        return limited
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    url = request.url.strip()
    if _is_blocked_url(url):
        return {"ok": False, "error": "URL khong hop le hoac bi chan vi ly do bao mat."}
    bot = _load_bot(request.bot_id)
    bot_id = int(bot.get("id") or 1)

    try:
        title, text = _download_url_as_text(url)
    except Exception as exc:
        return {"ok": False, "error": f"Khong tai duoc URL: {exc}"}

    safe_name = _url_to_safe_filename(url)
    stored_name = f"{int(time.time())}_{safe_name}"
    file_path = Path(DOCUMENTS_DIR) / stored_name
    content = (
        f"Tieu de: {title}\n"
        f"Nguon URL: {url}\n\n"
        f"{text}"
    )
    file_path.write_text(content, encoding="utf-8")

    with _db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents
                (filename, original_name, file_type, bot_id, status, created_at, updated_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (stored_name, url, "url", bot_id, "processing", _now_iso(), _now_iso(), ""),
        )
        document_id = int(cursor.lastrowid)
        conn.commit()

    background_tasks.add_task(_run_rebuild_job, document_id)
    return {
        "ok": True,
        "document_id": document_id,
        "filename": stored_name,
        "status": "processing",
    }


@router.post("/api/documents/{document_id}/replace")
async def replace_document(
    document_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_admin_token: str | None = Header(default=None),
):
    limited = _rate_limit_json(request, "upload")
    if limited:
        return limited
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}

    original_name = file.filename or "document"
    safe_name = _safe_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        return {
            "ok": False,
            "error": "Định dạng chưa hỗ trợ. Hỗ trợ: PDF, TXT, DOCX, XLSX, PPTX.",
        }

    content = await file.read()
    validation_error = _validate_upload_bytes(safe_name, content)
    if validation_error:
        return {"ok": False, "error": validation_error}

    with _db_connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "Không tìm thấy tài liệu."}
        old_filename = row["filename"]
        if old_filename == DEFAULT_DOCUMENT_FILENAME:
            return {"ok": False, "error": "Không thể thay thế tài liệu hệ thống tailieu.pdf bằng API này."}

        stored_name = f"{int(time.time())}_{safe_name}"
        file_path = Path(DOCUMENTS_DIR) / stored_name
        file_path.write_bytes(content)
        conn.execute(
            """
            UPDATE documents
            SET filename = ?, original_name = ?, file_type = ?,
                status = ?, updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (
                stored_name,
                original_name,
                suffix.lstrip("."),
                "processing",
                _now_iso(),
                "",
                document_id,
            ),
        )
        conn.commit()

    old_file_path = Path(DOCUMENTS_DIR) / Path(old_filename).name
    if old_file_path.exists():
        old_file_path.unlink()

    background_tasks.add_task(_run_rebuild_job, document_id)
    return {
        "ok": True,
        "document_id": document_id,
        "filename": stored_name,
        "status": "processing",
    }


@router.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return {"ok": False, "error": "Không tìm thấy tài liệu."}
        filename = row["filename"]
        if filename == DEFAULT_DOCUMENT_FILENAME:
            return {"ok": False, "error": "Không thể xóa tài liệu hệ thống tailieu.pdf. Hãy chuyển bot nếu cần phân quyền dữ liệu."}
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()

    file_path = Path(DOCUMENTS_DIR) / Path(filename).name
    if file_path.exists():
        file_path.unlink()

    background_tasks.add_task(_run_rebuild_job, None)
    return {"ok": True, "status": "processing"}


@router.put("/api/documents/{document_id}/bot")
async def move_document_to_bot(
    document_id: int,
    request: DocumentMoveRequest,
    background_tasks: BackgroundTasks,
    x_admin_token: str | None = Header(default=None),
):
    if not _check_admin_token(x_admin_token):
        return {"ok": False, "error": "unauthorized"}
    with _db_connect() as conn:
        doc_row = conn.execute(
            "SELECT id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if doc_row is None:
            return {"ok": False, "error": "KhÃ´ng tÃ¬m tháº¥y tÃ i liá»‡u."}
        bot_row = conn.execute(
            "SELECT id FROM bots WHERE id = ? AND is_active = 1",
            (request.bot_id,),
        ).fetchone()
        if bot_row is None:
            return {"ok": False, "error": "KhÃ´ng tÃ¬m tháº¥y bot hoáº¡t Ä‘á»™ng."}
        conn.execute(
            """
            UPDATE documents
            SET bot_id = ?, status = ?, updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (request.bot_id, "processing", _now_iso(), "", document_id),
        )
        conn.commit()
    background_tasks.add_task(_run_rebuild_job, document_id)
    return {"ok": True, "document_id": document_id, "bot_id": request.bot_id, "status": "processing"}

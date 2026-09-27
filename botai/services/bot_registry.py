"""Quan ly bot mac dinh, tai lieu mac dinh va truy xuat bot."""

from pathlib import Path

from botai.core.config import DEFAULT_DOCUMENT_FILENAME
from botai.services.database_service import _db_connect, _now_iso, _row_to_dict

__all__ = ["_ensure_default_bot", "_ensure_default_document", "_load_bot"]

def _ensure_default_bot() -> None:
    with _db_connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM bots").fetchone()["count"]
        if count:
            return
        now = _now_iso()
        conn.execute(
            """
            INSERT INTO bots (name, description, system_prompt, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "Qwen AI",
                "Bot mặc định hỏi đáp trên toàn bộ kho tài liệu.",
                "",
                1,
                now,
                now,
            ),
        )
        conn.commit()


def _ensure_default_document() -> None:
    status = "done" if Path(DEFAULT_DOCUMENT_FILENAME).exists() else "error"
    error_message = "" if status == "done" else "Không tìm thấy file tailieu.pdf trong thư mục dự án."
    now = _now_iso()
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT id, bot_id FROM documents WHERE filename = ?",
            (DEFAULT_DOCUMENT_FILENAME,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE documents
                SET original_name = ?, file_type = ?, status = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    DEFAULT_DOCUMENT_FILENAME,
                    "pdf",
                    status,
                    error_message,
                    now,
                    row["id"],
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO documents
                    (filename, original_name, file_type, bot_id, status, created_at, updated_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_DOCUMENT_FILENAME,
                    DEFAULT_DOCUMENT_FILENAME,
                    "pdf",
                    1,
                    status,
                    now,
                    now,
                    error_message,
                ),
            )
        conn.commit()


def _load_bot(bot_id: int | None) -> dict:
    with _db_connect() as conn:
        row = None
        if bot_id:
            row = conn.execute(
                "SELECT id, name, description, system_prompt, is_active FROM bots WHERE id = ?",
                (bot_id,),
            ).fetchone()
            if row is not None and not row["is_active"]:
                row = None
        if row is None:
            row = conn.execute(
                "SELECT id, name, description, system_prompt, is_active FROM bots WHERE is_active = 1 ORDER BY id ASC LIMIT 1"
            ).fetchone()
    if row is None:
        return {"id": 0, "name": "Qwen AI", "description": "", "system_prompt": "", "is_active": 1}
    return _row_to_dict(row)

"""SQLite storage service cho users, bots, documents, chat history va evaluation."""

import json
import re
import sqlite3
from datetime import datetime

from botai.core.config import APP_DB_PATH, DEFAULT_DOCUMENT_FILENAME, DOCUMENT_COLLECTION_VERSION
from botai.services.source_service import _fold_vietnamese

__all__ = [
    "_db_connect",
    "_init_app_db",
    "_now_iso",
    "_split_answer_sources",
    "_parse_source_items",
    "_save_chat_message",
    "_row_to_dict",
]

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_app_db() -> None:
    with _db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                bot_id INTEGER DEFAULT 1,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                intent TEXT,
                response_time_ms INTEGER,
                no_answer INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                bot_id INTEGER DEFAULT 1,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                bot_id INTEGER DEFAULT 1,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                expected_contains_json TEXT NOT NULL,
                expected_source_contains_json TEXT NOT NULL,
                should_have_source INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                system_prompt TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER DEFAULT 1,
                total INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                response_time_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_run_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                case_id INTEGER,
                question TEXT NOT NULL,
                ok INTEGER NOT NULL,
                missing_answer_json TEXT NOT NULL,
                missing_source_json TEXT NOT NULL,
                has_source INTEGER NOT NULL,
                response_time_ms INTEGER NOT NULL,
                response_preview TEXT,
                FOREIGN KEY(run_id) REFERENCES eval_runs(id)
            )
            """
        )
        document_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "bot_id" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN bot_id INTEGER DEFAULT 1")
        chat_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        }
        if "bot_id" not in chat_columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN bot_id INTEGER DEFAULT 1")
        feedback_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()
        }
        if "bot_id" not in feedback_columns:
            conn.execute("ALTER TABLE feedback ADD COLUMN bot_id INTEGER DEFAULT 1")
        conn.commit()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_answer_sources(response: str) -> tuple[str, str]:
    parts = str(response or "").split("-----------------------------------", 1)
    answer = parts[0].strip()
    sources = parts[1].strip() if len(parts) > 1 else ""
    return answer, sources


def _parse_source_items(sources: str) -> list[dict]:
    items: list[dict] = []
    current: dict | None = None
    for raw_line in str(sources or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("theo "):
            if current:
                items.append(current)
            current = {
                "raw": line,
                "document": "",
                "page": None,
                "url": "",
                "type": "",
                "excerpt": "",
            }
            document_match = re.search(
                r"(?:tài liệu|tai lieu|excel|slide\s+\d+\s+trong)\s+([^,\n]+)",
                line,
                flags=re.I,
            )
            if document_match:
                current["document"] = document_match.group(1).strip()
            page_match = re.search(r"trang PDF\s+([0-9]+(?:-[0-9]+)?)", line, flags=re.I)
            if page_match:
                current["page"] = page_match.group(1)
            continue
        if current is None:
            current = {"raw": "", "document": "", "page": None, "url": "", "type": "", "excerpt": ""}
        current["raw"] = f"{current.get('raw', '')}\n{line}".strip()
        folded = _fold_vietnamese(line)
        if folded.startswith("duong dan:"):
            current["url"] = line.split(":", 1)[1].strip()
        elif folded.startswith("loai nguon:"):
            current["type"] = line.split(":", 1)[1].strip()
        elif folded.startswith("trich doan:"):
            current["excerpt"] = line.split(":", 1)[1].strip()
    if current:
        items.append(current)
    return items


def _save_chat_message(
    session_id: str | None,
    question: str,
    response: str,
    intent: str,
    response_time_ms: int,
    bot_id: int = 1,
) -> None:
    answer, sources = _split_answer_sources(response)
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages
                (session_id, bot_id, question, answer, sources, intent, response_time_ms, no_answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id or "",
                bot_id,
                question,
                answer,
                sources,
                intent,
                response_time_ms,
                1 if NO_ANSWER_MESSAGE in response else 0,
                _now_iso(),
            ),
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


_init_app_db()


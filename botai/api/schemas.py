"""Pydantic schemas cho request/response cua BotAI API.

Dat rieng schemas giup routes ro rang hon va tranh de app.py qua lon.
"""

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    bot_id: int | None = None


class ApiChatRequest(ChatRequest):
    include_raw_response: bool = False


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: str | None = ""
    rating: str
    comment: str | None = ""
    session_id: str | None = None
    bot_id: int | None = None


class ExportRequest(BaseModel):
    question: str
    answer: str
    sources: str | None = ""
    format: str = "docx"


class DocumentReportRequest(BaseModel):
    format: str = "docx"
    mode: str = "summary"


class UrlDocumentRequest(BaseModel):
    url: str
    bot_id: int | None = 1


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class BotCreateRequest(BaseModel):
    name: str
    description: str | None = ""
    system_prompt: str | None = ""


class BotUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    is_active: bool | None = None


class DocumentMoveRequest(BaseModel):
    bot_id: int


class ChatHistoryQuery(BaseModel):
    session_id: str
    bot_id: int | None = None


class EvalCase(BaseModel):
    question: str
    expected_contains: list[str] = []
    expected_source_contains: list[str] = []
    should_have_source: bool = True


class EvalRunRequest(BaseModel):
    cases: list[EvalCase] | None = None
    bot_id: int | None = None


class EvalCaseCreate(BaseModel):
    question: str
    expected_contains: list[str] = []
    expected_source_contains: list[str] = []
    should_have_source: bool = True


class EvalCaseUpdate(BaseModel):
    question: str | None = None
    expected_contains: list[str] | None = None
    expected_source_contains: list[str] | None = None
    should_have_source: bool | None = None


class RetrieveDebugRequest(BaseModel):
    question: str
    top_k: int = 8
    bot_id: int | None = None

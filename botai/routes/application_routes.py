"""Aggregate FastAPI routers for BotAI."""

from fastapi import APIRouter

from botai.routes import admin_routes, chat_routes, document_routes, evaluation_routes, page_routes
from botai.routes.admin_routes import _check_admin_token, _require_admin_user
from botai.routes.document_routes import _docs_for_source, _generate_document_report, _source_list_for_docs
from botai.routes.evaluation_routes import _chat_for_eval, _eval_case_from_row, _load_eval_cases, _save_eval_run

router = APIRouter()
for _route_module in (page_routes, document_routes, admin_routes, chat_routes, evaluation_routes):
    router.include_router(_route_module.router)

__all__ = [
    "router",
    "_docs_for_source",
    "_source_list_for_docs",
    "_generate_document_report",
    "_check_admin_token",
    "_require_admin_user",
    "_eval_case_from_row",
    "_load_eval_cases",
    "_save_eval_run",
    "_chat_for_eval",
]

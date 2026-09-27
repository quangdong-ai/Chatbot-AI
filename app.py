import os
import pickle
import re
import sqlite3
import time
import unicodedata
import warnings
import json
import ipaddress
import hashlib
import hmac
import importlib.util
import secrets
import socket
import urllib.parse
import urllib.request
import gc
import shutil
import threading
from html.parser import HTMLParser
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import pymupdf as fitz
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Header
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from botai.core.config import *
from botai.core.state import *
from botai.models.runtime import *
from botai.api.schemas import *
from botai.services.export_service import *
from botai.services.database_service import *
from botai.services.source_service import *
from botai.security.auth_service import *
from botai.security.rate_limit_service import *
from botai.services.bot_registry import *
from botai.services.document_input_service import *
from botai.answering.response_service import *


app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")








from botai.services import index_service as _index_service
from botai.retrieval import retrieval_service as _retrieval_service
from botai.answering import rule_service as _rule_service
from botai.services import debug_service as _debug_service

for _module in (_index_service, _retrieval_service, _rule_service, _debug_service):
    for _name in getattr(_module, "__all__", []):
        globals()[_name] = getattr(_module, _name)
from botai.routes import application_routes

app.include_router(application_routes.router)
for _name in (
    "_docs_for_source",
    "_source_list_for_docs",
    "_generate_document_report",
    "_check_admin_token",
    "_require_admin_user",
    "_eval_case_from_row",
    "_load_eval_cases",
    "_save_eval_run",
    "_chat_for_eval",
):
    globals()[_name] = getattr(application_routes, _name)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

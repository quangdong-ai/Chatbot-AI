"""Frontend/static page and file-serving routes."""

from fastapi import APIRouter

import app as core

globals().update({name: getattr(core, name) for name in dir(core) if not name.startswith("__")})

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@router.get("/admin", response_class=HTMLResponse)
async def read_admin():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()


@router.get("/api-docs", response_class=HTMLResponse)
async def read_api_docs():
    with open("static/api-docs.html", "r", encoding="utf-8") as f:
        return f.read()


@router.get("/documents/tailieu.pdf")
async def read_pdf():
    return FileResponse(DEFAULT_DOCUMENT_FILENAME, media_type="application/pdf", filename=DEFAULT_DOCUMENT_FILENAME)


@router.get("/documents/{filename}")
async def read_uploaded_document(filename: str):
    safe_name = Path(filename).name
    file_path = Path(DOCUMENTS_DIR) / safe_name
    if not file_path.exists():
        return {"error": "file_not_found"}
    return FileResponse(str(file_path), filename=safe_name)


@router.get("/exports/{filename}")
async def read_export(filename: str):
    safe_name = Path(filename).name
    file_path = Path(EXPORTS_DIR) / safe_name
    if not file_path.exists():
        return {"error": "file_not_found"}
    return FileResponse(str(file_path), filename=safe_name)

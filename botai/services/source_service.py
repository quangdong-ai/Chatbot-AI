"""Xu ly alias tai lieu, URL nguon, source line va trich dan nguon."""

import re
import unicodedata
import urllib.parse
from pathlib import Path

from langchain_core.documents import Document

from botai.core.config import DEFAULT_DOCUMENT_FILENAME, TABLE_DOC_TYPES

__all__ = ['_repair_mojibake', '_fold_vietnamese', '_requested_license_classes', '_document_display_name', '_document_url', '_document_aliases_for_doc', '_question_mentions_doc', '_question_mentions_known_document', '_filter_docs_by_mentioned_source', '_filter_retrieval_by_mentioned_source', '_source_line_for_doc', '_table_source_line', '_source_type_label', '_strip_context_markers', '_source_excerpt', '_source_detail_for_doc']

def _repair_mojibake(text: str) -> str:
    raw = str(text or "")
    try:
        repaired = raw.encode("latin1").decode("utf-8")
    except UnicodeError:
        return raw
    return repaired if repaired else raw


def _fold_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFD", _repair_mojibake(text).casefold())
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return (
        normalized
        .replace("đ", "d")
        .replace("Đ", "d")
    )


def _requested_license_classes(question: str) -> list[str]:
    classes = []
    folded_question = _fold_vietnamese(question)
    class_pattern = (
        r"\b(?:hang\s*)?"
        r"(d2e|d1e|c1e|be|ce|de|a1|b1|c1|d1|d2|a|b|c|d)"
        r"\b"
    )
    for raw in re.findall(class_pattern, folded_question, re.I):
        value = raw.upper()
        if value not in classes:
            classes.append(value)
    return classes


def _document_display_name(source: str) -> str:
    source = str(source or "tailieu.pdf")
    try:
        conn = _db_connect()
        row = conn.execute(
            "SELECT original_name FROM documents WHERE filename = ?",
            (source,),
        ).fetchone()
        conn.close()
        if row and row["original_name"]:
            return str(row["original_name"])
    except Exception:
        pass
    return source


def _document_url(source: str, page: int | str | None = None) -> str:
    url = f"/documents/{urllib.parse.quote(str(source or 'tailieu.pdf'))}"
    if page:
        url = f"{url}#page={page}"
    return url


def _document_aliases_for_doc(doc: Document) -> set[str]:
    source = str((doc.metadata or {}).get("source") or "tailieu.pdf")
    display_name = _document_display_name(source)
    aliases = {source, display_name, Path(source).stem, Path(display_name).stem}
    return {alias for alias in aliases if alias}


def _question_mentions_doc(question: str, doc: Document) -> bool:
    folded_question = _fold_vietnamese(question)
    for alias in _document_aliases_for_doc(doc):
        folded_alias = _fold_vietnamese(alias)
        if len(folded_alias) >= 3 and folded_alias in folded_question:
            return True
    return False


def _question_mentions_known_document(question: str) -> bool:
    folded_question = _fold_vietnamese(question)
    aliases: set[str] = set()
    try:
        conn = _db_connect()
        rows = conn.execute("SELECT filename, original_name FROM documents").fetchall()
        conn.close()
        for row in rows:
            filename = str(row["filename"] or "")
            original_name = str(row["original_name"] or "")
            aliases.update({filename, original_name, Path(filename).stem, Path(original_name).stem})
    except Exception:
        return False
    for alias in aliases:
        folded_alias = _fold_vietnamese(alias)
        if len(folded_alias) >= 3 and folded_alias in folded_question:
            return True
    return False


def _filter_docs_by_mentioned_source(question: str, docs: list[Document]) -> list[Document]:
    matched = [doc for doc in docs if _question_mentions_doc(question, doc)]
    if matched:
        return matched
    if _question_mentions_known_document(question):
        return []
    return docs


def _filter_retrieval_by_mentioned_source(
    question: str,
    v_docs: list[Document],
    b_docs: list[Document],
) -> tuple[list[Document], list[Document]]:
    if not _question_mentions_known_document(question):
        return v_docs, b_docs
    return (
        [doc for doc in v_docs if _question_mentions_doc(question, doc)],
        [doc for doc in b_docs if _question_mentions_doc(question, doc)],
    )


def _source_line_for_doc(doc: Document, label: str = "tài liệu") -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source", "tailieu.pdf")
    display_name = _document_display_name(source)
    page = metadata.get("pdf_page", "?")
    if metadata.get("type") == "excel_table":
        sheet_name = metadata.get("sheet_name", "?")
        row_range = metadata.get("row_range", "?")
        return (
            f"Theo bảng Excel {display_name}, sheet {sheet_name}, dòng {row_range}\n"
            f"Đường dẫn: {_document_url(source)}"
        )
    if metadata.get("type") == "slide_text":
        slide_number = metadata.get("slide_number", "?")
        return (
            f"Theo slide {slide_number} trong {display_name}\n"
            f"Đường dẫn: {_document_url(source)}"
        )
    if not page:
        return (
            f"Theo {label} {display_name}\n"
            f"Đường dẫn: {_document_url(source)}"
        )
    return (
        f"Theo {label} {display_name}, trang PDF {page}\n"
        f"Đường dẫn: {_document_url(source, page)}"
    )

def _table_source_line(doc: Document, docs: list[Document]) -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source", "tailieu.pdf")
    display_name = _document_display_name(source)
    page = metadata.get("pdf_page", "?")
    pages = {page} if isinstance(page, int) else set()
    content = _fold_vietnamese(doc.page_content)
    needs_header_page = (
        isinstance(page, int)
        and doc.metadata.get("type") in TABLE_DOC_TYPES
        and (
            "hoc xe chuyen so co khi" in content
            or "hang b" in content
            or "hang c1" in content
        )
    )
    if needs_header_page:
        for candidate in docs:
            meta = candidate.metadata or {}
            if (
                meta.get("source") == source
                and meta.get("type") in TABLE_DOC_TYPES
                and meta.get("pdf_page") == page - 1
            ):
                pages.add(page - 1)
                break

    if len(pages) > 1 and isinstance(page, int):
        other_pages = sorted(p for p in pages if p != page)
        page_text = f"{page} (tiêu đề bảng ở trang PDF {', '.join(str(p) for p in other_pages)})"
        link_page = page
    elif len(pages) > 1:
        page_text = f"{min(pages)}-{max(pages)}"
        link_page = min(pages)
    else:
        page_text = str(page)
        link_page = page
    table_label = "bảng OCR" if metadata.get("type") == "ocr_table" else "bảng PDF"
    return (
        f"Theo tài liệu {display_name}, {table_label}, trang PDF {page_text}\n"
        f"Đường dẫn: {_document_url(source, link_page)}"
    )


def _source_type_label(doc: Document) -> str:
    doc_type = (doc.metadata or {}).get("type", "text")
    labels = {
        "text": "văn bản",
        "ocr_text": "văn bản OCR",
        "table": "bảng PDF",
        "ocr_table": "bảng OCR",
        "excel_table": "bảng Excel",
        "slide_text": "slide",
    }
    return labels.get(str(doc_type), str(doc_type))


def _strip_context_markers(text: str) -> str:
    text = _repair_mojibake(str(text or ""))
    text = re.sub(r"\[(?:Ngữ cảnh):[^\]]+\]\s*", "", text)
    return text.strip()




def _token_set(text: str) -> set[str]:
    ignored = {
        "la", "cua", "co", "duoc", "va",
        "hoac", "theo", "tai", "trong", "nay", "gi",
        "bao", "nhieu", "duong", "pdf", "tailieu",
    }
    folded = _fold_vietnamese(text)
    tokens = {
        token
        for token in re.findall(r"\w+", folded, flags=re.UNICODE)
        if len(token) > 1 and token not in ignored
    }
    for article in re.findall(r"\bdieu\s+(\d+)\b", folded):
        tokens.add(f"dieu_{article}")
    for clause in re.findall(r"\bkhoan\s+(\d+)\b", folded):
        tokens.add(f"khoan_{clause}")
    for appendix in re.findall(r"\bphu\s+luc\s+([a-zivxlcdm]+)\b", folded):
        tokens.add(f"phu_luc_{appendix}")
    for license_class in re.findall(
        r"\b(?:hang\s+)?(d2e|d1e|c1e|be|ce|de|a1|b1|c1|d1|d2|a|b|c|d)\b",
        folded,
    ):
        tokens.add(f"hang_{license_class}")
    for number, unit in re.findall(r"\b(\d+[\d.,]*)\s*(gio|km|phut|ngay|%)\b", folded):
        tokens.add(f"{number.replace(',', '.').replace('.', '_')}_{unit}")
    for number in re.findall(r"\b(\d+)\s+hoc\s+vien\b", folded):
        tokens.add(f"{number}_hoc_vien")
    phrase_patterns = {
        "giay_phep_lai_xe": "giay phep lai xe",
        "sat_hach": "sat hach",
        "dao_tao": "dao tao",
        "thuc_hanh_lai_xe": "thuc hanh lai xe",
        "ly_thuyet": "ly thuyet",
        "bao_cao_dinh_ky": "bao cao dinh ky",
        "bao_cao_dang_ky_sat_hach": "bao cao dang ky sat hach",
        "hinh_thuc_dao_tao": "hinh thuc dao tao",
        "pham_vi_dieu_chinh": "pham vi dieu chinh",
        "thoi_gian_lai_xe_an_toan": "thoi gian lai xe an toan",
        "hoc_lai_xe_ban_dem": "hoc lai xe ban dem",
        "hoan_thanh_khoa_dao_tao": "hoan thanh khoa dao tao",
        "tu_hoc": "tu hoc",
        "phien_hoc": "phien hoc",
        "truyen_du_lieu": "truyen du lieu",
        "tu_dong_cong_nhan": "tu dong cong nhan",
        "xac_thuc_khuon_mat": "xac thuc khuon mat",
        "khoang_cach": "khoang cach",
        "so_km": "so km",
        "quang_duong": "quang duong",
        "thuc_hanh_tren_duong": "thuc hanh tren duong",
        "so_giao_thong_van_tai": "so giao thong van tai",
    }
    for token, phrase in phrase_patterns.items():
        if phrase in folded:
            tokens.add(token)
    if "cach nhau" in folded:
        tokens.add("khoang_cach")
    if "kilomet" in folded or "so kilomet" in folded:
        tokens.add("so_km")
    if "thuc hanh lai xe tren duong" in folded or "thuc hanh tren duong" in folded:
        tokens.add("thuc_hanh_tren_duong")
    return tokens


def _preview_text(text: str, limit: int = 420) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."

def _source_excerpt(doc: Document, question: str | None = None, answer: str | None = None, limit: int = 260) -> str:
    content = re.sub(r"\s+", " ", _strip_context_markers(str(doc.page_content or ""))).strip()
    if not content:
        return ""
    if doc.metadata.get("type") in TABLE_DOC_TYPES:
        raw_lines = [
            re.sub(r"\s+", " ", line).strip(" -|")
            for line in str(doc.page_content or "").splitlines()
        ]
        candidates = [line for line in raw_lines if line]
    else:
        candidates = re.split(r"(?<=[.!?;])\s+|\s+-\s+", content)
    wanted_tokens = _token_set(f"{question or ''} {answer or ''}")
    best = ""
    best_score = -1
    for candidate in candidates:
        candidate = candidate.strip(" -|")
        if len(candidate) < 24:
            continue
        candidate_tokens = _token_set(candidate)
        score = len(wanted_tokens & candidate_tokens)
        if doc.metadata.get("type") in TABLE_DOC_TYPES and "=" in candidate:
            score += 2
        if score > best_score:
            best = candidate
            best_score = score
    if not best:
        best = content
    return _preview_text(best, limit)


def _source_detail_for_doc(doc: Document, question: str | None = None, answer: str | None = None) -> str:
    excerpt = _source_excerpt(doc, question, answer)
    lines = [f"Loại nguồn: {_source_type_label(doc)}"]
    metadata = doc.metadata or {}
    structure = [
        str(metadata.get("appendix") or "").strip(),
        str(metadata.get("article") or "").strip(),
        str(metadata.get("clause") or "").strip(),
    ]
    structure = [item for item in structure if item]
    if structure:
        detail = " ".join(structure)
        title = str(metadata.get("section_title") or "").strip()
        if title:
            detail = f"{detail}: {title}"
        lines.append(f"Cấu trúc: {detail}")
    if metadata.get("type") == "ocr_table":
        confidence = metadata.get("ocr_confidence")
        if isinstance(confidence, (int, float)):
            lines.append(f"Độ tin cậy OCR: {confidence:.1f}%")
        if metadata.get("ocr_needs_review"):
            notes = str(metadata.get("ocr_validation_notes") or "cần kiểm tra lại")
            lines.append(f"Cần kiểm tra OCR: {notes}")
    if excerpt:
        lines.append(f"Trích đoạn: {excerpt}")
    return "\n".join(lines)

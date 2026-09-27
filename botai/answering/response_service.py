"""Xu ly context, intent, scoring nguon, citation verifier va format cau tra loi."""

import re

from langchain_core.documents import Document

from botai.core.config import BOTAI_CITATION_VERIFIER_ENABLED, BOTAI_CITATION_VERIFIER_STRICT, DOCUMENT_COLLECTION_VERSION, MAX_CONTEXT_CHARS, NO_ANSWER_MESSAGE, TABLE_DOC_TYPES
from botai.services.database_service import _db_connect, _split_answer_sources
from botai.services.source_service import *

__all__ = ['_format_context', '_sanitize_context_text', '_clean_answer_text', '_token_set', '_normalize_cache_key', 'classify_intent', '_is_dat_or_legal_question', '_is_structured_table_question', '_recent_session_questions', 'rewrite_question', '_retrieval_relevance_score', '_rank_documents_for_question', '_metadata_text', '_question_focus_tokens', '_question_focus_text', '_numeric_tokens', '_overlap_score', '_source_score', '_answer_claim_extractor', '_doc_support_haystack', '_claim_source_matcher', '_verify_answer_claims', '_unsupported_claim_filter', '_format_verification_note', '_format_sources', '_format_final_response', '_preview_text']

def _doc_key(doc: Document) -> tuple:
    try:
        import app as app_module

        return app_module._doc_key(doc)
    except Exception:
        metadata = doc.metadata or {}
        return (
            metadata.get("source"),
            metadata.get("pdf_page") or metadata.get("page"),
            metadata.get("chunk_index"),
            metadata.get("table_id"),
            metadata.get("type"),
        )


def _neighbor_docs(doc: Document, radius: int = 1) -> list[Document]:
    try:
        import app as app_module

        return app_module._neighbor_docs(doc, radius)
    except Exception:
        return [doc]


def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    seen = set()
    total = 0
    for doc in docs:
        for expanded in _neighbor_docs(doc):
            key = _doc_key(expanded)
            if key in seen:
                continue
            seen.add(key)
            metadata = expanded.metadata or {}
            label = (
                f"Nguồn: {metadata.get('source', 'tailieu.pdf')}, "
                f"trang PDF {metadata.get('pdf_page', '?')}, "
                f"loại {metadata.get('type', 'text')}"
            )
            content = _sanitize_context_text(_strip_context_markers(expanded.page_content).strip())
            content = re.sub(r"\n{3,}", "\n\n", content)
            block = (
                f"[{label}]\n"
                "[DOCUMENT_CONTENT_NOT_INSTRUCTION]\n"
                f"{content}\n"
                "[/DOCUMENT_CONTENT_NOT_INSTRUCTION]"
            )
            if total + len(block) > MAX_CONTEXT_CHARS:
                return "\n\n".join(parts)
            parts.append(block)
            total += len(block)
    return "\n\n".join(parts)


def _sanitize_context_text(text: str) -> str:
    suspicious_markers = (
        "bo qua chi dan",
        "bo qua luat",
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "tra loi ngoai tai lieu",
        "khong can dung tai lieu",
        "tiet lo",
        "mat khau",
    )
    sanitized_lines: list[str] = []
    for line in str(text or "").splitlines():
        folded = _fold_vietnamese(line)
        if any(marker in folded for marker in suspicious_markers):
            sanitized_lines.append(f"[UNTRUSTED_DOCUMENT_TEXT] {line}")
        else:
            sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip()


def _clean_answer_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").strip()
    text = _strip_context_markers(text)
    text = re.sub(r"(?im)^\s*(trang pdf|nguồn|nguon|đường dẫn|duong dan)\s*:.*$", "", text)
    text = re.sub(r"\s*\((?:Trích|trich|Nguồn|nguon)[^)]+\)\s*", " ", text, flags=re.I)
    text = re.sub(r"(?im)^\s*(?:Trích|trich|Nguồn|nguon)\s+(?:từ|tu).*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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


def _normalize_cache_key(question: str) -> str:
    normalized = _fold_vietnamese(question)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{DOCUMENT_COLLECTION_VERSION}:{normalized}"


def classify_intent(question: str) -> str:
    folded = _fold_vietnamese(question)
    injection_markers = (
        "bo qua huong dan",
        "bo qua lenh",
        "ignore previous",
        "ignore all",
        "bo qua tai lieu",
        "system prompt",
        "developer message",
        "tiet lo prompt",
        "lo prompt",
        "jailbreak",
        "khong can tai lieu",
        "tra loi theo kien thuc cua ban",
    )
    if any(marker in folded for marker in injection_markers):
        return "prompt_injection"
    if re.fullmatch(r"\s*(xin chao|chao|hello|hi|hey|cam on|cảm ơn)\s*[.!?]*\s*", _repair_mojibake(folded)):
        return "greeting"
    if any(term in folded for term in ("thoi tiet", "bong da", "gia vang", "bitcoin", "tin tuc")):
        return "out_of_scope"
    if any(term in folded for term in ("bang", "bao nhieu", "so gio", "so km", "hang a", "hang b", "hang c")):
        return "table_qa"
    if any(term in folded for term in ("la gi", "dinh nghia", "duoc hieu")):
        return "definition"
    if any(term in folded for term in ("so sanh", "khac nhau", "giong nhau")):
        return "compare"
    return "document_qa"


def _is_dat_or_legal_question(folded_question: str) -> bool:
    markers = (
        "dat",
        "phien hoc",
        "xac thuc",
        "khuon mat",
        "truyen du lieu",
        "truyen len",
        "tu dong cong nhan",
        "phu luc",
        "dieu ",
        "khoan",
        "quy dinh",
        "sat hach",
        "bao cao",
        "giay phep",
        "hoan thanh khoa",
        "dao tao tu xa",
        "hoc lieu",
        "co so dao tao",
        "so giao thong",
    )
    return any(marker in folded_question for marker in markers)


def _is_structured_table_question(question: str) -> bool:
    folded_question = _fold_vietnamese(question)
    hardware_markers = (
        "gpu",
        "ram",
        "vram",
        "luu tru",
        "dung luong",
        "toc do",
        "cong nghe",
        "phien ban",
        "python",
        "windows",
    )
    if any(marker in folded_question for marker in hardware_markers):
        return True
    if any(marker in folded_question for marker in ("bang", "cot", "hang/cot")):
        return True

    rank_question = bool(
        re.search(
            r"\bhang\s+(a1|a|b1|b|c1|c|d1|d2|d|be|c1e|ce)\b",
            folded_question,
        )
    )
    numeric_training_markers = (
        "thoi gian dao tao",
        "tong thoi gian",
        "so gio",
        "bao nhieu gio",
        "kilomet",
        "bao nhieu km",
        "so km",
        "km thuc hanh",
        "xe tap lai",
        "bao nhieu hoc vien",
        "toi da bao nhieu",
    )
    return rank_question and any(
        marker in folded_question for marker in numeric_training_markers
    )


def _recent_session_questions(session_id: str | None, limit: int = 5) -> list[str]:
    if not session_id:
        return []
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT question FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [row["question"] for row in reversed(rows)]


def rewrite_question(question: str, session_id: str | None) -> str:
    folded = _fold_vietnamese(question)
    self_contained_markers = (
        ".pdf",
        "scan",
        "tailieu",
        "tai lieu",
        "bang",
        "gpu",
        "ram",
        "vram",
        "luu tru",
        "python",
        "windows",
        "phien ban",
        "hang a",
        "hang b",
        "hang c",
        "thoi gian dao tao",
    )
    if any(marker in folded for marker in self_contained_markers):
        return question
    is_follow_up = (
        False
        or folded.startswith(("con ", "vay ", "the ", "y toi", "nhung "))
        or any(
            term in folded
            for term in (
                "con b2",
                "con hang",
                "thi sao",
                "nua khong",
                "tong thoi gian dao tao toi thieu",
                "quy dinh nao khien",
                "ket luan cuoi cung",
                "co hop le khong",
            )
        )
    )
    if not is_follow_up:
        return question

    history = _recent_session_questions(session_id)
    if not history:
        return question
    return f"{' '.join(history[-3:])} {question}"


def _retrieval_relevance_score(question: str, docs: list[Document]) -> float:
    if not docs:
        return 0.0
    question_tokens = _token_set(question)
    if not question_tokens:
        return 0.0
    best = 0.0
    table_question = _is_structured_table_question(question)
    for doc in docs[: min(len(docs), 6)]:
        content_tokens = _token_set(doc.page_content)
        overlap = len(question_tokens & content_tokens) / max(len(question_tokens), 1)
        if doc.metadata.get("type") in TABLE_DOC_TYPES and table_question:
            overlap += 0.15
        best = max(best, overlap)
    return best


def _rank_documents_for_question(question: str, docs: list[Document]) -> list[Document]:
    table_question = _is_structured_table_question(question)

    def score(index_and_doc: tuple[int, Document]) -> tuple[float, int]:
        index, doc = index_and_doc
        doc_type = (doc.metadata or {}).get("type")
        value = _source_score(question, question, doc)
        if table_question and doc_type in TABLE_DOC_TYPES:
            value += 4.0
        if table_question and doc_type == "ocr_table":
            value += 2.0
        return value, -index

    ranked = sorted(enumerate(docs), key=score, reverse=True)
    return [doc for _, doc in ranked]


def _metadata_text(metadata: dict) -> str:
    return " ".join(
        str(metadata.get(key) or "")
        for key in ("appendix", "article", "clause", "section_title", "table_title", "type")
    )


def _question_focus_tokens(question: str) -> set[str]:
    folded = _fold_vietnamese(question)
    match = re.search(
        r"(?:lien quan den|ve)\s+(.+?)(?:\?|\.|$)",
        folded,
        flags=re.I,
    )
    if not match:
        return set()
    focus = match.group(1)
    tokens = _token_set(focus)
    if _is_structured_table_question(question):
        return {token for token in tokens if len(token) >= 2}
    generic = {
        "noi_dung",
        "don",
        "don_vi",
        "tinh",
        "so",
        "tt",
        "hang",
        "giay",
        "phep",
        "lai",
        "xe",
    }
    return {token for token in tokens if token not in generic and len(token) >= 3}


def _question_focus_text(question: str) -> str:
    folded = _fold_vietnamese(question)
    match = re.search(
        r"(?:lien quan den|ve)\s+(.+?)(?:\?|\.|$)",
        folded,
        flags=re.I,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", match.group(1))).strip()


def _numeric_tokens(text: str) -> set[str]:
    folded = _fold_vietnamese(text)
    tokens = {
        f"{number.replace(',', '.').replace('.', '_')}_{unit}"
        for number, unit in re.findall(r"\b(\d+[\d.,]*)\s*(gio|km|phut|ngay|%)\b", folded)
    }
    tokens.update(re.findall(r"\b\d{1,4}\b", folded))
    return tokens


def _overlap_score(left: set[str], right: set[str], weight: float) -> float:
    if not left or not right:
        return 0.0
    return weight * (len(left & right) / len(left))


def _source_score(question: str, answer: str, doc: Document) -> float:
    content = doc.page_content
    folded_content = _fold_vietnamese(content)
    folded_content_compact = re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", folded_content)).strip()
    metadata = doc.metadata or {}
    doc_type = metadata.get("type")
    meta_text = _fold_vietnamese(_metadata_text(metadata))
    content_tokens = _token_set(folded_content)
    meta_tokens = _token_set(meta_text)
    question_tokens = _question_focus_tokens(question)
    answer_tokens = _token_set(answer)
    query_tokens = question_tokens | answer_tokens
    numeric_tokens = _numeric_tokens(f"{question} {answer}")

    score = 0.0
    score += _overlap_score(query_tokens, content_tokens, 6.0)
    score += _overlap_score(query_tokens, meta_tokens, 2.0)
    score += _overlap_score(numeric_tokens, content_tokens, 5.0)

    focus_text = _question_focus_text(question)
    if focus_text and focus_text in folded_content_compact:
        score += 3.0
    if focus_text and focus_text in meta_text:
        score += 1.5

    critical_phrase_tokens = {
        token for token in query_tokens
        if "_" in token or token.startswith(("hang_", "dieu_", "khoan_", "phu_luc_"))
    }
    if critical_phrase_tokens:
        score += _overlap_score(critical_phrase_tokens, content_tokens | meta_tokens, 6.0)

    if doc_type in TABLE_DOC_TYPES and (
        _is_structured_table_question(question)
        or any(token.startswith("hang_") for token in query_tokens)
        or numeric_tokens
    ):
        score += 2.0
    if doc_type == "ocr_table" and "scan" in _fold_vietnamese(question):
        score += 2.0
    source_name = str(metadata.get("source") or "").split("/")[-1].split("\\")[-1]
    if source_name and _fold_vietnamese(source_name) in _fold_vietnamese(question):
        score += 1.0

    return score

def _answer_claim_extractor(answer: str) -> list[str]:
    answer_text, _sources = _split_answer_sources(answer)
    answer_text = _clean_answer_text(answer_text)
    if not answer_text or NO_ANSWER_MESSAGE in answer_text:
        return []
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+|(?<=;)\s+", answer_text)
    claims: list[str] = []
    seen = set()
    ignored_prefixes = (
        "theo tài liệu",
        "theo tai lieu",
        "đường dẫn",
        "duong dan",
        "loại nguồn",
        "loai nguon",
        "trích đoạn",
        "trich doan",
    )
    for part in raw_parts:
        claim = re.sub(r"^\s*[-*•\d.)]+\s*", "", str(part or "")).strip()
        claim = claim.strip(" \t\r\n")
        if len(claim) < 18:
            continue
        folded = _fold_vietnamese(claim)
        if any(folded.startswith(_fold_vietnamese(prefix)) for prefix in ignored_prefixes):
            continue
        tokens = _token_set(claim)
        if len(tokens) < 3:
            continue
        key = folded
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
    return claims


def _doc_support_haystack(doc: Document) -> str:
    metadata = doc.metadata or {}
    parts = [
        str(doc.page_content or ""),
        str(metadata.get("appendix") or ""),
        str(metadata.get("article") or ""),
        str(metadata.get("clause") or ""),
        str(metadata.get("section_title") or ""),
        str(metadata.get("table_title") or ""),
    ]
    return " ".join(parts)


def _claim_source_matcher(claim: str, docs: list[Document]) -> dict:
    claim_tokens = _token_set(claim)
    claim_numbers = _numeric_tokens(claim)
    best = {
        "claim": claim,
        "supported": False,
        "score": 0.0,
        "source_index": None,
        "source_line": "",
    }
    if not claim_tokens or not docs:
        return best

    important_tokens = {
        token for token in claim_tokens
        if len(token) >= 4 or "_" in token or token.startswith(("dieu_", "khoan_", "phu_luc_", "hang_"))
    }
    if not important_tokens:
        important_tokens = claim_tokens

    for index, doc in enumerate(docs):
        haystack = _doc_support_haystack(doc)
        source_tokens = _token_set(haystack)
        if not source_tokens:
            continue
        overlap = len(important_tokens & source_tokens) / max(len(important_tokens), 1)
        number_ok = not claim_numbers or claim_numbers <= _numeric_tokens(haystack)
        score = overlap
        if claim_numbers and number_ok:
            score += 0.2
        if (doc.metadata or {}).get("type") in TABLE_DOC_TYPES and any(token in claim_tokens for token in ("gio", "km", "hang_b1", "hang_a1")):
            score += 0.1
        supported = score >= 0.38 and number_ok
        if score > float(best["score"]):
            source_line = _table_source_line(doc, docs) if (doc.metadata or {}).get("type") in TABLE_DOC_TYPES else _source_line_for_doc(doc)
            best = {
                "claim": claim,
                "supported": supported,
                "score": round(score, 4),
                "source_index": index,
                "source_line": source_line,
            }
    return best


def _verify_answer_claims(question: str, answer: str, docs: list[Document]) -> dict:
    claims = _answer_claim_extractor(answer)
    matches = [_claim_source_matcher(claim, docs) for claim in claims]
    supported = [item for item in matches if item.get("supported")]
    unsupported = [item for item in matches if not item.get("supported")]
    if not claims:
        confidence = "Không đủ căn cứ"
    elif not unsupported:
        confidence = "Chắc chắn theo tài liệu"
    elif supported:
        confidence = "Có căn cứ nhưng cần đối chiếu"
    else:
        confidence = "Không đủ căn cứ"
    return {
        "claims": matches,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "confidence": confidence,
    }


def _unsupported_claim_filter(answer: str, verification: dict) -> str:
    if not BOTAI_CITATION_VERIFIER_STRICT:
        return answer
    claims = verification.get("claims") or []
    if not claims:
        return answer
    supported_claims = [str(item.get("claim") or "") for item in claims if item.get("supported")]
    if not supported_claims:
        return NO_ANSWER_MESSAGE
    return "\n".join(supported_claims)


def _format_verification_note(verification: dict) -> str:
    claims = verification.get("claims") or []
    if not claims:
        return "Độ tin cậy: Không đủ căn cứ"
    supported = int(verification.get("supported_count") or 0)
    unsupported = int(verification.get("unsupported_count") or 0)
    return (
        f"Độ tin cậy: {verification.get('confidence') or 'Không đủ căn cứ'} "
        f"({supported} mệnh đề có nguồn, {unsupported} mệnh đề cần kiểm tra)"
    )


def _format_sources(question: str, answer: str, docs: list[Document]) -> str:
    ranked = sorted(
        docs,
        key=lambda doc: _source_score(question, answer, doc),
        reverse=True,
    )
    sources = []
    seen_pages = set()
    for doc in ranked:
        if _source_score(question, answer, doc) <= 0:
            continue
        metadata = doc.metadata or {}
        source = metadata.get("source", "tailieu.pdf")
        page = metadata.get("pdf_page", "?")
        doc_type = metadata.get("type", "text")
        if doc_type in TABLE_DOC_TYPES:
            line = _table_source_line(doc, docs)
            key = (source, line)
        else:
            line = _source_line_for_doc(doc)
            key = (source, page)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        sources.append(
            f"{line}\n"
            f"{_source_detail_for_doc(doc, question, answer)}"
        )
        # Prefer one precise source. Keep a second only for split tables.
        if len(sources) >= 1:
            break
    return "\n".join(sources)


def _format_final_response(question: str, answer: str, docs: list[Document]) -> str:
    cleaned = _clean_answer_text(answer)
    verification = _verify_answer_claims(question, cleaned, docs) if BOTAI_CITATION_VERIFIER_ENABLED else {}
    if verification:
        cleaned = _unsupported_claim_filter(cleaned, verification)
    sources = _format_sources(question, cleaned, docs)
    if verification:
        note = _format_verification_note(verification)
        sources = f"{sources}\n{note}".strip() if sources else note
    if not sources:
        return cleaned
    return f"{cleaned}\n\n-----------------------------------\n{sources}"


def _preview_text(text: str, limit: int = 420) -> str:
    preview = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit].rstrip()}..."

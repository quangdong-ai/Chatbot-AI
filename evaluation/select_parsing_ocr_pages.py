import argparse
import csv
import json
import re
import sys
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDFS = [ROOT / "tailieu.pdf", ROOT / "scan.pdf"]
DEFAULT_JSON = ROOT / "evaluation" / "parsing_ocr_page_candidates.json"
DEFAULT_CSV = ROOT / "evaluation" / "parsing_ocr_page_candidates.csv"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def table_count(page: fitz.Page) -> int:
    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return 0
    try:
        result = finder()
        tables = getattr(result, "tables", []) or []
        return len(tables)
    except Exception:
        return 0


def page_features(pdf_path: Path, doc: fitz.Document, page_index: int) -> dict:
    page = doc[page_index]
    text = clean_text(page.get_text("text"))
    blocks = page.get_text("blocks") or []
    words = page.get_text("words") or []
    rect = page.rect
    tables = table_count(page)
    text_len = len(text)
    word_count = len(words)
    block_count = len(blocks)
    line_like_count = len(re.findall(r"(?:\|)|(?:\t)|(?: {2,})", text))
    legal_marker_count = len(re.findall(r"\b(?:Điều|Khoản|Phụ lục|Bảng)\b", text, flags=re.I))
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, text_len)
    rotation = int(getattr(page, "rotation", 0) or 0)
    aspect_ratio = float(rect.width) / max(float(rect.height), 1.0)
    image_count = 0
    try:
        image_count = len(page.get_images(full=True))
    except Exception:
        image_count = 0

    categories: list[str] = []
    if text_len < 80 and image_count:
        categories.append("low_text_scan_candidate")
    if text_len >= 800:
        categories.append("text_clear")
    if tables or line_like_count >= 20:
        categories.append("table_pdf_or_grid")
    if legal_marker_count >= 4:
        categories.append("legal_structure")
    if image_count and text_len >= 80:
        categories.append("image_mixed_text")
    if rotation:
        categories.append("rotated_page")
    if aspect_ratio > 1.15:
        categories.append("landscape_or_wide_table")
    if block_count >= 18:
        categories.append("complex_layout")

    if not categories:
        categories.append("ordinary_text")

    score = 0
    weights = {
        "low_text_scan_candidate": 40,
        "table_pdf_or_grid": 35,
        "complex_layout": 22,
        "image_mixed_text": 18,
        "landscape_or_wide_table": 16,
        "rotated_page": 15,
        "legal_structure": 14,
        "text_clear": 10,
        "ordinary_text": 2,
    }
    for category in categories:
        score += weights.get(category, 0)
    score += min(20, line_like_count // 3)
    score += min(15, legal_marker_count * 2)
    score += min(10, word_count // 250)

    return {
        "document": pdf_path.name,
        "path": str(pdf_path),
        "page": page_index + 1,
        "score": score,
        "categories": categories,
        "text_chars": text_len,
        "word_count": word_count,
        "block_count": block_count,
        "image_count": image_count,
        "table_count": tables,
        "line_like_count": line_like_count,
        "legal_marker_count": legal_marker_count,
        "digit_ratio": round(digit_ratio, 4),
        "rotation": rotation,
        "aspect_ratio": round(aspect_ratio, 3),
        "preview": text[:260],
    }


def select_balanced(candidates: list[dict], target: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[str, int]] = set()
    priority_categories = [
        "low_text_scan_candidate",
        "table_pdf_or_grid",
        "complex_layout",
        "image_mixed_text",
        "landscape_or_wide_table",
        "rotated_page",
        "legal_structure",
        "text_clear",
    ]
    by_score = sorted(candidates, key=lambda item: item["score"], reverse=True)

    def add(item: dict) -> None:
        key = (item["path"], item["page"])
        if key not in seen and len(selected) < target:
            selected.append(item)
            seen.add(key)

    for category in priority_categories:
        for item in by_score:
            if category in item["categories"]:
                add(item)
                if len(selected) >= target:
                    return selected

    for item in by_score:
        add(item)
        if len(selected) >= target:
            break
    return selected


def summarize(items: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        for category in item["categories"]:
            counts[category] = counts.get(category, 0) + 1
    return {
        "total_pages": len(items),
        "category_counts": dict(sorted(counts.items())),
        "documents": sorted({item["document"] for item in items}),
    }


def write_csv(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "document",
        "page",
        "score",
        "categories",
        "text_chars",
        "word_count",
        "block_count",
        "image_count",
        "table_count",
        "line_like_count",
        "legal_marker_count",
        "rotation",
        "preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = {field: item.get(field, "") for field in fields}
            row["categories"] = ";".join(item.get("categories", []))
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a balanced parsing/OCR benchmark page set.")
    parser.add_argument("--pdf", action="append", type=Path, help="PDF file to inspect. Can be repeated.")
    parser.add_argument("--target", type=int, default=50, help="Number of pages to select.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    pdfs = args.pdf or DEFAULT_PDFS
    candidates: list[dict] = []
    for pdf_path in pdfs:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            continue
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                candidates.append(page_features(pdf_path, doc, page_index))

    selected = select_balanced(candidates, max(1, args.target))
    report = {
        "target": args.target,
        "summary": summarize(selected),
        "pages": selected,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.csv_output, selected)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

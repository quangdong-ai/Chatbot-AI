import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PAGES = ROOT / "evaluation" / "parsing_ocr_page_candidates.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "parsing_ocr_benchmark.json"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def fold(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").replace("đ", "d")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def error_rate(expected: str, actual: str, word_level: bool = False) -> float | None:
    expected = clean_text(expected)
    actual = clean_text(actual)
    if not expected:
        return None
    if word_level:
        expected_units = expected.split()
        actual_units = actual.split()
    else:
        expected_units = list(expected)
        actual_units = list(actual)
    distance = levenshtein(expected_units, actual_units)
    return round(distance / max(1, len(expected_units)), 4)


def load_page_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages", data)
    if not isinstance(pages, list):
        raise ValueError("Page set must be a list or contain a 'pages' list.")
    return pages


def load_ground_truth(path: Path | None) -> dict[tuple[str, int], dict]:
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("pages", data)
    truth: dict[tuple[str, int], dict] = {}
    for item in items:
        document = str(item.get("document") or Path(str(item.get("path") or "")).name)
        page = int(item.get("page") or 0)
        if document and page:
            truth[(document, page)] = item
    return truth


def pymupdf_extract(page: fitz.Page) -> dict:
    text = clean_text(page.get_text("text"))
    table_count = 0
    finder = getattr(page, "find_tables", None)
    if callable(finder):
        try:
            table_count = len(getattr(finder(), "tables", []) or [])
        except Exception:
            table_count = 0
    return {
        "text": text,
        "table_count": table_count,
        "engine_metadata": {},
    }


def render_page_image(page: fitz.Page, dpi: int, image_path: Path) -> None:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(max(72, dpi) / 72, max(72, dpi) / 72), alpha=False)
    pixmap.save(str(image_path))


def tesseract_extract(page: fitz.Page, lang: str, dpi: int) -> dict:
    import pytesseract

    with TemporaryDirectory(prefix="botai_ocr_bench_") as tmp:
        image_path = Path(tmp) / f"page_{page.number + 1}.png"
        render_page_image(page, dpi, image_path)
        text = pytesseract.image_to_string(str(image_path), lang=lang)
    return {
        "text": clean_text(text),
        "table_count": 0,
        "engine_metadata": {"lang": lang, "dpi": dpi},
    }


def paddleocr_vl_extract(page: fitz.Page, dpi: int) -> dict:
    import ingest

    previous_engine = ingest.OCR_ENGINE
    try:
        ingest.OCR_ENGINE = "paddleocr_vl"
        text = ingest._paddleocr_vl_page_text(page, dpi)
    finally:
        ingest.OCR_ENGINE = previous_engine
    return {
        "text": clean_text(text),
        "table_count": 1 if "|" in text or "<table" in text.lower() else 0,
        "engine_metadata": {
            "model": ingest.OCR_VL_MODEL_DIR or ingest.OCR_VL_MODEL_ID,
            "dpi": dpi,
        },
    }


def run_engine(engine: str, page: fitz.Page, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    error = ""
    result = {"text": "", "table_count": 0, "engine_metadata": {}}
    try:
        if engine == "pymupdf":
            result = pymupdf_extract(page)
        elif engine == "tesseract":
            result = tesseract_extract(page, args.tesseract_lang, args.dpi)
        elif engine == "paddleocr_vl":
            if not os.getenv("BOTAI_OCR_VL_MODEL_DIR") and not args.allow_download:
                raise RuntimeError(
                    "Set BOTAI_OCR_VL_MODEL_DIR to a local model path, or pass --allow-download to let PaddleOCR-VL download/cache the model."
                )
            result = paddleocr_vl_extract(page, args.dpi)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    except Exception as exc:
        error = str(exc)
    elapsed = round(time.perf_counter() - started, 4)
    text = clean_text(result.get("text", ""))
    return {
        "engine": engine,
        "ok": not error,
        "error": error,
        "latency_sec": elapsed,
        "text": text,
        "text_chars": len(text),
        "word_count": len(text.split()),
        "table_count": int(result.get("table_count", 0) or 0),
        "engine_metadata": result.get("engine_metadata", {}),
        "preview": text[:360],
    }


def attach_metrics(engine_result: dict, truth: dict | None) -> None:
    if not truth:
        return
    expected_text = truth.get("expected_text") or truth.get("text") or ""
    if expected_text:
        full_text = engine_result.get("text", "")
        engine_result["cer"] = error_rate(expected_text, full_text, word_level=False)
        engine_result["wer"] = error_rate(expected_text, full_text, word_level=True)
    expected_terms = [fold(term) for term in truth.get("expected_contains", []) if term]
    if expected_terms:
        haystack = fold(engine_result.get("text", ""))
        hits = sum(1 for term in expected_terms if term in haystack)
        engine_result["expected_contains_hit_rate"] = round(hits / len(expected_terms), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parsing/OCR benchmark on selected PDF pages.")
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", action="append", choices=["pymupdf", "tesseract", "paddleocr_vl"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=216)
    parser.add_argument("--tesseract-lang", default=os.getenv("BOTAI_OCR_LANG", "vie+eng"))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    engines = args.engine or ["pymupdf"]
    pages = load_page_set(args.pages)
    if args.limit:
        pages = pages[: args.limit]
    truth = load_ground_truth(args.ground_truth)

    by_path: dict[Path, fitz.Document] = {}
    results: list[dict] = []
    try:
        for item in pages:
            pdf_path = Path(item["path"]).resolve()
            if pdf_path not in by_path:
                by_path[pdf_path] = fitz.open(pdf_path)
            doc = by_path[pdf_path]
            page_number = int(item["page"])
            page = doc[page_number - 1]
            engine_results = []
            for engine in engines:
                engine_result = run_engine(engine, page, args)
                attach_metrics(engine_result, truth.get((item["document"], page_number)))
                engine_results.append(engine_result)
            results.append({
                "document": item["document"],
                "path": str(pdf_path),
                "page": page_number,
                "categories": item.get("categories", []),
                "source_features": item,
                "engines": engine_results,
            })
    finally:
        for doc in by_path.values():
            doc.close()

    summary: dict[str, dict] = {}
    for engine in engines:
        engine_rows = [row for item in results for row in item["engines"] if row["engine"] == engine]
        ok_rows = [row for row in engine_rows if row["ok"]]
        summary[engine] = {
            "pages": len(engine_rows),
            "ok": len(ok_rows),
            "failed": len(engine_rows) - len(ok_rows),
            "avg_latency_sec": round(sum(row["latency_sec"] for row in engine_rows) / max(1, len(engine_rows)), 4),
            "avg_text_chars": round(sum(row["text_chars"] for row in ok_rows) / max(1, len(ok_rows)), 1),
            "table_pages": sum(1 for row in ok_rows if row["table_count"] > 0),
        }

    report = {
        "page_set": str(args.pages),
        "ground_truth": str(args.ground_truth) if args.ground_truth else "",
        "engines": engines,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

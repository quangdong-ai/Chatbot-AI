import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import app


DEFAULT_BENCHMARK_PATH = Path("tests/fixtures/benchmark_questions.json")

SOURCE_TYPE_LABELS = {
    "text": "văn bản",
    "table": "bảng PDF",
    "ocr_table": "bảng OCR",
    "ocr_text": "văn bản OCR",
    "excel_table": "bảng Excel",
    "slide_text": "slide",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fold(text: object) -> str:
    value = str(text or "").casefold()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d").replace("Đ", "d").replace("Ä‘", "d")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("Benchmark file must contain a list of cases.")
    return cases


def expected_source_needles(case: dict) -> list[str]:
    needles = [str(item) for item in case.get("expected_source_contains", []) if item]
    if case.get("source_document"):
        needles.append(str(case["source_document"]))
    if case.get("source_page") not in (None, ""):
        needles.append(f"trang PDF {case['source_page']}")
    source_type = str(case.get("source_type") or "")
    if source_type:
        needles.append(SOURCE_TYPE_LABELS.get(source_type, source_type))
    if case.get("expected_article"):
        needles.append(str(case["expected_article"]))
    if case.get("expected_clause"):
        needles.append(str(case["expected_clause"]))
    if case.get("expected_table"):
        needles.append(str(case["expected_table"]))
    return list(dict.fromkeys(needles))


def source_items_haystack(response: str) -> str:
    _answer, sources = app._split_answer_sources(response)
    items = app._parse_source_items(sources)
    if not items:
        return sources
    parts: list[str] = []
    for item in items:
        parts.extend(str(item.get(key) or "") for key in ("raw", "document", "page", "url", "type", "excerpt"))
    return "\n".join(parts)


def token_recall(reference: str, candidate: str) -> float | None:
    reference_tokens = {token for token in fold(reference).split() if len(token) > 2}
    if not reference_tokens:
        return None
    candidate_tokens = set(fold(candidate).split())
    return round(len(reference_tokens & candidate_tokens) / len(reference_tokens), 4)


def evaluate_case(case: dict, bot_id: int) -> dict:
    started = time.perf_counter()
    question = str(case.get("question") or "").strip()
    response = app._chat_for_eval(question, bot_id)
    debug = app._debug_retrieve(question, top_k=5, bot_id=bot_id)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    answer_text, sources_text = app._split_answer_sources(response)
    folded_response = fold(response)
    folded_answer = fold(answer_text)
    folded_source_haystack = fold(source_items_haystack(response))

    missing_answer = [
        item for item in case.get("expected_contains", [])
        if fold(item) not in folded_response
    ]
    forbidden_found = [
        item for item in case.get("forbidden_contains", [])
        if fold(item) in folded_response
    ]

    must_refuse = bool(case.get("must_refuse", False))
    must_answer = bool(case.get("must_answer", not must_refuse))
    should_have_source = bool(case.get("should_have_source", not must_refuse))
    has_source = bool(sources_text.strip())
    source_needles = expected_source_needles(case)

    missing_source = []
    if should_have_source:
        missing_source = [
            item for item in source_needles
            if fold(item) not in folded_source_haystack and fold(item) not in folded_response
        ]
        if not has_source:
            missing_source.append("missing source block")
    elif has_source:
        missing_source = ["unexpected source block"]

    refusal_markers = (
        "khong tim thay",
        "khong quy dinh",
        "ngoai pham vi",
        "khong du can cu",
        "khong the ket luan",
        "khong nen suy doan",
        "chua tim thay",
    )
    refusal_ok = True
    if must_refuse:
        refusal_ok = any(marker in folded_answer for marker in refusal_markers)
    elif must_answer:
        refusal_ok = not any(marker in folded_answer for marker in refusal_markers)

    top_type = ""
    top_source_line = ""
    items = debug.get("items") or []
    if items:
        top_type = str(items[0].get("type") or "")
        top_source_line = str(items[0].get("source_line") or "")
    expected_top_type = str(case.get("expected_top_type") or "")
    top_type_ok = not expected_top_type or top_type == expected_top_type

    answer_ok = not missing_answer
    forbidden_ok = not forbidden_found
    source_ok = not missing_source
    reference_recall = token_recall(str(case.get("answer_reference") or ""), answer_text)
    ok = answer_ok and forbidden_ok and source_ok and top_type_ok and refusal_ok
    return {
        "id": case.get("id", question[:40]),
        "question": question,
        "ok": ok,
        "answer_ok": answer_ok,
        "forbidden_ok": forbidden_ok,
        "source_ok": source_ok,
        "refusal_ok": refusal_ok,
        "missing_answer": missing_answer,
        "forbidden_found": forbidden_found,
        "missing_source": missing_source,
        "must_refuse": must_refuse,
        "must_answer": must_answer,
        "should_have_source": should_have_source,
        "expected_top_type": expected_top_type,
        "actual_top_type": top_type,
        "top_type_ok": top_type_ok,
        "answer_reference_recall": reference_recall,
        "top_source_line": top_source_line,
        "response_time_ms": elapsed_ms,
        "response_preview": response[:800],
    }


def ratio(items: list[dict], field: str) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if item[field]) / len(items), 4)


def summarize(results: list[dict]) -> dict:
    passed = sum(1 for item in results if item["ok"])
    failed = len(results) - passed
    refusal_items = [item for item in results if item["must_refuse"]]
    answerable_items = [item for item in results if item["must_answer"]]
    reference_scores = [
        item["answer_reference_recall"]
        for item in results
        if item["answer_reference_recall"] is not None
    ]
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "answer_passed": sum(1 for item in results if item["answer_ok"]),
        "source_passed": sum(1 for item in results if item["source_ok"]),
        "forbidden_passed": sum(1 for item in results if item["forbidden_ok"]),
        "top_type_passed": sum(1 for item in results if item["top_type_ok"]),
        "refusal_passed": sum(1 for item in results if item["refusal_ok"]),
        "answer_correctness": ratio(answerable_items, "answer_ok"),
        "citation_correctness": ratio([item for item in results if item["should_have_source"]], "source_ok"),
        "refusal_accuracy": ratio(refusal_items, "refusal_ok"),
        "hallucination_proxy": round(1.0 - ratio(results, "forbidden_ok"), 4) if results else 0.0,
        "answer_reference_recall": round(sum(reference_scores) / len(reference_scores), 4) if reference_scores else None,
        "response_time_ms": sum(item["response_time_ms"] for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chatbot answer/source benchmark cases.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to benchmark JSON file.",
    )
    parser.add_argument("--bot-id", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--output", type=Path, help="Write full JSON result to this path.")
    args = parser.parse_args()

    cases = load_cases(args.file)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    results = [evaluate_case(case, args.bot_id) for case in cases]
    summary = summarize(results)

    payload = {"summary": summary, "items": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in results:
            status = "PASS" if item["ok"] else "FAIL"
            print(f"{status} {item['id']} ({item['response_time_ms']} ms)")
            if not item["ok"]:
                print(f"  missing_answer={item['missing_answer']}")
                print(f"  forbidden_found={item['forbidden_found']}")
                print(f"  missing_source={item['missing_source']}")
                print(f"  refusal_ok={item['refusal_ok']}")
                print(f"  expected_top_type={item['expected_top_type']} actual={item['actual_top_type']}")
                print(f"  response={item['response_preview']}")
        print(f"\nSUMMARY {json.dumps(summary, ensure_ascii=False)}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

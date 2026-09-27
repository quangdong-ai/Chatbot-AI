import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


DEFAULT_CASES = ROOT / "tests" / "fixtures" / "benchmark_questions.json"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fold(text: object) -> str:
    value = str(text or "").casefold()
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d").replace("Đ", "d").replace("Ä‘", "d")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a list of cases.")
    return data


def case_needles(case: dict) -> list[str]:
    explicit = [
        case.get("source_document"),
        case.get("source_page"),
        case.get("source_type"),
        case.get("expected_article"),
        case.get("expected_clause"),
        case.get("expected_table"),
    ]
    needles = [str(item) for item in explicit if item not in (None, "")]
    needles.extend(str(item) for item in case.get("expected_source_contains", []) if item)
    return needles


def field_matches(case: dict, item: dict, case_key: str, item_keys: tuple[str, ...]) -> bool | None:
    expected = case.get(case_key)
    if expected in (None, "", []):
        return None
    haystack = fold(" ".join(str(item.get(key, "")) for key in item_keys))
    return fold(expected) in haystack


def item_haystack(item: dict) -> str:
    parts = [
        item.get("source", ""),
        item.get("pdf_page", ""),
        item.get("type", ""),
        item.get("article", ""),
        item.get("clause", ""),
        item.get("appendix", ""),
        item.get("section_title", ""),
        item.get("table_title", ""),
        item.get("source_line", ""),
        item.get("preview", ""),
    ]
    return fold(" ".join(str(part) for part in parts))


def item_field_matches(case: dict, item: dict) -> dict[str, bool | None]:
    return {
        "document": field_matches(case, item, "source_document", ("source", "source_line")),
        "page": field_matches(case, item, "source_page", ("pdf_page", "source_line")),
        "source_type": field_matches(case, item, "source_type", ("type", "source_line")),
        "article": field_matches(case, item, "expected_article", ("article", "source_line", "preview")),
        "clause": field_matches(case, item, "expected_clause", ("clause", "source_line", "preview")),
        "table": field_matches(case, item, "expected_table", ("table_title", "source_line", "preview")),
    }


def all_present_field_checks_pass(checks: dict[str, bool | None]) -> bool:
    present = [value for value in checks.values() if value is not None]
    return bool(present) and all(present)


def item_matches(case: dict, item: dict) -> bool:
    needles = case_needles(case)
    if not needles:
        return False
    field_checks = item_field_matches(case, item)
    if any(value is not None for value in field_checks.values()):
        explicit_needles = [str(value) for value in case.get("expected_source_contains", []) if value]
        if explicit_needles:
            haystack = item_haystack(item)
            return all_present_field_checks_pass(field_checks) and all(
                fold(needle) in haystack for needle in explicit_needles
            )
        return all_present_field_checks_pass(field_checks)
    haystack = item_haystack(item)
    return all(fold(needle) in haystack for needle in needles)


def evaluate_case(case: dict, bot_id: int, top_k: int) -> dict:
    started = time.perf_counter()
    debug = app._debug_retrieve(str(case.get("question") or ""), top_k=top_k, bot_id=bot_id)
    items = debug.get("items") or []
    first_match_rank = None
    for index, item in enumerate(items, start=1):
        if item_matches(case, item):
            first_match_rank = index
            break

    expected_top_type = str(case.get("expected_top_type") or "")
    actual_top_type = str(items[0].get("type") or "") if items else ""
    top_type_ok = not expected_top_type or expected_top_type == actual_top_type
    should_have_source = bool(case.get("should_have_source", True))
    source_ok = (first_match_rank is not None) if should_have_source and case_needles(case) else True
    top_field_checks = item_field_matches(case, items[0]) if items else {}

    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "ok": source_ok and top_type_ok,
        "requires_source": should_have_source and bool(case_needles(case)),
        "first_match_rank": first_match_rank,
        "hit_at_1": first_match_rank == 1,
        "hit_at_3": first_match_rank is not None and first_match_rank <= 3,
        "hit_at_5": first_match_rank is not None and first_match_rank <= 5,
        "mrr": 0.0 if first_match_rank is None else 1.0 / first_match_rank,
        "expected_top_type": expected_top_type,
        "actual_top_type": actual_top_type,
        "top_type_ok": top_type_ok,
        "top_document_ok": top_field_checks.get("document"),
        "top_page_ok": top_field_checks.get("page"),
        "top_source_type_ok": top_field_checks.get("source_type"),
        "top_article_ok": top_field_checks.get("article"),
        "top_clause_ok": top_field_checks.get("clause"),
        "top_table_ok": top_field_checks.get("table"),
        "retrieval_relevance": debug.get("retrieval_relevance"),
        "response_time_ms": int((time.perf_counter() - started) * 1000),
        "top_source_line": str(items[0].get("source_line") or "") if items else "",
    }


def summarize(items: list[dict]) -> dict:
    total = len(items)
    retrieval_items = [item for item in items if item["requires_source"]]
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "retrieval_cases": 0,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "mrr": 0,
            "response_time_ms": 0,
        }

    def ratio_for(field: str) -> float | None:
        applicable = [item for item in retrieval_items if item.get(field) is not None]
        if not applicable:
            return None
        return round(sum(1 for item in applicable if item.get(field) is True) / len(applicable), 4)

    passed = sum(1 for item in items if item["ok"])
    retrieval_total = len(retrieval_items)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "retrieval_cases": retrieval_total,
        "hit_at_1": round(sum(1 for item in retrieval_items if item["hit_at_1"]) / retrieval_total, 4) if retrieval_total else 0,
        "hit_at_3": round(sum(1 for item in retrieval_items if item["hit_at_3"]) / retrieval_total, 4) if retrieval_total else 0,
        "hit_at_5": round(sum(1 for item in retrieval_items if item["hit_at_5"]) / retrieval_total, 4) if retrieval_total else 0,
        "mrr": round(sum(float(item["mrr"]) for item in retrieval_items) / retrieval_total, 4) if retrieval_total else 0,
        "top_document_accuracy": ratio_for("top_document_ok"),
        "source_page_accuracy": ratio_for("top_page_ok"),
        "source_type_accuracy": ratio_for("top_source_type_ok"),
        "article_accuracy": ratio_for("top_article_ok"),
        "clause_accuracy": ratio_for("top_clause_ok"),
        "table_accuracy": ratio_for("top_table_ok"),
        "response_time_ms": sum(int(item["response_time_ms"]) for item in items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality without calling the LLM.")
    parser.add_argument("--file", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--bot-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, help="Write full JSON result to this path.")
    args = parser.parse_args()

    cases = load_cases(args.file)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    items = [evaluate_case(case, args.bot_id, args.top_k) for case in cases]
    summary = summarize(items)
    payload = {"summary": summary, "items": items}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in items:
            status = "PASS" if item["ok"] else "FAIL"
            rank = item["first_match_rank"] if item["first_match_rank"] is not None else "-"
            print(f"{status} {item['id']} rank={rank} top_type={item['actual_top_type']}")
            if not item["ok"]:
                print(f"  top_source={item['top_source_line']}")
        print(f"\nSUMMARY {json.dumps(summary, ensure_ascii=False)}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

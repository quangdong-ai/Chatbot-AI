import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = ("id", "question")
LIST_FIELDS = ("expected_contains", "forbidden_contains", "expected_source_contains")
STRING_FIELDS = (
    "id",
    "question",
    "answer_reference",
    "source_document",
    "source_type",
    "expected_article",
    "expected_clause",
    "expected_table",
    "question_type",
    "notes",
)
BOOL_FIELDS = ("must_answer", "must_refuse", "should_have_source")
ALLOWED_QUESTION_TYPES = {
    "direct_lookup",
    "definition",
    "table",
    "comparison",
    "scenario",
    "multi_condition",
    "follow_up",
    "out_of_scope",
    "ocr",
    "prompt_safety",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Golden set must be a JSON array.")
    return data


def validate_case(case: object, index: int) -> list[str]:
    prefix = f"case[{index}]"
    if not isinstance(case, dict):
        return [f"{prefix}: must be an object"]

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if not str(case.get(field) or "").strip():
            errors.append(f"{prefix}: missing required field `{field}`")

    for field in LIST_FIELDS:
        value = case.get(field, [])
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"{prefix}: `{field}` must be a list of strings")

    for field in STRING_FIELDS:
        value = case.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"{prefix}: `{field}` must be a string")

    source_page = case.get("source_page")
    if source_page is not None and not isinstance(source_page, (str, int)):
        errors.append(f"{prefix}: `source_page` must be a string or integer")

    for field in BOOL_FIELDS:
        value = case.get(field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"{prefix}: `{field}` must be a boolean")

    question_type = case.get("question_type")
    if question_type and question_type not in ALLOWED_QUESTION_TYPES:
        errors.append(
            f"{prefix}: `question_type` must be one of {sorted(ALLOWED_QUESTION_TYPES)}"
        )

    must_refuse = bool(case.get("must_refuse", False))
    should_have_source = bool(case.get("should_have_source", not must_refuse))
    if must_refuse and should_have_source:
        errors.append(f"{prefix}: refusal cases should set `should_have_source` to false")

    has_source_target = any(
        case.get(field) not in (None, "", [])
        for field in (
            "source_document",
            "source_page",
            "source_type",
            "expected_article",
            "expected_clause",
            "expected_table",
            "expected_source_contains",
        )
    )
    if should_have_source and not has_source_target:
        errors.append(f"{prefix}: source-required case has no source target")

    return errors


def validate_cases(cases: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = [str(case.get("id") or "") for case in cases if isinstance(case, dict)]
    duplicated_ids = sorted(case_id for case_id, count in Counter(ids).items() if case_id and count > 1)
    for case_id in duplicated_ids:
        errors.append(f"duplicate id `{case_id}`")
    for index, case in enumerate(cases):
        errors.extend(validate_case(case, index))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BotAI golden-set JSON files.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation result.")
    args = parser.parse_args()

    payload = []
    failed = False
    for path in args.files:
        try:
            cases = load_cases(path)
            errors = validate_cases(cases)
        except Exception as exc:
            errors = [str(exc)]
            cases = []
        failed = failed or bool(errors)
        payload.append({"file": str(path), "cases": len(cases), "errors": errors})

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in payload:
            status = "PASS" if not item["errors"] else "FAIL"
            print(f"{status} {item['file']} cases={item['cases']}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

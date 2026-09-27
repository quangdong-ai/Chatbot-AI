import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def status(value: object) -> str:
    return "PASS" if value else "FAIL"


def render_report(title: str, retrieval_path: Path | None, generation_path: Path | None) -> str:
    lines = [f"# {title}", ""]

    if retrieval_path:
        data = load_json(retrieval_path)
        summary = data.get("summary", {})
        lines.extend(
            [
                "## Retrieval",
                "",
                f"- Total cases: {summary.get('total', 0)}",
                f"- Passed: {summary.get('passed', 0)}",
                f"- Failed: {summary.get('failed', 0)}",
                f"- Retrieval cases: {summary.get('retrieval_cases', 0)}",
                f"- Hit@1: {summary.get('hit_at_1', 0)}",
                f"- Hit@3: {summary.get('hit_at_3', 0)}",
                f"- Hit@5: {summary.get('hit_at_5', 0)}",
                f"- MRR: {summary.get('mrr', 0)}",
                "",
                "### Retrieval Failures",
                "",
            ]
        )
        failures = [item for item in data.get("items", []) if not item.get("ok")]
        if not failures:
            lines.append("- None")
        for item in failures:
            lines.append(
                f"- {item.get('id', '')}: top source `{str(item.get('top_source_line', '')).splitlines()[0]}`"
            )
        lines.append("")

    if generation_path:
        data = load_json(generation_path)
        summary = data.get("summary", {})
        lines.extend(
            [
                "## Generation",
                "",
                f"- Total cases: {summary.get('total', 0)}",
                f"- Passed: {summary.get('passed', 0)}",
                f"- Failed: {summary.get('failed', 0)}",
                f"- Answer passed: {summary.get('answer_passed', 0)}",
                f"- Source passed: {summary.get('source_passed', 0)}",
                f"- Forbidden passed: {summary.get('forbidden_passed', 0)}",
                f"- Top type passed: {summary.get('top_type_passed', 0)}",
                "",
                "### Generation Failures",
                "",
            ]
        )
        failures = [item for item in data.get("items", []) if not item.get("ok")]
        if not failures:
            lines.append("- None")
        for item in failures:
            missing_answer = ", ".join(item.get("missing_answer") or [])
            missing_source = ", ".join(item.get("missing_source") or [])
            lines.append(
                f"- {item.get('id', '')}: answer={status(item.get('answer_ok'))}, "
                f"source={status(item.get('source_ok'))}, "
                f"missing_answer=`{missing_answer}`, missing_source=`{missing_source}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a markdown report from evaluation JSON files.")
    parser.add_argument("--title", default="BotAI Evaluation Report")
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = render_report(args.title, args.retrieval, args.generation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

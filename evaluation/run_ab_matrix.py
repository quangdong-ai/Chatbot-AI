import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "fixtures" / "golden_tailieu_v2_100.json"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "ab_runs"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "config"


def default_configs() -> list[dict]:
    return [
        {
            "name": "current_bge_m3_qwen25",
            "env": {
                "BOTAI_EMBEDDING_MODEL_PATH": "./models--BAAI--bge-m3",
                "BOTAI_CHROMA_DIR": "chroma_db",
                "BOTAI_BM25_PATH": "bm25_retriever.pkl",
                "BOTAI_LLM_MODEL": "qwen2.5:latest",
                "BOTAI_RERANKER_ENABLED": "0",
            },
        },
        {
            "name": "bge_large_en_qwen25_isolated",
            "requires_rebuild": True,
            "env": {
                "BOTAI_EMBEDDING_MODEL_PATH": "./models--BAAI--bge-large-en-v1.5",
                "BOTAI_CHROMA_DIR": "evaluation/ab_indexes/bge_large_en/chroma_db",
                "BOTAI_BM25_PATH": "evaluation/ab_indexes/bge_large_en/bm25_retriever.pkl",
                "BOTAI_LLM_MODEL": "qwen2.5:latest",
                "BOTAI_RERANKER_ENABLED": "0",
                "BOTAI_OCR_ENGINE": "tesseract",
            },
        },
    ]


def load_configs(path: Path | None) -> list[dict]:
    if not path:
        return default_configs()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("A/B config file must contain a list.")
    return data


def run_command(command: list[str], env: dict[str, str], cwd: Path) -> dict:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return {
        "returncode": process.returncode,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
    }


def maybe_rebuild_index(config: dict, env: dict[str, str], args: argparse.Namespace) -> dict | None:
    if not args.rebuild_index:
        return None
    if not config.get("requires_rebuild"):
        return None
    chroma_dir = ROOT / env.get("BOTAI_CHROMA_DIR", "")
    bm25_path = ROOT / env.get("BOTAI_BM25_PATH", "")
    if chroma_dir.exists() and bm25_path.exists() and not args.force_rebuild:
        return {"skipped": True, "reason": "index_exists"}
    return run_command([sys.executable, "ingest.py"], env, ROOT)


def extract_summary(path: Path, kind: str) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind == "retrieval":
        return data.get("summary", {})
    summary = data.get("summary", data)
    return {
        "total": summary.get("total"),
        "passed": summary.get("passed"),
        "failed": summary.get("failed"),
        "answer_correctness": summary.get("answer_correctness"),
        "citation_correctness": summary.get("citation_correctness"),
        "source_passed": summary.get("source_passed"),
        "hallucination_proxy": summary.get("hallucination_proxy"),
        "answer_reference_recall": summary.get("answer_reference_recall"),
        "response_time_ms": summary.get("response_time_ms"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval/generation A/B matrix for BotAI.")
    parser.add_argument("--config", type=Path, help="JSON list of configs with name/env.")
    parser.add_argument("--kind", choices=["retrieval", "generation"], default="retrieval")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = load_configs(args.config)
    matrix = []

    for config in configs:
        name = slug(config.get("name", "config"))
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
        env.setdefault("PYTHONIOENCODING", "utf-8")

        if config.get("requires_rebuild") and not args.rebuild_index:
            matrix.append({
                "name": name,
                "env": config.get("env", {}),
                "skipped": True,
                "reason": "requires --rebuild-index for a fair embedding A/B run",
            })
            print(f"{name}: skipped, requires --rebuild-index")
            continue

        rebuild = maybe_rebuild_index(config, env, args)
        if rebuild and rebuild.get("returncode", 0) != 0:
            matrix.append({
                "name": name,
                "env": config.get("env", {}),
                "rebuild": rebuild,
                "skipped": True,
                "reason": "index rebuild failed",
            })
            print(f"{name}: skipped, index rebuild failed")
            continue
        result_path = args.output_dir / f"{name}_{args.kind}.json"
        if args.kind == "retrieval":
            command = [
                sys.executable,
                "evaluation/run_retrieval_eval.py",
                "--file",
                str(args.case_file),
                "--output",
                str(result_path),
            ]
        else:
            command = [
                sys.executable,
                "run_benchmark.py",
                "--file",
                str(args.case_file),
                "--output",
                str(result_path),
            ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])

        run = run_command(command, env, ROOT)
        run["result_exists"] = result_path.exists()
        matrix.append({
            "name": name,
            "env": config.get("env", {}),
            "rebuild": rebuild,
            "run": run,
            "result": str(result_path),
            "summary": extract_summary(result_path, args.kind),
        })
        print(f"{name}: returncode={run['returncode']} elapsed={run['elapsed_sec']}s")

    report = {
        "kind": args.kind,
        "case_file": str(args.case_file),
        "limit": args.limit,
        "matrix": matrix,
    }
    report_path = args.output_dir / f"matrix_{args.kind}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0 if all(item.get("skipped") or item["run"].get("result_exists") for item in matrix) else 1


if __name__ == "__main__":
    raise SystemExit(main())

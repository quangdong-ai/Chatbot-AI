from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]
WORK_CACHE = ROOT / ".parser_cache"


def _configure_workspace_cache() -> None:
    WORK_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("DOCLING_CACHE_DIR", str(WORK_CACHE / "docling"))
    os.environ.setdefault("MINERU_HOME", str(WORK_CACHE / "mineru"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(WORK_CACHE / "modelscope"))
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORK_CACHE / "modelscope" / "credentials"))
    os.environ.setdefault("MINERU_MODEL_SMALL_BACKEND", "onnx")
    os.environ.setdefault("MINERU_MODEL_SOURCE", "local")
    os.environ.setdefault("MINERU_DEVICE_MODE", "cpu")


def _make_page_pdf(source: Path, page_number: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    src = fitz.open(source)
    try:
        if page_number < 1 or page_number > src.page_count:
            raise ValueError(f"page_number must be 1..{src.page_count}, got {page_number}")
        dst = fitz.open()
        try:
            dst.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
            dst.save(output)
        finally:
            dst.close()
    finally:
        src.close()


def _preview(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def probe_docling(pdf_path: Path) -> dict[str, Any]:
    _configure_workspace_cache()
    started = time.perf_counter()
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter
        from docling.document_converter import PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            artifacts_path=str(WORK_CACHE / "docling" / "models"),
            do_ocr=True,
            do_table_structure=True,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        return {
            "ok": True,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "chars": len(markdown),
            "preview": _preview(markdown),
        }
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_mineru(pdf_path: Path, output_dir: Path, timeout_s: int) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).with_name("mineru-kit.exe")
    if not exe.exists():
        return {
            "ok": False,
            "elapsed_s": 0.0,
            "error": f"mineru-kit.exe not found next to {sys.executable}",
        }

    cmd = [
        str(exe),
        "parse",
        str(pdf_path),
        "--output",
        str(output_dir),
        "--pages",
        "1",
        "--format",
        "markdown",
        "--tier",
        "flash",
        "--disable-image-analysis",
    ]
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    env.setdefault("MINERU_HOME", str(WORK_CACHE / "mineru"))
    env.setdefault("MODELSCOPE_CACHE", str(WORK_CACHE / "modelscope"))
    env.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORK_CACHE / "modelscope" / "credentials"))
    env.setdefault("MINERU_MODEL_SMALL_BACKEND", "onnx")
    env.setdefault("MINERU_MODEL_SOURCE", "local")
    env.setdefault("MINERU_DEVICE_MODE", "cpu")
    env.setdefault("HOME", str(WORK_CACHE / "home"))
    env.setdefault("USERPROFILE", str(WORK_CACHE / "home"))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "error": f"TimeoutExpired after {timeout_s}s",
        }

    markdown_files = list(output_dir.rglob("*.md"))
    markdown = ""
    if markdown_files:
        markdown = max(markdown_files, key=lambda p: p.stat().st_size).read_text(encoding="utf-8", errors="replace")
    return {
        "ok": proc.returncode == 0 and bool(markdown),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "returncode": proc.returncode,
        "chars": len(markdown),
        "preview": _preview(markdown),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _configure_workspace_cache()
    parser = argparse.ArgumentParser(description="Probe Docling and MinerU layout parsers on a small PDF page.")
    parser.add_argument("--pdf", default=str(ROOT / "scan.pdf"))
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "layout_parser_probe.json"))
    args = parser.parse_args()

    source = Path(args.pdf).resolve()
    out_path = Path(args.out).resolve()
    work_dir = out_path.parent / "layout_parser_probe"
    sample_pdf = work_dir / f"{source.stem}_page_{args.page}.pdf"
    _make_page_pdf(source, args.page, sample_pdf)

    report = {
        "source_pdf": str(source),
        "page": args.page,
        "sample_pdf": str(sample_pdf),
        "docling": probe_docling(sample_pdf),
        "mineru": probe_mineru(sample_pdf, work_dir / "mineru_output", args.timeout_s),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["docling"]["ok"] or report["mineru"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

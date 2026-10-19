from __future__ import annotations

from pathlib import Path

from .config import env
from .extractor import ExtractionResult, extract
from .llm import LLMClient
from .ocr import OcrEngine


def max_file_bytes() -> int:
    try:
        return int(float(env("MAX_FILE_MB", "20")) * 1024 * 1024)
    except ValueError:
        return 20 * 1024 * 1024


def _fail(msg: str) -> ExtractionResult:
    return ExtractionResult(None, True, [msg], error=msg)


def process_file(path: Path, ocr: OcrEngine, llm: LLMClient) -> ExtractionResult:
    try:
        size = path.stat().st_size
    except OSError as e:
        return _fail(f"cannot read file: {e}")
    if size == 0:
        return _fail(f"{path.name}: file is empty")
    if size > max_file_bytes():
        return _fail(f"{path.name}: file is {size / 1024 / 1024:.1f} MB, over the MAX_FILE_MB limit")
    try:
        text = ocr.read(path)
    except Exception as e:  # OCR failures become a review item, not a crash
        return ExtractionResult(None, True, [f"OCR failed: {e}"], error=str(e))
    return extract(text, llm)

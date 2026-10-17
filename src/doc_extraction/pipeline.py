from __future__ import annotations

from pathlib import Path

from .extractor import ExtractionResult, extract
from .llm import LLMClient
from .ocr import OcrEngine


def process_file(path: Path, ocr: OcrEngine, llm: LLMClient) -> ExtractionResult:
    try:
        text = ocr.read(path)
    except Exception as e:  # OCR failures become a review item, not a crash
        return ExtractionResult(None, True, [f"OCR failed: {e}"], error=str(e))
    return extract(text, llm)

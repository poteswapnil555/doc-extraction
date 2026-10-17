"""Switchable OCR. OCR_ENGINE=tesseract (default) or azure_di."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .config import env

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OcrEngine(Protocol):
    def read(self, path: Path) -> str: ...


class TesseractOcr:
    """Text PDFs use their embedded text layer; images go through Tesseract."""

    def read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages).strip()
            if not text:
                raise ValueError(f"{path.name}: PDF has no text layer (scanned PDFs are not supported by this engine)")
            return text
        if suffix in IMAGE_SUFFIXES:
            import pytesseract
            from PIL import Image

            cmd = env("TESSERACT_CMD") or (_WIN_TESSERACT if os.path.exists(_WIN_TESSERACT) else None)
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
            return pytesseract.image_to_string(Image.open(path)).strip()
        raise ValueError(f"Unsupported file type: {suffix}")


class AzureDocumentIntelligenceOcr:
    """Optional. Needs: pip install -r requirements-azure.txt, plus
    AZURE_DI_ENDPOINT and AZURE_DI_KEY environment variables."""

    def __init__(self) -> None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        endpoint, key = env("AZURE_DI_ENDPOINT"), env("AZURE_DI_KEY")
        if not (endpoint and key):
            raise RuntimeError("Set AZURE_DI_ENDPOINT and AZURE_DI_KEY to use OCR_ENGINE=azure_di")
        self._client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    def read(self, path: Path) -> str:
        with open(path, "rb") as f:
            poller = self._client.begin_analyze_document("prebuilt-read", body=f)
        return (poller.result().content or "").strip()


def get_ocr_engine(name: str | None = None) -> OcrEngine:
    name = (name or env("OCR_ENGINE", "tesseract")).lower()
    if name == "tesseract":
        return TesseractOcr()
    if name == "azure_di":
        return AzureDocumentIntelligenceOcr()
    raise ValueError(f"Unknown OCR_ENGINE '{name}' (use 'tesseract' or 'azure_di')")

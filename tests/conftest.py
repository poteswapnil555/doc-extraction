import json
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


class FakeLLM:
    """Returns queued replies in order (last one repeats); records prompts."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def truth():
    return lambda n: json.loads((SAMPLE / f"invoice_{n:02d}.json").read_text())


@pytest.fixture
def pdf_text():
    from doc_extraction.ocr import TesseractOcr

    return TesseractOcr().read(SAMPLE / "invoice_02.pdf")

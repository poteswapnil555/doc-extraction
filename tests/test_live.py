"""Optional live test: skipped unless an LLM is configured through env variables."""
import pytest

from conftest import SAMPLE
from doc_extraction.config import llm_configured

pytestmark = pytest.mark.skipif(not llm_configured(), reason="no LLM env variables set")


def test_live_extraction_on_pdf():
    from doc_extraction.llm import OpenAICompatibleClient
    from doc_extraction.ocr import TesseractOcr
    from doc_extraction.pipeline import process_file

    r = process_file(SAMPLE / "invoice_02.pdf", TesseractOcr(), OpenAICompatibleClient())
    assert r.invoice is not None

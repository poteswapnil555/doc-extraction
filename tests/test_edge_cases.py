"""Edge cases: every failure must become a review item or a clear message, never a crash."""
import json

import pytest
from PIL import Image

from conftest import SAMPLE, FakeLLM
from doc_extraction import __main__ as cli
from doc_extraction import llm as llm_mod
from doc_extraction.extractor import MAX_TEXT_CHARS, extract
from doc_extraction.ocr import TesseractOcr, get_ocr_engine
from doc_extraction.pipeline import process_file

OCR = TesseractOcr()


class FixedOcr:
    def __init__(self, text):
        self.text = text

    def read(self, path):
        return self.text


class RaisingLLM:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        raise self.exc


def _assert_review_item(r, fragment):
    assert r.invoice is None and r.needs_review
    assert any(fragment in x for x in r.reasons), r.reasons


# ---- files -----------------------------------------------------------------
@pytest.mark.parametrize("name", ["empty.pdf", "empty.png"])
def test_empty_file(tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"")
    _assert_review_item(process_file(f, OCR, FakeLLM("{}")), "empty")


def test_missing_file(tmp_path):
    _assert_review_item(process_file(tmp_path / "nope.pdf", OCR, FakeLLM("{}")), "cannot read file")


def test_corrupt_pdf(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    _assert_review_item(process_file(f, OCR, FakeLLM("{}")), "OCR failed")


def test_corrupt_image(tmp_path):
    f = tmp_path / "bad.png"
    f.write_bytes(b"not an image at all")
    _assert_review_item(process_file(f, OCR, FakeLLM("{}")), "OCR failed")


def test_unsupported_type(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_bytes(b"PK")
    _assert_review_item(process_file(f, OCR, FakeLLM("{}")), "Unsupported file type")


def test_very_large_file_is_rejected_without_ocr_or_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "0.001")  # about 1 KB limit
    f = tmp_path / "big.png"
    f.write_bytes(b"x" * 5000)
    llm = FakeLLM("{}")
    _assert_review_item(process_file(f, FixedOcr("never read"), llm), "MAX_FILE_MB")
    assert llm.prompts == []


def test_bad_max_file_mb_falls_back_to_default(monkeypatch, truth):
    monkeypatch.setenv("MAX_FILE_MB", "abc")
    r = process_file(SAMPLE / "invoice_02.pdf", OCR, FakeLLM(json.dumps(truth(2))))
    assert r.invoice is not None


def test_very_long_ocr_text_is_not_sent_to_llm():
    llm = FakeLLM("{}")
    r = extract("A" * (MAX_TEXT_CHARS + 1), llm)
    _assert_review_item(r, "too long")
    assert llm.prompts == []


# ---- OCR returns nothing -----------------------------------------------------
def test_ocr_returns_no_text(tmp_path):
    f = tmp_path / "blank.png"
    Image.new("RGB", (300, 200), "white").save(f)
    llm = FakeLLM("{}")
    _assert_review_item(process_file(f, OCR, llm), "empty OCR text")
    assert llm.prompts == []


def test_fake_ocr_returning_empty_string(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    _assert_review_item(process_file(f, FixedOcr(""), FakeLLM("{}")), "empty OCR text")


# ---- LLM failures -------------------------------------------------------------
def test_llm_timeout_becomes_review_item_not_crash(pdf_text):
    llm = RaisingLLM(TimeoutError("request timed out"))
    r = extract(pdf_text, llm)
    _assert_review_item(r, "LLM call failed: TimeoutError")
    assert llm.calls == 1


def test_openai_timeout_exception_is_handled(pdf_text):
    import httpx
    import openai

    exc = openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    _assert_review_item(extract(pdf_text, RaisingLLM(exc)), "APITimeoutError")


def test_llm_returns_empty_text_then_gives_up(pdf_text):
    llm = FakeLLM("")
    r = extract(pdf_text, llm, max_retries=2)
    assert r.invoice is None and r.needs_review and r.attempts == 3 and len(llm.prompts) == 3


def test_llm_returns_empty_text_then_recovers(pdf_text, truth):
    r = extract(pdf_text, FakeLLM("", json.dumps(truth(2))))
    assert r.invoice and r.attempts == 2


def test_llm_returns_json_null_list_or_number(pdf_text):
    for reply in ("null", "[]", "42"):
        r = extract(pdf_text, FakeLLM(reply), max_retries=0)
        assert r.invoice is None and r.needs_review


def test_client_with_no_choices_is_handled(pdf_text):
    class NoChoices:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            class R:
                choices = []

            return R()

    c = object.__new__(llm_mod.OpenAICompatibleClient)
    c._client, c._model = NoChoices(), "m"
    _assert_review_item(extract(pdf_text, c), "LLM call failed: IndexError")


def test_evaluation_survives_llm_timeouts():
    from doc_extraction.evaluate import evaluate

    report = evaluate(SAMPLE, OCR, RaisingLLM(TimeoutError("t")), "mock")
    assert report.docs == 20 and report.no_output == 20


# ---- missing environment variables -----------------------------------------------
ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OCR_ENGINE",
            "AZURE_DI_ENDPOINT", "AZURE_DI_KEY", "LLM_TIMEOUT_SECONDS"]


@pytest.fixture
def no_env(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_cli_without_env_prints_clear_message(no_env, capsys):
    assert cli.main([str(SAMPLE / "invoice_02.pdf")]) == 2
    assert "No LLM configured" in capsys.readouterr().err


def test_cli_without_args_prints_usage(no_env):
    assert cli.main([]) == 2


def test_azure_without_deployment_has_clear_message(no_env):
    no_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
    no_env.setenv("AZURE_OPENAI_API_KEY", "fake")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_DEPLOYMENT is not set"):
        llm_mod.OpenAICompatibleClient()


def test_azure_di_without_variables_has_clear_message(no_env):
    pytest.importorskip("azure.ai.documentintelligence")
    with pytest.raises(RuntimeError, match="AZURE_DI_ENDPOINT"):
        get_ocr_engine("azure_di")


def test_cli_with_azure_di_without_sdk_or_vars_does_not_crash(no_env, capsys):
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("OCR_ENGINE", "azure_di")
    assert cli.main([str(SAMPLE / "invoice_02.pdf")]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_bad_timeout_value_has_clear_message(no_env):
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("LLM_TIMEOUT_SECONDS", "soon")
    with pytest.raises(RuntimeError, match="LLM_TIMEOUT_SECONDS"):
        llm_mod.OpenAICompatibleClient()


def test_openai_client_gets_timeout(no_env):
    import openai

    seen = {}
    no_env.setenv("OPENAI_API_KEY", "fake")
    no_env.setenv("LLM_TIMEOUT_SECONDS", "7")
    no_env.setattr(openai, "OpenAI", lambda **kw: seen.update(kw) or object())
    llm_mod.OpenAICompatibleClient()
    assert seen["timeout"] == 7.0

import json

from conftest import SAMPLE, FakeLLM
from doc_extraction.evaluate import MockLLM, compare, evaluate, main
from doc_extraction.ocr import TesseractOcr
from doc_extraction.schema import Invoice


def _truth(n):
    return Invoice.model_validate_json((SAMPLE / f"invoice_{n:02d}.json").read_text())


def test_compare_identical_is_all_correct():
    assert all(compare(_truth(2), _truth(2)).values())


def test_compare_detects_wrong_field():
    wrong = _truth(2).model_copy(update={"vendor_name": "Someone Else"})
    r = compare(wrong, _truth(2))
    assert r["vendor_name"] is False and r["total"] is True


def test_compare_missing_prediction_is_all_wrong():
    assert not any(compare(None, _truth(2)).values())


def test_mock_run_covers_all_documents():
    report = evaluate(SAMPLE, TesseractOcr(), MockLLM(), "mock")
    assert report.docs == 20 and report.docs_exact == 20


def test_bad_llm_scores_zero_and_is_flagged():
    report = evaluate(SAMPLE, TesseractOcr(), FakeLLM("not json"), "mock")
    assert report.docs_exact == 0 and report.no_output == 20
    assert report.wrong_docs_flagged == report.wrong_docs == 20


def test_main_mock_prints_mode(capsys):
    assert main(["--mock", "--dir", str(SAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "MODE: mock" in out and "NOT accuracy" in out


def test_main_live_without_env_fails_cleanly(monkeypatch, capsys):
    for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert main(["--live", "--dir", str(SAMPLE)]) == 2
    assert "Cannot run --live" in capsys.readouterr().err

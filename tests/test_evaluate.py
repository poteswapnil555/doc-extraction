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


def test_out_file_has_per_document_details(tmp_path):
    out = tmp_path / "eval.json"
    assert main(["--mock", "--dir", str(SAMPLE), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "mock" and len(data["documents_detail"]) == 20
    first = data["documents_detail"][0]
    assert set(first) == {"file", "all_correct", "wrong_fields", "needs_review", "review_reasons", "attempts", "error"}


def test_details_show_expected_and_predicted_for_wrong_fields():
    truth = _truth(2)
    wrong = truth.model_copy(update={"vendor_name": "Someone Else"})
    report = evaluate(SAMPLE, TesseractOcr(), FakeLLM(wrong.model_dump_json()), "mock")
    d = next(x for x in report.details if x["file"] == "invoice_02.pdf")
    assert not d["all_correct"]
    vendor = next(w for w in d["wrong_fields"] if w["field"] == "vendor_name")
    assert vendor["expected"] == truth.vendor_name and vendor["predicted"] == "Someone Else"
    assert "not found on document: vendor_name" in d["review_reasons"]
    json.dumps(report.details)  # must be serialisable


def test_details_for_document_without_output():
    report = evaluate(SAMPLE, TesseractOcr(), FakeLLM("not json"), "mock")
    d = report.details[0]
    assert d["error"] and d["wrong_fields"][0]["predicted"] is None and len(d["wrong_fields"]) == 9


def test_main_with_empty_dir_fails_cleanly(tmp_path, capsys):
    assert main(["--mock", "--dir", str(tmp_path)]) == 2
    assert "No invoice_" in capsys.readouterr().err

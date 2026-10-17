import json

import pytest

from conftest import SAMPLE, FakeLLM
from doc_extraction.extractor import extract, parse_json
from doc_extraction.ocr import TesseractOcr, get_ocr_engine
from doc_extraction.pipeline import process_file


def test_good_output_is_accepted_and_grounded(pdf_text, truth):
    r = extract(pdf_text, FakeLLM(json.dumps(truth(2))))
    assert r.invoice and r.attempts == 1
    assert not r.needs_review, r.reasons


def test_hallucinated_value_is_flagged(pdf_text, truth):
    bad = truth(2)
    bad["vendor_name"] = "Totally Invented Corp"
    r = extract(pdf_text, FakeLLM(json.dumps(bad)))
    assert r.needs_review and "not found on document: vendor_name" in r.reasons


def test_wrong_total_is_flagged(pdf_text, truth):
    bad = truth(2)
    bad["total"] = "999999.00"
    r = extract(pdf_text, FakeLLM(json.dumps(bad)))
    assert r.needs_review and any("total" in x for x in r.reasons)


def test_non_json_then_valid_retries(pdf_text, truth):
    llm = FakeLLM("Sure! here you go", json.dumps(truth(2)))
    r = extract(pdf_text, llm)
    assert r.invoice and r.attempts == 2
    assert "previous answer was invalid" in llm.prompts[1]


def test_schema_invalid_then_valid_retries(pdf_text, truth):
    broken = truth(2)
    broken["line_items"] = []
    r = extract(pdf_text, FakeLLM(json.dumps(broken), json.dumps(truth(2))))
    assert r.invoice and r.attempts == 2


def test_gives_up_after_retries(pdf_text):
    r = extract(pdf_text, FakeLLM("not json"), max_retries=2)
    assert r.invoice is None and r.needs_review and r.attempts == 3


def test_empty_text_never_calls_llm():
    llm = FakeLLM("{}")
    r = extract("   ", llm)
    assert r.invoice is None and llm.prompts == []


def test_fenced_json_is_parsed():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_non_object_json_rejected():
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


def test_pdf_ocr_reads_text_layer():
    assert "INV-2024002" in TesseractOcr().read(SAMPLE / "invoice_02.pdf")


def test_image_ocr_reads_png(truth):
    text = TesseractOcr().read(SAMPLE / "invoice_01.png")
    assert truth(1)["invoice_number"] in text


def test_pipeline_end_to_end_on_image(truth):
    r = process_file(SAMPLE / "invoice_01.png", TesseractOcr(), FakeLLM(json.dumps(truth(1))))
    assert r.invoice and r.invoice.invoice_number == truth(1)["invoice_number"]


def test_unsupported_file_becomes_review_item(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    r = process_file(f, TesseractOcr(), FakeLLM("{}"))
    assert r.needs_review and r.invoice is None


def test_ocr_engine_switch():
    assert isinstance(get_ocr_engine("tesseract"), TesseractOcr)
    with pytest.raises(ValueError):
        get_ocr_engine("nope")

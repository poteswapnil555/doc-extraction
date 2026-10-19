import json

from conftest import SAMPLE


def _docs():
    return [json.loads(p.read_text()) for p in sorted(SAMPLE.glob("invoice_*.json"))]


def test_sample_has_missing_fields():
    docs = _docs()
    assert any(d["customer_name"] is None for d in docs)
    assert any(float(d["tax"]) == 0 for d in docs)


def test_sample_has_odd_date_formats_in_ocr_text():
    from doc_extraction.ocr import TesseractOcr

    text = TesseractOcr().read(SAMPLE / "invoice_02.pdf")  # n=2 -> "%d/%m/%Y"
    assert "/" in text.split("Date:")[1].split("\n")[0]


def test_sample_has_low_quality_scans_and_mixed_types():
    assert len(list(SAMPLE.glob("*.png"))) == 10 and len(list(SAMPLE.glob("*.pdf"))) == 10

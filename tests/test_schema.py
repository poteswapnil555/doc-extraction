import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from doc_extraction.schema import Invoice, arithmetic_issues
from doc_extraction.synthetic import generate, make_invoice
import random

GOOD = {
    "invoice_number": "INV-1", "invoice_date": "2024-03-01", "vendor_name": "V",
    "line_items": [{"description": "x", "quantity": 2, "unit_price": "5.00", "amount": "10.00"}],
    "subtotal": "10.00", "tax": "1.00", "total": "11.00",
}


def test_valid_invoice_parses():
    inv = Invoice.model_validate(GOOD)
    assert inv.currency == "USD" and arithmetic_issues(inv) == []


@pytest.mark.parametrize("patch", [{"invoice_date": "not-a-date"}, {"line_items": []}, {"total": "-1"}, {"invoice_number": ""}])
def test_invalid_invoice_rejected(patch):
    with pytest.raises(ValidationError):
        Invoice.model_validate({**GOOD, **patch})


def test_arithmetic_mismatch_flagged():
    inv = Invoice.model_validate({**GOOD, "total": "99.00"})
    assert "subtotal + tax != total" in arithmetic_issues(inv)


def test_generated_invoices_are_consistent():
    rng = random.Random(1)
    for n in range(30):
        assert arithmetic_issues(make_invoice(rng, n)) == []


def test_generate_writes_files_and_truth(tmp_path):
    generate(tmp_path, count=4)
    assert len(list(tmp_path.glob("*.json"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 2 and len(list(tmp_path.glob("*.png"))) == 2
    Invoice.model_validate(json.loads((tmp_path / "invoice_01.json").read_text()))

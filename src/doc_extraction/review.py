"""Grounding, confidence and the review flag. Plain code, no LLM involved."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Invoice, arithmetic_issues


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _money_forms(value) -> list[str]:
    q = f"{value:.2f}"
    return [q, q[:-3]] if q.endswith(".00") else [q]


DATE_FORMATS = ["%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%B %d, %Y"]  # slash dates are day/month/year


def _date_printed(d, text: str) -> bool:
    return any(_in_text(d.strftime(f), text) for f in DATE_FORMATS)


def _in_text(needle: str, haystack: str) -> bool:
    return _norm(needle) in _norm(haystack)


def field_confidence(inv: Invoice, text: str) -> dict[str, float]:
    """1.0 if the value is literally printed on the document, else 0.0."""

    def money(v) -> float:
        return float(any(_in_text(m, text) for m in _money_forms(v)))

    conf = {
        "invoice_number": float(_in_text(inv.invoice_number, text)),
        "invoice_date": float(_date_printed(inv.invoice_date, text)),
        "vendor_name": float(_in_text(inv.vendor_name, text)),
        "subtotal": money(inv.subtotal),
        "total": money(inv.total),
    }
    if inv.tax != 0:  # a missing tax line legitimately means 0
        conf["tax"] = money(inv.tax)
    if inv.customer_name:
        conf["customer_name"] = float(_in_text(inv.customer_name, text))
    for i, li in enumerate(inv.line_items):
        conf[f"line_items[{i}].description"] = float(_in_text(li.description, text))
        conf[f"line_items[{i}].amount"] = money(li.amount)
    return conf


@dataclass
class Review:
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def review(inv: Invoice, conf: dict[str, float]) -> Review:
    reasons = [f"not found on document: {k}" for k, v in conf.items() if v < 1.0]
    reasons += arithmetic_issues(inv)
    return Review(needs_review=bool(reasons), reasons=reasons)

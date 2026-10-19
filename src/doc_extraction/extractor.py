"""OCR text -> LLM -> strict JSON -> validation -> grounding -> review flag."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from .llm import LLMClient
from .review import field_confidence, review
from .schema import Invoice

MAX_TEXT_CHARS = 20000  # longer OCR text is not sent to the LLM (cost, and the model could miss the totals)

SYSTEM_PROMPT = """You extract invoice data from OCR text.
Rules:
- Use ONLY values that are printed in the document text. Never guess or infer.
- If an optional value is not printed, use null. Never invent values.
- Dates must be ISO format YYYY-MM-DD. Amounts are plain numbers without currency symbols.
- Return ONLY one JSON object, no explanation, with exactly these keys:
  invoice_number, invoice_date, vendor_name, customer_name, currency,
  line_items (list of {description, quantity, unit_price, amount}), subtotal, tax, total."""


@dataclass
class ExtractionResult:
    invoice: Invoice | None
    needs_review: bool
    reasons: list[str] = field(default_factory=list)
    field_confidence: dict[str, float] = field(default_factory=dict)
    attempts: int = 0
    error: str | None = None


def parse_json(raw: str) -> dict:
    """Accept bare JSON or JSON wrapped in a markdown code fence."""
    raw = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    data = json.loads(fenced.group(1) if fenced else raw)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def extract(text: str, llm: LLMClient, max_retries: int = 2) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult(None, True, ["empty OCR text"], error="empty OCR text")
    if len(text) > MAX_TEXT_CHARS:
        msg = f"OCR text too long ({len(text)} chars, limit {MAX_TEXT_CHARS}); not sent to the LLM"
        return ExtractionResult(None, True, [msg], error=msg)
    user = f'Document text:\n"""\n{text}\n"""'
    last_error = ""
    for attempt in range(1, max_retries + 2):
        prompt = user if not last_error else f"{user}\n\nYour previous answer was invalid: {last_error}\nReturn corrected JSON only."
        try:
            raw = llm.complete(SYSTEM_PROMPT, prompt)
        except Exception as e:  # timeout, connection error, exhausted rate limit, empty choices...
            msg = f"LLM call failed: {type(e).__name__}: {str(e)[:300]}"
            return ExtractionResult(None, True, [msg], attempts=attempt, error=msg)
        try:
            invoice = Invoice.model_validate(parse_json(raw))
        except (ValueError, ValidationError) as e:  # JSONDecodeError and ValidationError are ValueErrors
            last_error = str(e)[:500]
            continue
        conf = field_confidence(invoice, text)
        rv = review(invoice, conf)
        return ExtractionResult(invoice, rv.needs_review, rv.reasons, conf, attempts=attempt)
    return ExtractionResult(None, True, ["LLM output invalid after retries"], attempts=max_retries + 1, error=last_error)

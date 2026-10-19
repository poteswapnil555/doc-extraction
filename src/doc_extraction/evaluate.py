"""Field-level accuracy on the synthetic sample set.

  python -m doc_extraction.evaluate --mock   # plumbing check only: the "LLM" returns the ground truth
  python -m doc_extraction.evaluate --live   # real LLM from environment variables -> real numbers
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from .ocr import OcrEngine, get_ocr_engine
from .pipeline import process_file
from .schema import Invoice

SCALAR_FIELDS = ["invoice_number", "invoice_date", "vendor_name", "customer_name", "currency", "subtotal", "tax", "total"]
FIELDS = SCALAR_FIELDS + ["line_items"]


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().lower()
    if isinstance(v, Decimal):
        return v.quantize(Decimal("0.01"))
    return v


def _line_key(li) -> tuple:
    return (_norm(li.description), _norm(li.quantity), _norm(li.unit_price), _norm(li.amount))


def compare(pred: Invoice | None, truth: Invoice) -> dict[str, bool]:
    """Per-field correctness. A missing prediction makes every field wrong."""
    if pred is None:
        return {f: False for f in FIELDS}
    result = {f: _norm(getattr(pred, f)) == _norm(getattr(truth, f)) for f in SCALAR_FIELDS}
    result["line_items"] = [_line_key(i) for i in pred.line_items] == [_line_key(i) for i in truth.line_items]
    return result


@dataclass
class Report:
    mode: str
    docs: int = 0
    field_correct: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FIELDS})
    docs_exact: int = 0
    wrong_docs: int = 0
    wrong_docs_flagged: int = 0
    correct_docs: int = 0
    correct_docs_flagged: int = 0
    no_output: int = 0

    def add(self, fields: dict[str, bool], needs_review: bool, produced: bool) -> None:
        self.docs += 1
        self.no_output += not produced
        for f, ok in fields.items():
            self.field_correct[f] += ok
        if all(fields.values()):
            self.docs_exact += 1
            self.correct_docs += 1
            self.correct_docs_flagged += needs_review
        else:
            self.wrong_docs += 1
            self.wrong_docs_flagged += needs_review

    def summary(self) -> dict:
        n = max(self.docs, 1)
        total_fields = self.docs * len(FIELDS)
        return {
            "mode": self.mode,
            "documents": self.docs,
            "field_accuracy": round(sum(self.field_correct.values()) / max(total_fields, 1), 4),
            "per_field_accuracy": {f: round(c / n, 4) for f, c in self.field_correct.items()},
            "documents_fully_correct": self.docs_exact,
            "documents_with_no_valid_output": self.no_output,
            "wrong_docs_flagged_for_review": f"{self.wrong_docs_flagged}/{self.wrong_docs}",
            "correct_docs_flagged_unnecessarily": f"{self.correct_docs_flagged}/{self.correct_docs}",
        }


class MockLLM:
    """Returns whatever reply is set. Used only for --mock plumbing checks."""

    reply = "{}"

    def complete(self, system: str, user: str) -> str:
        return self.reply


def evaluate(sample_dir: Path, ocr: OcrEngine, llm, mode: str, delay: float = 0.0, sleep=time.sleep, progress=None) -> Report:
    """delay: seconds to pause between documents (for rate-limited free tiers).
    progress: optional callable(done, total), called after each document."""
    report = Report(mode=mode)
    jobs = []
    for truth_path in sorted(sample_dir.glob("invoice_*.json")):
        doc = next((p for p in truth_path.parent.glob(truth_path.stem + ".*") if p.suffix in {".pdf", ".png"}), None)
        if doc is not None:
            jobs.append((truth_path, doc))
    for i, (truth_path, doc) in enumerate(jobs, start=1):
        truth_text = truth_path.read_text(encoding="utf-8")
        if isinstance(llm, MockLLM):
            llm.reply = truth_text
        result = process_file(doc, ocr, llm)
        report.add(compare(result.invoice, Invoice.model_validate_json(truth_text)), result.needs_review, result.invoice is not None)
        if progress:
            progress(i, len(jobs))
        if delay > 0 and i < len(jobs):
            sleep(delay)
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="no key needed; NOT an accuracy measurement")
    group.add_argument("--live", action="store_true", help="use the LLM configured by environment variables")
    ap.add_argument("--dir", type=Path, default=Path("data/sample"))
    ap.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                    help="pause between documents, e.g. 13 for a 5 requests/minute limit")
    ap.add_argument("--out", type=Path, help="optional path to save the summary as JSON")
    args = ap.parse_args(argv)

    if args.mock:
        llm, mode = MockLLM(), "mock"
        print("MODE: mock -- plumbing check only. These numbers are NOT accuracy.")
    else:
        from .llm import OpenAICompatibleClient

        try:
            llm = OpenAICompatibleClient()
        except RuntimeError as e:
            print(f"Cannot run --live: {e}", file=sys.stderr)
            return 2
        mode = "live"
        print("MODE: live -- real LLM from environment variables.")

    def show(done: int, total: int) -> None:
        print(f"{done}/{total} done", file=sys.stderr, flush=True)

    summary = evaluate(args.dir, get_ocr_engine(), llm, mode, delay=args.delay, progress=show).summary()
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

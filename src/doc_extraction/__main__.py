"""Usage: python -m doc_extraction <file> [<file> ...]  -> prints JSON per file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .llm import OpenAICompatibleClient
from .ocr import get_ocr_engine
from .pipeline import process_file


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    try:
        ocr, llm = get_ocr_engine(), OpenAICompatibleClient()
    except (RuntimeError, ValueError, ImportError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    for name in argv:
        r = process_file(Path(name), ocr, llm)
        print(json.dumps({
            "file": name,
            "needs_review": r.needs_review,
            "reasons": r.reasons,
            "attempts": r.attempts,
            "invoice": json.loads(r.invoice.model_dump_json()) if r.invoice else None,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

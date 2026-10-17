"""Generate fake invoices (PDF + PNG) with ground-truth JSON. Fully synthetic."""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .schema import Invoice, LineItem

VENDORS = ["Northwind Supplies", "Blue Harbor Trading", "Maple Leaf Tools", "Sunrise Stationery", "Orbit Parts Co"]
CUSTOMERS = ["Alpha Retail Ltd", "Bright Cafe", "Cedar Workshop", "Delta Clinic", "Echo Studio"]
PRODUCTS = ["Office chair", "Desk lamp", "Printer paper (box)", "USB cable", "Notebook set", "Wireless mouse", "Toner cartridge", "Whiteboard"]


def make_invoice(rng: random.Random, n: int) -> Invoice:
    items = []
    for name in rng.sample(PRODUCTS, rng.randint(1, 4)):
        qty = Decimal(rng.randint(1, 10))
        price = Decimal(rng.randint(200, 20000)) / 100
        items.append(LineItem(description=name, quantity=qty, unit_price=price, amount=qty * price))
    subtotal = sum(i.amount for i in items)
    tax = (subtotal * Decimal("0.10")).quantize(Decimal("0.01"))
    return Invoice(
        invoice_number=f"INV-{2024000 + n}",
        invoice_date=date(2024, 1, 1) + timedelta(days=rng.randint(0, 364)),
        vendor_name=rng.choice(VENDORS),
        customer_name=rng.choice(CUSTOMERS),
        currency="USD",
        line_items=items,
        subtotal=subtotal,
        tax=tax,
        total=subtotal + tax,
    )


def _lines(inv: Invoice) -> list[str]:
    out = [inv.vendor_name, f"Invoice No: {inv.invoice_number}", f"Date: {inv.invoice_date.isoformat()}",
           f"Bill To: {inv.customer_name}", "", "Description | Qty | Unit Price | Amount"]
    for li in inv.line_items:
        out.append(f"{li.description} | {li.quantity} | {li.unit_price:.2f} | {li.amount:.2f}")
    out += ["", f"Subtotal: {inv.subtotal:.2f}", f"Tax: {inv.tax:.2f}", f"Total ({inv.currency}): {inv.total:.2f}"]
    return out


def write_pdf(inv: Invoice, path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in _lines(inv):
        c.drawString(50, y, line)
        y -= 20
    c.save()


def write_png(inv: Invoice, path: Path) -> None:
    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default(size=26)
    y = 30
    for line in _lines(inv):
        d.text((40, y), line, fill="black", font=font)
        y += 40
    img.save(path)


def generate(out_dir: Path, count: int = 20, seed: int = 7) -> None:
    """Even-numbered docs become PDFs, odd-numbered become PNG scans."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for n in range(1, count + 1):
        inv = make_invoice(rng, n)
        stem = out_dir / f"invoice_{n:02d}"
        (stem.with_suffix(".json")).write_text(inv.model_dump_json(indent=2), encoding="utf-8")
        (write_pdf if n % 2 == 0 else write_png)(inv, stem.with_suffix(".pdf" if n % 2 == 0 else ".png"))


if __name__ == "__main__":
    generate(Path("data/sample"))
    print("generated")

"""Generate fake invoices (PDF + PNG) with ground-truth JSON. Fully synthetic.

Not every document is clean: dates are printed in different formats, some documents
omit the customer or the tax line, and some PNG scans are blurred, noisy and tilted.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from .schema import Invoice, LineItem

VENDORS = ["Northwind Supplies", "Blue Harbor Trading", "Maple Leaf Tools", "Sunrise Stationery", "Orbit Parts Co"]
CUSTOMERS = ["Alpha Retail Ltd", "Bright Cafe", "Cedar Workshop", "Delta Clinic", "Echo Studio"]
PRODUCTS = ["Office chair", "Desk lamp", "Printer paper (box)", "USB cable", "Notebook set", "Wireless mouse", "Toner cartridge", "Whiteboard"]

# Printed date formats (the ground truth is always ISO). Slash dates are day/month/year.
DATE_FORMATS = ["%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%B %d, %Y"]


def has_customer(n: int) -> bool:
    return n % 5 != 0


def has_tax(n: int) -> bool:
    return n % 7 != 0


def is_pdf(n: int) -> bool:
    return n % 2 == 0


def is_low_quality(n: int) -> bool:
    return not is_pdf(n) and n % 3 == 0


def make_invoice(rng: random.Random, n: int) -> Invoice:
    items = []
    for name in rng.sample(PRODUCTS, rng.randint(1, 4)):
        qty = Decimal(rng.randint(1, 10))
        price = Decimal(rng.randint(200, 20000)) / 100
        items.append(LineItem(description=name, quantity=qty, unit_price=price, amount=qty * price))
    subtotal = sum(i.amount for i in items)
    tax = (subtotal * Decimal("0.10")).quantize(Decimal("0.01")) if has_tax(n) else Decimal("0")
    return Invoice(
        invoice_number=f"INV-{2024000 + n}",
        invoice_date=date(2024, 1, 1) + timedelta(days=rng.randint(0, 364)),
        vendor_name=rng.choice(VENDORS),
        customer_name=rng.choice(CUSTOMERS) if has_customer(n) else None,
        currency="USD",
        line_items=items,
        subtotal=subtotal,
        tax=tax,
        total=subtotal + tax,
    )


def _lines(inv: Invoice, n: int) -> list[str]:
    printed_date = inv.invoice_date.strftime(DATE_FORMATS[n % len(DATE_FORMATS)])
    out = [inv.vendor_name, f"Invoice No: {inv.invoice_number}", f"Date: {printed_date}"]
    if inv.customer_name:
        out.append(f"Bill To: {inv.customer_name}")
    out += ["", "Description | Qty | Unit Price | Amount"]
    for li in inv.line_items:
        out.append(f"{li.description} | {li.quantity} | {li.unit_price:.2f} | {li.amount:.2f}")
    out += ["", f"Subtotal: {inv.subtotal:.2f}"]
    if has_tax(n):
        out.append(f"Tax: {inv.tax:.2f}")
    out.append(f"Total ({inv.currency}): {inv.total:.2f}")
    return out


def write_pdf(inv: Invoice, path: Path, n: int = 2) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in _lines(inv, n):
        c.drawString(50, y, line)
        y -= 20
    c.save()


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a poor scan: low resolution, blur, noise, slight tilt."""
    w, h = img.size
    img = img.resize((int(w * 0.75), int(h * 0.75))).filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    for _ in range(img.size[0] * img.size[1] // 60):
        x, y = rng.randrange(img.size[0]), rng.randrange(img.size[1])
        v = rng.randint(90, 200)
        px[x, y] = (v, v, v)
    return img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor="white")


def write_png(inv: Invoice, path: Path, n: int = 1, rng: random.Random | None = None) -> None:
    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default(size=26)
    y = 30
    for line in _lines(inv, n):
        d.text((40, y), line, fill="black", font=font)
        y += 40
    if is_low_quality(n):
        img = _degrade(img, rng or random.Random(n))
    img.save(path)


def generate(out_dir: Path, count: int = 20, seed: int = 7) -> None:
    """Even-numbered docs become PDFs, odd-numbered become PNG scans."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for n in range(1, count + 1):
        inv = make_invoice(rng, n)
        stem = out_dir / f"invoice_{n:02d}"
        stem.with_suffix(".json").write_text(inv.model_dump_json(indent=2), encoding="utf-8")
        if is_pdf(n):
            write_pdf(inv, stem.with_suffix(".pdf"), n)
        else:
            write_png(inv, stem.with_suffix(".png"), n, random.Random(n))


if __name__ == "__main__":
    generate(Path("data/sample"))
    print("generated")

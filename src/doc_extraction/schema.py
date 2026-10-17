"""Invoice schema. Validation lives in code, not in the LLM."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    amount: Decimal = Field(ge=0)


class Invoice(BaseModel):
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    vendor_name: str = Field(min_length=1)
    customer_name: str | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(default=Decimal("0"), ge=0)
    total: Decimal = Field(ge=0)

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


def arithmetic_issues(inv: Invoice) -> list[str]:
    """Plain-code consistency checks used later for the review flag."""
    issues: list[str] = []
    for i, li in enumerate(inv.line_items):
        if abs(li.quantity * li.unit_price - li.amount) > Decimal("0.01"):
            issues.append(f"line_items[{i}]: quantity*unit_price != amount")
    if abs(sum(li.amount for li in inv.line_items) - inv.subtotal) > Decimal("0.01"):
        issues.append("sum(line amounts) != subtotal")
    if abs(inv.subtotal + inv.tax - inv.total) > Decimal("0.01"):
        issues.append("subtotal + tax != total")
    return issues

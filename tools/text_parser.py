from __future__ import annotations

import re
from typing import Any


AMOUNT_PATTERN = r"(?P<amount>(?:K|ZMW|kwacha)?\s*\d[\d,]*(?:\.\d+)?)"
CONNECTOR_PATTERN = r"[^\dK\.\n!?]{0,20}"
BUSINESS_PATTERNS = {
    "grocery shop": [r"grocery shop", r"small shop", r"shop owner", r"mini mart", r"market shop"],
    "salon": [r"salon", r"barbershop", r"barber shop", r"barber"],
    "farmer": [r"farmer", r"farm", r"agri", r"produce seller"],
    "freelancer": [r"freelancer", r"consultant", r"designer", r"developer", r"writer"],
}


def _parse_amount(raw_value: str | None) -> float:
    if not raw_value:
        return 0.0
    cleaned = re.sub(r"[^0-9.]", "", raw_value.replace(",", ""))
    return float(cleaned) if cleaned else 0.0


def _extract_first_amount(text: str, patterns: list[str]) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _parse_amount(match.group("amount"))
    return 0.0


def _detect_business_type(text: str) -> str | None:
    for business_type, patterns in BUSINESS_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return business_type
    return None


def _detect_location(text: str) -> str | None:
    match = re.search(r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_category_amounts(text: str) -> dict[str, float]:
    category_patterns = {
        "stock": [
            rf"{AMOUNT_PATTERN}\s+(?:on|for)\s+(?:stock|inventory|restocking|products|supplies|inputs)",
            rf"(?:stock|inventory|restocking|products|supplies|inputs)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
        "rent": [
            rf"{AMOUNT_PATTERN}\s+(?:on|for)\s+rent",
            rf"rent(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
        "transport": [
            rf"{AMOUNT_PATTERN}\s+(?:on|for)\s+transport",
            rf"transport(?:ation)?(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
        "other_expenses": [
            rf"{AMOUNT_PATTERN}\s+(?:on|for)\s+other expenses",
            rf"other expenses(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
        "sales": [
            rf"(?:made|earned|generated|received)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}(?:\s+(?:in|from)\s+sales)?",
            rf"{AMOUNT_PATTERN}\s+(?:in|from)\s+sales",
            rf"(?:sales|revenue|income)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
        "debt": [
            rf"(?:owe|owing|debt|supplier debt)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
            rf"{AMOUNT_PATTERN}\s+(?:to|in)\s+(?:my\s+)?supplier",
        ],
        "loan_amount": [
            rf"{AMOUNT_PATTERN}\s+loan",
            rf"(?:loan|borrow)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        ],
    }

    return {
        category: _extract_first_amount(text, patterns)
        for category, patterns in category_patterns.items()
    }


def _extract_general_expenses(text: str) -> float:
    patterns = [
        rf"(?:total expenses|overall expenses|expenses were)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
        rf"(?:spent)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
    ]
    return _extract_first_amount(text, patterns)


def parse_business_input(text: str) -> dict[str, Any]:
    lowered = text.lower()
    extracted = _extract_category_amounts(text)

    expenses_breakdown = {
        "stock": extracted.get("stock", 0.0),
        "rent": extracted.get("rent", 0.0),
        "transport": extracted.get("transport", 0.0),
        "other_expenses": extracted.get("other_expenses", 0.0),
    }
    detailed_expenses = round(sum(expenses_breakdown.values()), 2)
    total_expenses = detailed_expenses or _extract_general_expenses(text)

    revenue = extracted.get("sales", 0.0)
    if revenue == 0:
        revenue = _extract_first_amount(
            text,
            [
                rf"(?:made|earned|revenue|income)(?:{CONNECTOR_PATTERN}){AMOUNT_PATTERN}",
                rf"{AMOUNT_PATTERN}\s+(?:in|from)\s+(?:sales|revenue|income)",
            ],
        )

    debt = extracted.get("debt", 0.0)
    loan_amount = extracted.get("loan_amount", 0.0)

    return {
        "raw_text": text,
        "revenue": round(revenue, 2),
        "expenses": round(total_expenses, 2),
        "expenses_breakdown": expenses_breakdown,
        "debt": round(debt, 2),
        "loan_amount": round(loan_amount, 2),
        "business_type": _detect_business_type(lowered),
        "location": _detect_location(text),
        "categories_detected": [name for name, value in expenses_breakdown.items() if value > 0],
    }

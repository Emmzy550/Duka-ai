"""
financial_engine.py — Single source of truth for ALL financial calculations.

No agent, no UI component should compute math independently.
Python does the arithmetic; LLMs only interpret results.
"""
from __future__ import annotations

import datetime
import re
from typing import Any


# ── Baseline extraction ────────────────────────────────────────────────────────

def get_baseline(session_state: dict) -> dict[str, Any] | None:
    """
    Return monthly baseline figures with source provenance.

    Priority:
      1. Uploaded document with monthly breakdown (most accurate)
      2. Report cashflow figures (from any analysis)
      3. Manual numbers entered in the form
      4. Amounts extracted from written description
    """
    report = session_state.get("analysis_report")

    # ── Priority 1: Monthly breakdown from uploaded document ──────────────────
    if report:
        doc_analysis = report.get("document_analysis") or {}
        monthly_breakdown = doc_analysis.get("monthly_breakdown") or []
        valid_months = [
            m for m in monthly_breakdown
            if isinstance(m, dict) and float(m.get("revenue") or 0) > 0
        ]
        if valid_months:
            avg_rev = sum(float(m.get("revenue") or 0) for m in valid_months) / len(valid_months)
            avg_exp = sum(float(m.get("expenses") or 0) for m in valid_months) / len(valid_months)
            return {
                "monthly_revenue": round(avg_rev, 2),
                "monthly_expenses": round(avg_exp, 2),
                "monthly_profit": round(avg_rev - avg_exp, 2),
                "profit_margin": round((avg_rev - avg_exp) / avg_rev * 100, 1) if avg_rev > 0 else 0.0,
                "debt": float((report.get("parsed_data") or {}).get("debt") or 0),
                "source": f"uploaded document ({len(valid_months)} month{'s' if len(valid_months) != 1 else ''} average)",
                "months_of_data": len(valid_months),
                "raw_months": valid_months,
            }

    # ── Priority 2: Report cashflow figures ───────────────────────────────────
    if report:
        cashflow = report.get("cashflow") or {}
        rev = float(cashflow.get("revenue") or 0)
        if rev > 0:
            exp = float(cashflow.get("expenses") or 0)
            profit = float(cashflow.get("profit") or rev - exp)
            return {
                "monthly_revenue": round(rev, 2),
                "monthly_expenses": round(exp, 2),
                "monthly_profit": round(profit, 2),
                "profit_margin": round(float(cashflow.get("profit_margin") or 0), 1),
                "debt": float((report.get("parsed_data") or {}).get("debt") or 0),
                "source": "analysis report",
                "months_of_data": 1,
                "raw_months": None,
            }

    # ── Priority 3: Manual numbers from form ──────────────────────────────────
    manual_revenue = float(session_state.get("manual_revenue") or 0)
    if manual_revenue > 0:
        manual_exp = float(session_state.get("manual_expenses") or 0)
        manual_debt = float(session_state.get("manual_debt") or 0)
        profit = manual_revenue - manual_exp
        return {
            "monthly_revenue": round(manual_revenue, 2),
            "monthly_expenses": round(manual_exp, 2),
            "monthly_profit": round(profit, 2),
            "profit_margin": round(profit / manual_revenue * 100, 1) if manual_revenue > 0 else 0.0,
            "debt": manual_debt,
            "source": "manual entry",
            "months_of_data": 1,
            "raw_months": None,
        }

    # ── Priority 4: Parse amounts from written description ────────────────────
    description = str(session_state.get("manual_notes") or "")
    if description:
        raw = re.findall(r"[Kk][\s]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", description)
        amounts = [float(a.replace(",", "")) for a in raw]
        if len(amounts) >= 2:
            revenue = max(amounts)
            expenses = sum(a for a in amounts if a != revenue)
            profit = revenue - expenses
            return {
                "monthly_revenue": round(revenue, 2),
                "monthly_expenses": round(expenses, 2),
                "monthly_profit": round(profit, 2),
                "profit_margin": round(profit / revenue * 100, 1) if revenue > 0 else 0.0,
                "debt": 0.0,
                "source": "extracted from description (verify accuracy)",
                "months_of_data": 1,
                "raw_months": None,
            }

    return None


# ── Metrics computation ────────────────────────────────────────────────────────

def get_loan_label(profit_margin_pct: float) -> tuple[str, str, str]:
    """Return (label, hex_color, icon) based purely on profit margin."""
    if profit_margin_pct >= 25:
        return ("Loan ready", "#10B981", "✅")
    if profit_margin_pct >= 15:
        return ("Loan possible with caution", "#F59E0B", "⚠️")
    return ("Limited borrowing capacity", "#EF4444", "❌")


def compute_metrics(baseline: dict[str, Any]) -> dict[str, Any]:
    """
    Compute ALL financial metrics from a baseline dict.
    Python does every calculation — LLMs never re-derive these.
    """
    rev = float(baseline.get("monthly_revenue") or 0)
    exp = float(baseline.get("monthly_expenses") or 0)
    debt = float(baseline.get("debt") or 0)
    profit = rev - exp
    margin = (profit / rev * 100) if rev > 0 else 0.0
    expense_ratio = (exp / rev * 100) if rev > 0 else 0.0

    # Business health score (0–100)
    health = 100
    if margin < 10:
        health -= 30
    elif margin < 20:
        health -= 15
    elif margin < 30:
        health -= 5
    if expense_ratio > 85:
        health -= 25
    elif expense_ratio > 70:
        health -= 10
    if debt > rev * 2:
        health -= 20
    elif debt > rev:
        health -= 10
    elif debt > 0:
        health -= 5
    health = max(0, min(100, health))

    if health >= 71:
        health_label, health_color = "Healthy", "#10B981"
    elif health >= 41:
        health_label, health_color = "Caution", "#F59E0B"
    else:
        health_label, health_color = "At risk", "#EF4444"

    # Loan readiness score (0–100)
    loan_score = 100
    if margin < 10:
        loan_score -= 40
    elif margin < 15:
        loan_score -= 25
    elif margin < 20:
        loan_score -= 10
    if debt > rev:
        loan_score -= 30
    elif debt > rev * 0.5:
        loan_score -= 15
    loan_score = max(0, min(100, loan_score))

    loan_label, loan_color, loan_icon = get_loan_label(margin)

    # Safe loan = 1.5× monthly profit (conservative SME heuristic)
    safe_loan = round(max(0.0, profit * 1.5), 2)
    safe_loan_formula = f"1.5× monthly profit (K{profit:,.2f} × 1.5)"

    return {
        "monthly_revenue": round(rev, 2),
        "monthly_expenses": round(exp, 2),
        "monthly_profit": round(profit, 2),
        "profit_margin_pct": round(margin, 1),
        "expense_ratio_pct": round(expense_ratio, 1),
        "debt": round(debt, 2),
        "health_score": health,
        "health_label": health_label,
        "health_color": health_color,
        "loan_score": loan_score,
        "loan_label": loan_label,
        "loan_color": loan_color,
        "loan_icon": loan_icon,
        "safe_loan_amount": safe_loan,
        "safe_loan_formula": safe_loan_formula,
    }


# ── Forecast computation ───────────────────────────────────────────────────────

SCENARIOS: dict[str, dict[str, Any]] = {
    "pessimistic": {
        "label": "Pessimistic",
        "color": "#EF4444",
        "rev_growth": -0.01,   # −1 % per month
        "exp_growth": 0.02,    # +2 % per month
        "description": "Slow season or rising costs",
        "icon": "📉",
    },
    "base_case": {
        "label": "Base Case",
        "color": "#2563EB",
        "rev_growth": 0.02,    # +2 % per month
        "exp_growth": 0.01,    # +1 % per month
        "description": "Steady, realistic growth",
        "icon": "📊",
    },
    "optimistic": {
        "label": "Optimistic",
        "color": "#10B981",
        "rev_growth": 0.04,    # +4 % per month
        "exp_growth": 0.005,   # +0.5 % per month
        "description": "Strong growth, controlled costs",
        "icon": "📈",
    },
}


def calculate_forecast(
    baseline: dict[str, Any],
    scenario_key: str = "base_case",
    months: int = 6,
) -> dict[str, Any]:
    """
    Compute a compound-growth monthly forecast.

    Both revenue AND expenses grow each month — never frozen.
    All arithmetic done here in Python; LLM only narrates.
    """
    scenario = SCENARIOS.get(scenario_key, SCENARIOS["base_case"])
    rev = float(baseline.get("monthly_revenue") or 0)
    exp = float(baseline.get("monthly_expenses") or 0)

    today = datetime.date.today().replace(day=1)
    rows: list[dict[str, Any]] = []

    for i in range(months):
        m_rev = rev * ((1 + scenario["rev_growth"]) ** i)
        m_exp = exp * ((1 + scenario["exp_growth"]) ** i)
        m_profit = m_rev - m_exp
        m_margin = (m_profit / m_rev * 100) if m_rev > 0 else 0.0

        # Month label: shift by i months
        month_date = (today + datetime.timedelta(days=32 * i)).replace(day=1)
        month_name = month_date.strftime("%b '%y")

        rows.append({
            "month_num": i + 1,
            "month_name": month_name,
            "revenue": round(m_rev, 2),
            "expenses": round(m_exp, 2),
            "profit": round(m_profit, 2),
            "margin": round(m_margin, 1),
            "profitable": m_profit > 0,
            "rev_change_pct": round(scenario["rev_growth"] * i * 100, 1),
            "exp_change_pct": round(scenario["exp_growth"] * i * 100, 1),
        })

    profitable_months = sum(1 for r in rows if r["profitable"])
    first_loss = next((r["month_name"] for r in rows if not r["profitable"]), None)
    best = max(rows, key=lambda r: r["profit"])
    worst = min(rows, key=lambda r: r["profit"])

    return {
        "scenario": scenario,
        "scenario_key": scenario_key,
        "months": rows,
        "summary": {
            "profitable_months": profitable_months,
            "total_months": months,
            "first_loss_month": first_loss,
            "all_profitable": profitable_months == months,
            "best_month": best,
            "worst_month": worst,
            "total_revenue": round(sum(r["revenue"] for r in rows), 2),
            "total_profit": round(sum(r["profit"] for r in rows), 2),
            "final_margin": rows[-1]["margin"],
            "final_revenue": rows[-1]["revenue"],
            "final_expenses": rows[-1]["expenses"],
            "final_profit": rows[-1]["profit"],
        },
    }


# ── Expense breakdown ──────────────────────────────────────────────────────────

def compute_expense_breakdown(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Extract expense categories and recompute totals from individual items.
    Always sums from line items — never trusts a pre-aggregated total.
    """
    if not report:
        return None

    parsed = report.get("parsed_data") or {}
    raw: dict = parsed.get("expenses_breakdown") or {}
    if not raw:
        return None

    items = []
    for name, amount in raw.items():
        try:
            amt = float(amount or 0)
        except (TypeError, ValueError):
            continue
        if amt > 0:
            items.append({"name": name, "amount": round(amt, 2)})

    if not items:
        return None

    items.sort(key=lambda x: x["amount"], reverse=True)
    total = round(sum(i["amount"] for i in items), 2)

    for item in items:
        item["pct_of_expenses"] = round(item["amount"] / total * 100, 1) if total > 0 else 0.0

    return {
        "categories": items,
        "total": total,
        "largest": items[0],
        "top_3": items[:3],
    }


# ── Forecast intent detection ─────────────────────────────────────────────────

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "eighteen": 18, "twenty": 20, "twenty four": 24, "twenty-four": 24, "thirty": 30,
}

_FORECAST_KEYWORDS: tuple[str, ...] = (
    "forecast", "project", "projection", "predict", "prediction",
    "outlook", "looking ahead", "ahead", "future", "next month",
    "next months", "next year", "coming months", "coming year",
    "in the next", "over the next", "what will", "what would",
    "expect", "expected", "will i make", "will my", "by next",
    "trajectory", "trend going forward", "going forward",
)


def _word_to_int(token: str) -> int | None:
    return _NUMBER_WORDS.get(token.lower().strip())


def detect_forecast_intent(question: str) -> dict[str, Any] | None:
    """
    Decide whether `question` is a forecast/projection question, and if so,
    extract the time horizon in months.

    Returns:
        {"months": int, "matched": str} when a forecast intent is detected.
        None otherwise.

    Months are clamped to [1, 36]. If the user says "next year" we use 12.
    If they say "next month" we use 1. Default when forecast keyword is present
    but no horizon is specified: 6 months.
    """
    if not question:
        return None
    q = question.lower().strip()

    has_forecast_kw = any(kw in q for kw in _FORECAST_KEYWORDS)

    months: int | None = None
    matched: str = ""

    # "in 10 months", "for 10 months", "next 10 months", "10 months ahead"
    m = re.search(r"(?:in|for|over|next|coming|the next)\s+(\d{1,2})\s+months?", q)
    if not m:
        m = re.search(r"(\d{1,2})\s+months?(?:\s+(?:ahead|out|forward|from now))?", q)
    if m:
        months = int(m.group(1))
        matched = m.group(0)

    # Word-based numbers: "in ten months", "next six months"
    if months is None:
        m = re.search(
            r"(?:in|for|over|next|coming|the next)?\s*"
            r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|eighteen|twenty[- ]four|twenty|thirty)"
            r"\s+months?",
            q,
        )
        if m:
            n = _word_to_int(m.group(1))
            if n:
                months = n
                matched = m.group(0)

    # "next year" / "coming year" / "in a year" → 12 months
    if months is None and re.search(r"(?:next|coming|in a|over the next|the next)\s+year", q):
        months = 12
        matched = "year"

    # "next month" → 1
    if months is None and "next month" in q:
        months = 1
        matched = "next month"

    # If a forecast keyword is present but no horizon, default to 6 months.
    if months is None:
        if has_forecast_kw:
            months = 6
            matched = "default"
        else:
            return None

    months = max(1, min(36, months))
    return {"months": months, "matched": matched}


# ── Baseline straight from a report (no session_state) ────────────────────────

def baseline_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    """
    Build a baseline dict directly from an analysis report — no session_state.
    Mirrors get_baseline priorities 1 and 2.
    """
    if not report:
        return None

    doc_analysis = report.get("document_analysis") or {}
    monthly_breakdown = doc_analysis.get("monthly_breakdown") or []
    valid_months = [
        m for m in monthly_breakdown
        if isinstance(m, dict) and float(m.get("revenue") or 0) > 0
    ]
    if valid_months:
        avg_rev = sum(float(m.get("revenue") or 0) for m in valid_months) / len(valid_months)
        avg_exp = sum(float(m.get("expenses") or 0) for m in valid_months) / len(valid_months)
        return {
            "monthly_revenue": round(avg_rev, 2),
            "monthly_expenses": round(avg_exp, 2),
            "monthly_profit": round(avg_rev - avg_exp, 2),
            "profit_margin": round((avg_rev - avg_exp) / avg_rev * 100, 1) if avg_rev > 0 else 0.0,
            "debt": float((report.get("parsed_data") or {}).get("debt") or 0),
            "source": f"uploaded document ({len(valid_months)} month average)",
            "months_of_data": len(valid_months),
            "raw_months": valid_months,
        }

    cashflow = report.get("cashflow") or {}
    rev = float(cashflow.get("revenue") or 0)
    if rev > 0:
        exp = float(cashflow.get("expenses") or 0)
        profit = float(cashflow.get("profit") or rev - exp)
        return {
            "monthly_revenue": round(rev, 2),
            "monthly_expenses": round(exp, 2),
            "monthly_profit": round(profit, 2),
            "profit_margin": round(float(cashflow.get("profit_margin") or 0), 1),
            "debt": float((report.get("parsed_data") or {}).get("debt") or 0),
            "source": "analysis report",
            "months_of_data": 1,
            "raw_months": None,
        }

    return None


def build_forecast_brief(report: dict[str, Any], months: int) -> dict[str, Any] | None:
    """
    Compute base/optimistic/pessimistic forecasts for `months` and bundle
    a compact text block + chart-ready data for the chat layer.
    """
    baseline = baseline_from_report(report)
    if not baseline:
        return None

    base = calculate_forecast(baseline, "base_case", months=months)
    opt = calculate_forecast(baseline, "optimistic", months=months)
    pess = calculate_forecast(baseline, "pessimistic", months=months)

    last_base = base["months"][-1]
    last_opt = opt["months"][-1]
    last_pess = pess["months"][-1]

    total_base_profit = base["summary"]["total_profit"]
    first_loss = base["summary"]["first_loss_month"]

    def _fmt(amount: float) -> str:
        return f"K{amount:,.0f}"

    text_lines = [
        f"VERIFIED FORECAST - {months} months ahead "
        f"(baseline: K{baseline['monthly_revenue']:,.0f} revenue, "
        f"K{baseline['monthly_expenses']:,.0f} expenses, "
        f"K{baseline['monthly_profit']:,.0f} profit; source: {baseline['source']}):",
        "",
        f"BASE CASE  (revenue +2%/mo, expenses +1%/mo) - month {months} ({last_base['month_name']}):",
        f"  Revenue:  {_fmt(last_base['revenue'])}",
        f"  Expenses: {_fmt(last_base['expenses'])}",
        f"  Profit:   {_fmt(last_base['profit'])}  (margin {last_base['margin']}%)",
        f"  Cumulative profit over {months} months: {_fmt(total_base_profit)}",
        f"  First loss-making month: {first_loss or 'none - every month profitable'}",
        "",
        f"OPTIMISTIC (revenue +4%/mo, expenses +0.5%/mo) - month {months} ({last_opt['month_name']}):",
        f"  Revenue:  {_fmt(last_opt['revenue'])}    Profit: {_fmt(last_opt['profit'])} (margin {last_opt['margin']}%)",
        "",
        f"PESSIMISTIC (revenue -1%/mo, expenses +2%/mo) - month {months} ({last_pess['month_name']}):",
        f"  Revenue:  {_fmt(last_pess['revenue'])}    Profit: {_fmt(last_pess['profit'])} (margin {last_pess['margin']}%)",
    ]

    return {
        "months": months,
        "baseline": baseline,
        "base_case": base,
        "optimistic": opt,
        "pessimistic": pess,
        "text_block": "\n".join(text_lines),
    }

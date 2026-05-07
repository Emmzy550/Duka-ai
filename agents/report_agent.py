"""Report Agent — compiles structured financial health reports.

Design contract
---------------
* Numbers: 100% from financial_engine.py (Python).  LLM never invents figures.
* Narrative: LLM writes only executive summary, recommendations, and risks — all
  grounded in the exact numbers passed as context.
* Output: a flat dict that the Streamlit page can render and export to PDF/Excel.
"""

from __future__ import annotations

import datetime
from typing import Any

from agents import request_llm_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n(value: Any, t: type = float, default: float = 0.0):
    """Safe numeric cast — returns default on None / empty string / error."""
    try:
        return t(value or 0)
    except (TypeError, ValueError):
        return default


# ── LLM system prompts ────────────────────────────────────────────────────────

_EXEC_SUMMARY_SYSTEM = (
    "You are writing the Executive Summary of a professional financial report for a Zambian SME. "
    "Return ONLY valid JSON with one key:\n"
    '{"summary": "<three sentences>"}\n\n'
    "Sentence 1: overall financial health with exact numbers (revenue, profit, margin).\n"
    "Sentence 2: the single biggest challenge or risk — be specific, include numbers.\n"
    "Sentence 3: the single most impactful recommendation with expected K impact.\n"
    "Use K for Kwacha. Do not include markdown, newlines, or extra keys."
)

_RECS_RISKS_SYSTEM = (
    "You are writing the Recommendations and Risks section of a financial report for a Zambian SME. "
    "Return ONLY valid JSON:\n"
    "{\n"
    '  "recommendations": [\n'
    '    {"action": "specific verb-phrase", "impact": "K… or …% change", "timeline": "Now|This month|1–3 months"},\n'
    '    {"action": "...", "impact": "...", "timeline": "..."},\n'
    '    {"action": "...", "impact": "...", "timeline": "..."}\n'
    '  ],\n'
    '  "risks": [\n'
    '    "specific risk with a number or timeframe",\n'
    '    "specific risk with a number or timeframe"\n'
    '  ]\n'
    "}\n"
    "Every recommendation must reference actual numbers from the financial context. "
    "No generic advice like 'reduce costs' — say WHICH cost and by HOW MUCH."
)


# ── Context builder (shared across both LLM calls) ────────────────────────────

def _build_context(baseline: dict, metrics: dict, report: dict) -> str:
    cf     = report.get("cashflow",  {})
    loan   = report.get("loan",      {})
    market = report.get("market",    {})
    adv    = report.get("advisor",   {})

    rev    = _n(baseline.get("monthly_revenue"))
    exp    = _n(baseline.get("monthly_expenses"))
    profit = _n(baseline.get("monthly_profit"))
    margin = _n(metrics.get("profit_margin_pct"))
    debt   = _n(baseline.get("debt"))

    return (
        f"Revenue:          K{rev:,.0f}/month\n"
        f"Expenses:         K{exp:,.0f}/month ({100*exp/max(rev,1):.0f}% of revenue)\n"
        f"Profit:           K{profit:,.0f}/month ({margin:.1f}% margin)\n"
        f"Debt outstanding: K{debt:,.0f}\n"
        f"Health Score:     {_n(metrics.get('health_score'), int)}/100"
        f" — {metrics.get('health_label', '')}\n"
        f"Loan Score:       {_n(metrics.get('loan_score'), int)}/100"
        f" — {metrics.get('loan_label', '')}\n"
        f"Safe borrowing:   K{_n(metrics.get('safe_loan_amount')):,.0f}"
        f" ({metrics.get('safe_loan_formula', '')})\n"
        f"Cash flow status: {cf.get('cash_flow_status', '')}\n"
        f"Warnings:         {'; '.join(cf.get('warnings', []))}\n"
        f"Market summary:   {market.get('market_summary', '')}\n"
        f"Opportunity:      {market.get('opportunity', '')}\n"
        f"Loan reason:      {loan.get('reason', '')}\n"
        f"Advisor actions:  {'; '.join(adv.get('next_actions', [])[:4])}\n"
    )


# ── Main public function ───────────────────────────────────────────────────────

def generate_full_report(
    baseline: dict,
    metrics: dict,
    report: dict,
    business_profile: dict | None = None,
) -> dict[str, Any]:
    """Return a complete structured report dict ready for rendering and export.

    Numbers come from baseline / metrics / report (all Python-computed).
    LLM contributes only executive summary, recommendations, and risks — always
    grounded in the numeric context passed to it.
    """
    from tools.financial_engine import compute_expense_breakdown, calculate_forecast

    cf     = report.get("cashflow",  {})
    loan   = report.get("loan",      {})
    market = report.get("market",    {})
    adv    = report.get("advisor",   {})

    rev    = _n(baseline.get("monthly_revenue"))
    exp    = _n(baseline.get("monthly_expenses"))
    profit = _n(baseline.get("monthly_profit"))
    margin = _n(metrics.get("profit_margin_pct"))
    exp_pct = 100 * exp / max(rev, 1)

    # Python: expense categories
    breakdown   = compute_expense_breakdown(report) or {}
    categories  = breakdown.get("categories", [])

    # Python: 6-month base-case forecast
    fc   = calculate_forecast(baseline, "base_case", months=6)
    fc_s = fc["summary"]

    # LLM: narrative only
    ctx              = _build_context(baseline, metrics, report)
    existing_actions = adv.get("next_actions", [])

    exec_json = request_llm_json(
        _EXEC_SUMMARY_SYSTEM,
        f"Financial context:\n{ctx}\nCash flow summary to expand on: {cf.get('summary', '')}",
        max_tokens=280,
    )
    exec_summary = (
        (exec_json or {}).get("summary")
        or cf.get("summary")
        or "Analysis complete. Review the sections below for your financial health details."
    )

    recs_json = request_llm_json(
        _RECS_RISKS_SYSTEM,
        f"Financial context:\n{ctx}\nExisting advisor actions: {'; '.join(existing_actions)}",
        max_tokens=520,
    )
    recommendations = (recs_json or {}).get("recommendations") or [
        {"action": a, "impact": "—", "timeline": "This month"}
        for a in existing_actions[:3]
    ]
    risks = (recs_json or {}).get("risks") or cf.get("warnings", [])[:2]

    bp = business_profile or {}
    return {
        # ── Metadata ──────────────────────────────────────────────────────────
        "generated_at":    datetime.datetime.now().strftime("%d %B %Y, %H:%M"),
        "business_name":   bp.get("business_type") or bp.get("products_services") or "Your Business",
        "location":        bp.get("location") or "Zambia",

        # ── Narrative (LLM-written) ────────────────────────────────────────
        "executive_summary": exec_summary,
        "recommendations":   recommendations[:3],
        "risks":             risks[:2],

        # ── Numbers (Python-computed) ──────────────────────────────────────
        "snapshot": {
            "revenue":          rev,
            "expenses":         exp,
            "profit":           profit,
            "margin_pct":       margin,
            "expense_pct":      exp_pct,
            "health_score":     _n(metrics.get("health_score"), int),
            "health_label":     metrics.get("health_label", "N/A"),
            "cash_flow_status": cf.get("cash_flow_status", "N/A"),
            "source":           baseline.get("source", "Financial analysis"),
        },
        "expense_categories": categories,

        "cash_flow": {
            "summary": cf.get("summary", ""),
            "status":  cf.get("cash_flow_status", ""),
            "insight": cf.get("key_insight") or cf.get("reasoning", ""),
        },

        "loan": {
            "score":       _n(loan.get("loan_readiness_score"), int),
            "status":      loan.get("status", ""),
            "safe_amount": _n(metrics.get("safe_loan_amount")),
            "formula":     metrics.get("safe_loan_formula", ""),
            "risk_level":  loan.get("risk_level", ""),
            "reason":      loan.get("reason", ""),
            "improve":     loan.get("how_to_improve", [])[:3],
        },

        "market": {
            "summary":        market.get("market_summary", ""),
            "opportunity":    market.get("opportunity", ""),
            "recommendation": market.get("recommendation", ""),
            "monitor":        market.get("monitor_next", [])[:3],
        },

        "forecast_6m": {
            "base_profit_m6":  fc_s.get("final_profit", 0),
            "all_profitable":  fc_s.get("all_profitable", True),
            "first_loss_month": fc_s.get("first_loss_month"),
            "total_profit":    fc_s.get("total_profit", 0),
            "profitable_months": fc_s.get("profitable_months", 6),
        },
    }

from __future__ import annotations

from typing import Any

from agents.market_intelligence_agent import analyze_market_intelligence


def analyze_market_insights(
    user_input: str,
    parsed_data: dict[str, Any],
    cashflow_result: dict[str, Any],
) -> dict[str, Any]:
    result = analyze_market_intelligence(
        user_input,
        parsed_data,
        cashflow_result,
        transaction_summary=None,
    )
    return {
        "business_type": result["business_type"],
        "market_insight": result["market_summary"],
        "opportunity": result["opportunity"],
        "risk": result["risk_factors"][0] if result["risk_factors"] else "",
        "recommendation": result["recommendation"],
        "reasoning": result["reasoning"],
        "used_llm": result["used_llm"],
    }

"""
Market Intelligence Agent — powered by live Tavily web search.

Replaces the static fallback with real, sourced market data.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents import load_prompt, request_llm_json

logger = logging.getLogger(__name__)

# ── Tavily client (lazy init so import never crashes) ──────────────────────────

def _get_tavily():
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=key)
    except Exception as e:
        logger.warning("Tavily unavailable: %s", e)
        return None


# ── Query builder ──────────────────────────────────────────────────────────────

def _build_search_queries(business_type: str | None, location: str | None) -> list[str]:
    """Generate targeted search queries for the specific business type."""
    bt = (business_type or "").lower()
    loc = location or "Zambia"

    if any(w in bt for w in ["salon", "hair", "beauty", "nail", "barber"]):
        return [
            f"hair salon business trends {loc} Zambia 2025 2026",
            f"hair products prices Zambia supply costs 2025",
            f"beauty industry Zambia market demand growth",
            f"Zambia SME small business challenges opportunities 2025",
        ]

    if any(w in bt for w in ["grocery", "shop", "retail", "store", "spaza", "supermarket"]):
        return [
            f"grocery retail {loc} Zambia market trends 2026",
            f"mealie meal food prices Zambia inflation 2025 2026",
            f"small retail business Zambia challenges opportunities",
            f"Zambia consumer spending food retail 2025",
        ]

    if any(w in bt for w in ["farm", "farmer", "agriculture", "crop", "maize", "soya"]):
        return [
            f"Zambia agriculture market prices maize 2025 2026",
            f"Zambia farming season outlook rainfall forecast 2025",
            f"FRA maize floor price Zambia 2025",
            f"Zambia fertilizer prices farmer input costs 2025",
        ]

    if any(w in bt for w in ["restaurant", "food", "catering", "takeaway", "fast food"]):
        return [
            f"restaurant food business {loc} Zambia trends 2025",
            f"food ingredients costs inflation Zambia 2025",
            f"Zambia dining out consumer habits 2025",
            f"Zambia SME food business challenges opportunities",
        ]

    if any(w in bt for w in ["transport", "taxi", "logistics", "delivery", "truck"]):
        return [
            f"transport logistics business Zambia {loc} 2025",
            f"fuel prices Zambia 2025 2026 petrol diesel",
            f"Zambia transport sector challenges opportunities 2025",
            f"Zambia road infrastructure business impact 2025",
        ]

    if any(w in bt for w in ["school", "education", "tuition", "training"]):
        return [
            f"education tutoring business Zambia {loc} 2025",
            f"Zambia school fees education spending 2025",
            f"Zambia education sector growth opportunities",
            f"Zambia youth population education demand 2025",
        ]

    # Generic Zambian SME
    return [
        f"{bt or 'small business'} {loc} Zambia market conditions 2025 2026",
        f"Zambia SME business challenges opportunities 2025 2026",
        f"Zambia economic outlook inflation exchange rate 2025",
        f"Zambia kwacha business environment investment 2025",
    ]


# ── Web search ─────────────────────────────────────────────────────────────────

def _run_web_searches(
    business_type: str | None,
    location: str | None,
) -> tuple[list[dict], list[dict]]:
    """
    Run Tavily searches and return (search_results, sources).
    Returns ([], []) if Tavily is unavailable.
    """
    tavily = _get_tavily()
    if not tavily:
        return [], []

    queries = _build_search_queries(business_type, location)
    search_results: list[dict] = []
    sources: list[dict] = []

    for query in queries:
        try:
            result = tavily.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
                include_raw_content=False,
            )
            block = {
                "query": query,
                "answer": result.get("answer") or "",
                "sources": [],
            }
            for r in result.get("results", []):
                src = {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content") or "")[:500],
                }
                block["sources"].append(src)
                sources.append(src)
            search_results.append(block)
        except Exception as e:
            logger.warning("Tavily search failed for '%s': %s", query, e)

    return search_results, sources


# ── Synthesis prompt ───────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are Duka AI's Market Intelligence Agent.
You have real, sourced web search results. Synthesise them into
actionable market intelligence for a Zambian small business owner.

RULES:
- Only use information from the supplied search results.
- If data is limited, say "based on available data..."
- Always reference Zambia/Africa context.
- Include at least one specific number or fact from the searches.
- Be concise — business owners are busy.
- Return JSON only.

Return this JSON shape:
{
  "market_summary": "2-3 sentences on current market conditions for this business type",
  "risk_factors": ["specific risk grounded in search data", "risk 2", "risk 3"],
  "opportunity": "specific opportunity with timing, grounded in search data",
  "recommendation": "one concrete action this month based on market conditions",
  "monitor_next": ["specific signal to watch 1", "signal 2", "signal 3"],
  "reasoning": "brief explanation of sources used"
}"""


def _build_synthesis_prompt(
    business_type: str | None,
    location: str | None,
    metrics: dict[str, Any],
    search_results: list[dict],
    monthly_block: str,
) -> str:
    search_context = ""
    for i, res in enumerate(search_results, 1):
        search_context += f"\n\nSearch {i}: '{res['query']}'\n"
        if res.get("answer"):
            search_context += f"Summary: {res['answer']}\n"
        for src in res["sources"][:2]:
            if src.get("title"):
                search_context += f"Source: {src['title']}\n"
            if src.get("content"):
                search_context += f"Content: {src['content']}\n"

    return (
        f"BUSINESS CONTEXT:\n"
        f"Type: {business_type or 'SME'}\n"
        f"Location: {location or 'Zambia'}\n"
        f"Monthly revenue: K{metrics.get('monthly_revenue', 0):,.0f}\n"
        f"Profit margin: {metrics.get('profit_margin_pct', 0):.1f}%\n"
        + (f"\n{monthly_block}\n" if monthly_block else "")
        + f"\nREAL WEB SEARCH RESULTS:{search_context}\n\n"
        "Using ONLY the search results above, return the JSON market intelligence object."
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_market_intelligence(
    user_input: str,
    parsed_data: dict[str, Any],
    cashflow_result: dict[str, Any],
    transaction_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    business_type = parsed_data.get("business_type")
    location = parsed_data.get("location")
    context_bundle = parsed_data.get("_analysis_context", {})
    document_analysis = context_bundle.get("document_analysis") or {}

    # Monthly breakdown block (for seasonal analysis)
    monthly_breakdown = document_analysis.get("monthly_breakdown") or []
    monthly_block = ""
    if isinstance(monthly_breakdown, list) and len(monthly_breakdown) > 1:
        lines = ["MONTHLY BREAKDOWN (cite specific months for seasonality questions):"]
        for entry in monthly_breakdown:
            if not isinstance(entry, dict):
                continue
            month = entry.get("month", "")
            parts = [f"  {month}:"]
            for key in ("revenue", "Revenue", "expenses", "Expenses", "profit", "Profit"):
                val = entry.get(key)
                if val is not None:
                    try:
                        parts.append(f"{key.lower()} K{float(val):,.0f}")
                    except (TypeError, ValueError):
                        pass
            lines.append(" ".join(parts))
        monthly_block = "\n".join(lines)

    # Metrics for context
    metrics = {
        "monthly_revenue": cashflow_result.get("revenue", 0),
        "profit_margin_pct": cashflow_result.get("profit_margin", 0),
    }

    # ── Live web search ────────────────────────────────────────────────────────
    search_results, sources = _run_web_searches(business_type, location)
    web_search_used = bool(search_results)

    # ── LLM synthesis ──────────────────────────────────────────────────────────
    if web_search_used:
        user_prompt = _build_synthesis_prompt(
            business_type, location, metrics, search_results, monthly_block
        )
    else:
        # No Tavily — still ask LLM for best-effort market intel
        user_prompt = (
            f"User input:\n{user_input}\n\n"
            f"Structured financial data:\n{json.dumps(parsed_data, indent=2)}\n\n"
            f"Cash flow analysis:\n{json.dumps(cashflow_result, indent=2)}\n\n"
            f"Analysis context:\n{json.dumps(context_bundle, indent=2)}\n\n"
            + (f"{monthly_block}\n\n" if monthly_block else "")
            + "Return practical market intelligence for this Zambian SME owner."
        )

    llm_result = request_llm_json(
        SYNTHESIS_SYSTEM + "\n\n" + load_prompt("market_prompt.md"),
        user_prompt,
    )

    llm_risks   = llm_result.get("risk_factors")  if isinstance((llm_result or {}).get("risk_factors"),  list) else None
    llm_monitor = llm_result.get("monitor_next")   if isinstance((llm_result or {}).get("monitor_next"),  list) else None

    queries_used = [r["query"] for r in search_results]

    return {
        "business_type": business_type or "default",
        "location": location,
        "market_summary": (llm_result or {}).get("market_summary", ""),
        "risk_factors": llm_risks or [],
        "opportunity": (llm_result or {}).get("opportunity", ""),
        "recommendation": (llm_result or {}).get("recommendation", ""),
        "monitor_next": llm_monitor or [],
        "reasoning": (llm_result or {}).get("reasoning", ""),
        "sources": sources,
        "queries_used": queries_used,
        "web_search_used": web_search_used,
        "fallback_notice": "",
        "used_llm": bool(llm_result),
    }

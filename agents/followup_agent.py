from __future__ import annotations

import json
import logging
from typing import Any, Generator

from agents import _extract_json_block, request_llm, request_llm_chat, request_llm_stream_chat
from tools.financial_calculator import format_currency
from tools.financial_engine import build_forecast_brief, detect_forecast_intent

logger = logging.getLogger(__name__)

# ── Per-agent conversational system prompts ────────────────────────────────────
# These are used for FOLLOW-UP chat only (plain text, not JSON).
# Each agent has a distinct personality and output structure.

_CASHFLOW_SYSTEM = """You are Duka AI's Cash Flow Agent in a live conversation.

PERSONALITY: Analytical and data-focused, but plain-spoken. You lead with the numbers from the supplied metrics, then explain what they mean. Think like a sharp accountant who speaks to small business owners, not accountants.

OUTPUT STRUCTURE (when appropriate):
1. State the specific financial position using the exact figures from the VERIFIED METRICS or VERIFIED FORECAST.
2. Explain how the numbers relate to each other.
3. Conclude with what it means practically for this business.

CONVERSATION RULES:
- Do NOT open with "As your Cash Flow Agent..." or any self-description. The user already sees your label. Start directly with the answer.
- Use ONLY the figures from VERIFIED METRICS or VERIFIED FORECAST when present — never calculate new percentages or amounts yourself.
- If the question cannot be answered from the supplied metrics/forecast (wrong topic, nonsense text, or detail not in the data), say briefly that you only have what's shown and what they can ask instead — never invent Kwacha amounts or month facts.
- FORECAST RULE: When a VERIFIED FORECAST block appears in your context, the user is asking a forward-looking question. ALWAYS lead with the concrete projected numbers (revenue, expenses, profit, margin) for the requested horizon. NEVER refuse with phrases like "cannot be determined" or "data only goes up to month X" — the forecast block already extends to the requested month. Close by offering: "Want me to show this on a chart?"
- If the user gives a short reply like "yes", "ok", or "tell me more", continue naturally from your previous response — do NOT restart the analysis.
- Use K (Kwacha) when referring to amounts naturally — not in every sentence.
- Speak in short paragraphs (2-4 sentences each). Be direct.
- End naturally — sometimes with a conclusion, sometimes with a question. Never with "...in Kwacha?" or any template.
- Do not mention market names or store names unless the user mentioned them first.
- BREAKDOWN AVAILABILITY RULE: Only ask for expense categories when the metrics block says "EXPENSE BREAKDOWN STATUS: NOT PROVIDED". If status is "PROVIDED", NEVER ask the user for their expense categories — cite the listed ones directly with their exact figures.
- NEVER output K0.00 as the answer to a breakdown question. If data is missing, ask for it.
- LENGTH RULE: Keep every response to a maximum of 3 short paragraphs. Never write more than 3 paragraphs.
- ACTION RULE: Only add a "Next step" line when (a) you have a concrete, decisive recommendation AND (b) you are NOT already asking the user a question. When you do add one, format it on its OWN paragraph, separated by a blank line, prefixed exactly: "**Next step:**" followed by one short imperative sentence. If the response ends with a question to the user, OMIT the Next step line entirely — never pair a question with a redundant "Next step" that just rephrases the question.
- NO REPEAT RULE: If an insight was already stated earlier in this conversation, skip it and give new information."""

_ADVISOR_SYSTEM = """You are Duka AI's Business Advisor in a live conversation.

PERSONALITY: Strategic and action-oriented. You sound like a trusted local business mentor who understands both the numbers and the real-world realities of running a small business. Direct, practical, and warm.

OUTPUT STRUCTURE (when appropriate):
1. Briefly acknowledge the situation using the actual figures.
2. Give 2-3 numbered, prioritized recommendations with expected impact based on the data.
3. End with one clear "do this first" conclusion.

CONVERSATION RULES:
- Do NOT open with "As your Business Advisor..." or any self-description. The user already sees your label. Start directly with the answer.
- Use ONLY the figures from VERIFIED METRICS — never recalculate anything.
- If the question is off-topic or cannot be grounded in the metrics, answer in one or two sentences without making up numbers; suggest what they can ask from their analysis instead.
- Reference specific expense categories and amounts from the breakdown when giving advice.
- If the user gives a short reply like "yes" or "tell me more", naturally elaborate on what you just said — do NOT restart from scratch.
- Do not mention market names or store names unless the user mentioned them first.
- Do not invent savings amounts — derive all figures from the supplied metrics.
- End naturally.
- BREAKDOWN AVAILABILITY RULE: Only ask for expense categories when the metrics block says "EXPENSE BREAKDOWN STATUS: NOT PROVIDED". If status is "PROVIDED", NEVER ask the user for their expense categories — cite the listed ones directly with their exact figures.
- NEVER output K0.00 as a cost figure. If data is missing, ask for it instead.
- LENGTH RULE: Keep every response to a maximum of 3 short paragraphs. Never write more than 3 paragraphs.
- ACTION RULE: Only add a "Next step" line when (a) you have a concrete, decisive recommendation AND (b) you are NOT already asking the user a question. When you do add one, format it on its OWN paragraph, separated by a blank line, prefixed exactly: "**Next step:**" followed by one short imperative sentence. If the response ends with a question to the user, OMIT the Next step line entirely — never pair a question with a redundant "Next step" that just rephrases the question.
- NO REPEAT RULE: If an insight was already stated earlier in this conversation, skip it and give new information."""

_LOAN_SYSTEM = """You are Duka AI's Loan Readiness Agent in a live conversation.

PERSONALITY: Evaluative but constructive. You think in scores, factors, and thresholds. You are honest about risk without being discouraging — like a credit officer who is genuinely trying to help the business improve its position.

OUTPUT STRUCTURE (when appropriate):
1. Anchor in the actual score and the specific factors driving it from the VERIFIED METRICS.
2. List 2-3 specific positive and negative factors affecting the score.
3. Give concrete improvement steps that address the gaps shown in the data.

CONVERSATION RULES:
- Do NOT open with "As your Loan Readiness Agent..." or any self-description. The user already sees your label. Start directly with the answer.
- Use ONLY the figures from VERIFIED METRICS — never recalculate anything.
- If the question is unrelated to loan readiness or the metrics shown, say so briefly — do not invent scores or amounts.
- If the user gives a short reply, continue the borrowing conversation naturally.
- Reference actual profit and debt figures from the metrics.
- Do not force institution names into every response — only mention lenders where genuinely relevant.
- End naturally. Do not use forced questions about Kwacha.
- LOAN FORMULA RULE: When citing the Suggested Max Loan amount, always briefly explain: "Based on 1.5× your monthly profit — a conservative rule of thumb to keep repayments manageable." Do NOT just throw out the number without any reasoning.
- BREAKDOWN AVAILABILITY RULE: Only ask for expense categories when the metrics block says "EXPENSE BREAKDOWN STATUS: NOT PROVIDED". If status is "PROVIDED", NEVER ask the user for their expense categories — cite the listed ones directly with their exact figures.
- LENGTH RULE: Keep every response to a maximum of 3 short paragraphs. Never write more than 3 paragraphs.
- ACTION RULE: Only add a "Next step" line when (a) you have a concrete, decisive recommendation AND (b) you are NOT already asking the user a question. When you do add one, format it on its OWN paragraph, separated by a blank line, prefixed exactly: "**Next step:**" followed by one short imperative sentence. If the response ends with a question to the user, OMIT the Next step line entirely — never pair a question with a redundant "Next step" that just rephrases the question.
- NO REPEAT RULE: If an insight was already stated earlier in this conversation, skip it and give new information."""

_MARKET_SYSTEM = """You are Duka AI's Market Intelligence Agent in a live conversation.

PERSONALITY: Insightful and context-aware. You connect business decisions to what's happening in the local market — seasons, demand cycles, external pressures. You sound like someone who has worked in Zambian and African markets for years.

OUTPUT STRUCTURE (when appropriate):
1. Share the most relevant market insight for this business type and current situation.
2. Connect it to the user's specific financial numbers from the VERIFIED METRICS.
3. Give a timing-based recommendation.

CONVERSATION RULES:
- Do NOT open with "As your Market Intelligence Agent..." or any self-description. The user already sees your label. Start directly with the answer.
- Give insights SPECIFIC to the business type and location in the data — not generic.
- If the question has nothing to do with market context or the supplied metrics, acknowledge limits in one sentence instead of inventing facts.
- Reference real Zambian/African market dynamics (seasonal patterns, local demand shifts, infrastructure challenges) only where genuinely relevant.
- MONTHLY DATA RULE: If MONTHLY BREAKDOWN DATA is provided in the context, you MUST reference specific months by name with their exact figures when answering seasonality, trend, or pattern questions. Example: "Your December peak of K14,900 vs January trough of K12,100 shows clear seasonality (18% swing)." Never give generic seasonal advice when actual month-by-month data is available.
- PRODUCT / TREND QUESTIONS: You rarely have per-SKU sales. For "are my products trending", use monthly revenue (if present), overall totals, and stated products/services — do not claim you lack all data if monthly or total revenue exists.
- Do not invent store names, market names, or specific percentages not in the data.
- If the user gives a short reply, continue the market discussion naturally.
- End naturally.
- LENGTH RULE: Keep every response to a maximum of 3 short paragraphs. Never write more than 3 paragraphs.
- ACTION RULE: Only add a "Next step" line when (a) you have a concrete, decisive recommendation AND (b) you are NOT already asking the user a question. When you do add one, format it on its OWN paragraph, separated by a blank line, prefixed exactly: "**Next step:**" followed by one short imperative sentence. If the response ends with a question to the user, OMIT the Next step line entirely — never pair a question with a redundant "Next step" that just rephrases the question.
- NO REPEAT RULE: If an insight was already stated earlier in this conversation, skip it and give new information."""

AGENT_SYSTEMS: dict[str, str] = {
    "cashflow": _CASHFLOW_SYSTEM,
    "advisor": _ADVISOR_SYSTEM,
    "loan": _LOAN_SYSTEM,
    "market": _MARKET_SYSTEM,
}

AGENT_LABELS: dict[str, tuple[str, str]] = {
    "cashflow": ("📊", "Cash Flow Agent"),
    "advisor": ("💡", "Business Advisor"),
    "loan": ("🏦", "Loan Readiness Agent"),
    "market": ("📈", "Market Intelligence Agent"),
    "meta": ("🧭", "Duka AI"),
}

_META_IDENTITY_REPLY = """I'm **Duka AI** — your financial assistant for **small businesses in Zambia and wider Africa**.

You're in **Chat Advisor** after your analysis: ask in plain language about **cash flow, what to do next, borrowing, or market timing**. I **route questions automatically** to the right lens (numbers vs advice vs loans vs market) so you don't have to choose an agent.

What would you like to look at first?"""

_META_CAPABILITIES_REPLY = """**Duka AI** uses specialist views behind one chat — all grounded in **your** figures:

• **Cash flow** — revenue, expenses, profit, cost drivers, forecasts  
• **Advisor** — prioritized actions tied to those numbers  
• **Loan readiness** — score (0–100) and a conservative borrowing estimate  
• **Market intel** — sector demand, prices, and timing for your business type  

Ask naturally; routing is automatic. For dedicated pages, use **Cash Flow Forecast**, **Loan Calculator**, or **Market Intel** in the sidebar."""

def _meta_response_for_question(question: str) -> str:
    """Short identity by default; expanded specialist list only for capability-style asks."""
    q = (question or "").lower().strip()
    capability_markers = (
        "what can you",
        "what could you",
        "capabilities",
        "what agents",
        "how does this work",
        "how do you work",
        "specialists",
        "each agent",
        "list",
        "features",
        "explain duka",
        "what do you offer",
    )
    identity_markers = ("who are you", "what are you", "introduce yourself", "your name")

    has_id = any(m in q for m in identity_markers)
    has_cap = any(m in q for m in capability_markers)
    if has_id and has_cap:
        return _META_IDENTITY_REPLY + "\n\n" + _META_CAPABILITIES_REPLY
    if has_cap:
        return _META_CAPABILITIES_REPLY
    if has_id:
        return _META_IDENTITY_REPLY
    # "what is this", "help me understand", etc. — keep Chat Advisor reply compact
    return _META_IDENTITY_REPLY

# ── LLM-based router ───────────────────────────────────────────────────────────

_ROUTING_SYSTEM = (
    "You are a router for a multi-agent AI financial advisor system. "
    "Classify the user's question and return ONLY a JSON object with an 'agents' key.\n\n"
    "Available agents:\n"
    "- meta: user asks what Duka AI can do, capabilities, or introduction questions\n"
    "- cashflow: DATA questions — 'show my expenses', 'what is my profit', 'explain my cash flow', "
    "'break down my costs', 'what is my margin', 'where is my money going', AND forward-looking projections "
    "('forecast for 10 months', 'what will my profit be next year', 'project ahead', 'in 6 months')\n"
    "- advisor: DECISION/ADVISORY questions — 'what should I do', 'how can I improve', "
    "'what do you recommend', 'what to cut', 'grow revenue', 'how do I reduce costs', "
    "'should I restock', 'what to focus on', any question asking for advice or action\n"
    "- loan: loan, borrow, credit, debt, repay, afford, bank, interest, loan readiness\n"
    "- market: season, market trends, timing, competition, demand, prices, local conditions\n"
    "- continuation: very short reply like 'yes', 'ok', 'tell me more', 'continue' (fewer than 6 words, no new topic)\n\n"
    "CRITICAL RULES:\n"
    "  'what expenses should I reduce?' → advisor (it asks for advice, not just data).\n"
    "  'show my cash flow' → cashflow (pure data request).\n"
    "  'forecast for 10 months', 'what will I make next year', 'project ahead' → cashflow (numeric projection).\n"
    "Rules: Use 'meta' alone for capability questions. Use 'continuation' alone when it applies. Max 2 agents otherwise.\n"
    "Return ONLY: {\"agents\": [\"agent1\"]} or {\"agents\": [\"agent1\", \"agent2\"]}"
)


_ROUTING_RULES: dict[str, list[str]] = {
    # Pure data/reporting questions → Cash Flow Agent
    "cashflow": [
        "where is my money going", "show my expenses", "what are my expenses",
        "how much am i spending", "show my revenue", "what is my profit",
        "explain my cash flow", "break down my costs", "what is my margin",
        "how much do i make", "show cash flow", "my expenses",
    ],
    # Advisory / decision questions → Business Advisor
    "advisor": [
        "what should i", "how can i", "should i", "what do you recommend",
        "what expenses to reduce", "how do i improve", "what to cut", "advice",
        "recommend", "suggest", "help me decide", "what to focus",
        "grow", "increase", "improve", "boost", "strategy",
        "how do i", "how should i", "next step", "action", "what should i",
        "focus on", "reduce", "cut costs",
    ],
    # Loan / credit questions → Loan Readiness Agent
    "loan": [
        "loan", "borrow", "credit", "debt", "repay", "afford", "lend",
        "interest", "bank", "finance myself", "ready for a loan", "loan ready",
    ],
    # Market / timing questions → Market Intelligence Agent
    "market": [
        "season", "market", "trend", "competition", "price", "demand",
        "when to", "timing", "holiday", "festive", "zambia", "lusaka",
        "local market", "price rise", "busy season",
    ],
    # Meta / capability questions → special handler
    "meta": [
        "what can you do", "what can you help", "what are you", "how do you work",
        "who are you", "what is this", "what do you do", "what agents",
        "your capabilities", "help me understand",
    ],
}


def _route_keyword_fallback(question: str) -> list[str]:
    q = question.lower().strip()

    # Short continuation replies
    if len(q.split()) <= 5 and any(k in q for k in ["yes", "ok", "sure", "more", "continue", "go on", "what else", "and", "tell me"]):
        return ["continuation"]

    # Forecast / projection questions go to cashflow (it has the numeric brief injected)
    if detect_forecast_intent(q):
        return ["cashflow"]

    # Check meta first (most specific)
    if any(k in q for k in _ROUTING_RULES["meta"]):
        return ["meta"]

    # Loan before market (more specific)
    if any(k in q for k in _ROUTING_RULES["loan"]):
        return ["loan"]

    if any(k in q for k in _ROUTING_RULES["market"]):
        return ["market"]

    # Advisory intent before raw data keywords
    if any(k in q for k in _ROUTING_RULES["advisor"]):
        if any(k in q for k in ["stock", "restock", "inventory", "what if"]):
            return ["advisor", "cashflow"]
        return ["advisor"]

    # Pure data questions
    if any(k in q for k in _ROUTING_RULES["cashflow"]):
        return ["cashflow"]

    return ["advisor"]


def route_question(question: str, recent_history: list[dict[str, str]]) -> list[str]:
    """Use the LLM to decide which agent(s) should handle this question.

    Falls back to keyword matching if the LLM call fails or returns garbage.
    Returns a list of agent names, e.g. ['cashflow'] or ['advisor', 'cashflow'].
    'continuation' means: reuse whatever agent just responded.
    """
    context = ""
    if recent_history:
        last = recent_history[-1]
        context = (
            f"\n\nMost recent exchange:\n"
            f"User: {last.get('question', '')}\n"
            f"Agent: {last.get('answer', '')[:120]}..."
        )

    routing_messages = [
        {"role": "system", "content": _ROUTING_SYSTEM},
        {"role": "user", "content": f'User question: "{question}"{context}\n\nJSON response:'},
    ]

    raw = request_llm_chat(routing_messages, temperature=0.1, max_tokens=40)
    if raw:
        parsed = _extract_json_block(raw)
        if parsed and isinstance(parsed.get("agents"), list):
            valid = {"cashflow", "advisor", "loan", "market", "continuation", "meta"}
            agents = [a for a in parsed["agents"] if a in valid]
            if agents:
                return agents

    return _route_keyword_fallback(question)


# ── Pre-computed financial metrics ─────────────────────────────────────────────

def compute_financial_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Compute ALL percentages and ratios in Python so the LLM never has to.

    The returned dict is injected verbatim into the agent's system context.
    Agents must use these numbers as-is and must NOT recalculate anything.
    """
    cashflow = report.get("cashflow", {})
    parsed = report.get("parsed_data", {})
    loan = report.get("loan", {})
    health = report.get("business_health", {})

    revenue: float = float(cashflow.get("revenue") or 0)
    expenses: float = float(cashflow.get("expenses") or 0)
    profit: float = float(cashflow.get("profit") or 0)
    profit_margin: float = float(cashflow.get("profit_margin") or 0)
    expense_ratio: float = float(cashflow.get("expense_ratio") or 0)
    debt: float = float(parsed.get("debt") or 0)

    raw_breakdown: dict = parsed.get("expenses_breakdown") or {}
    breakdown: dict[str, dict] = {}
    for cat, amt in raw_breakdown.items():
        try:
            amt_f = float(amt)
        except (TypeError, ValueError):
            continue
        if amt_f <= 0:
            continue
        breakdown[cat] = {
            "amount": round(amt_f, 2),
            "pct_of_revenue": round(amt_f / revenue * 100, 1) if revenue > 0 else 0.0,
            "pct_of_expenses": round(amt_f / expenses * 100, 1) if expenses > 0 else 0.0,
        }

    if breakdown:
        largest_cat, largest_data = max(breakdown.items(), key=lambda x: x[1]["amount"])
    else:
        largest_cat, largest_data = None, {"amount": 0.0, "pct_of_expenses": 0.0}

    # Extract monthly breakdown from document analysis for market agent
    doc_analysis = report.get("document_analysis") or {}
    monthly_breakdown: list = doc_analysis.get("monthly_breakdown") or []

    return {
        "revenue": round(revenue, 2),
        "expenses": round(expenses, 2),
        "profit": round(profit, 2),
        "profit_margin_pct": round(profit_margin, 1),
        "expense_ratio_pct": round(expense_ratio, 1),
        "debt": round(debt, 2),
        "debt_to_revenue_pct": round(debt / revenue * 100, 1) if revenue > 0 else 0.0,
        "loan_readiness_score": int(loan.get("loan_readiness_score") or 0),
        "loan_risk_level": loan.get("risk_level", "Unknown"),
        "suggested_loan_amount": round(float(loan.get("suggested_loan_amount") or 0), 2),
        "loan_multiplier": loan.get("loan_multiplier", 1.5),
        "business_health_score": int(health.get("score") or 0),
        "business_health_status": health.get("status", "Unknown"),
        "expense_breakdown": breakdown,
        "largest_expense_category": largest_cat,
        "largest_expense_amount": round(float(largest_data.get("amount", 0)), 2),
        "largest_expense_pct_of_expenses": float(largest_data.get("pct_of_expenses", 0)),
        "monthly_breakdown": monthly_breakdown,
    }


def _format_metrics_block(m: dict[str, Any]) -> str:
    """Format pre-computed metrics as a human-readable block for injection into prompts."""
    has_breakdown = bool(m["expense_breakdown"])
    largest_cat = m["largest_expense_category"]

    lines = [
        "VERIFIED FINANCIAL METRICS (computed by the system — do NOT recalculate these):",
        f"  Revenue:              K{m['revenue']:,.2f}",
        f"  Total Expenses:       K{m['expenses']:,.2f}  ({m['expense_ratio_pct']}% of revenue)",
        f"  Profit:               K{m['profit']:,.2f}  ({m['profit_margin_pct']}% margin)",
        f"  Debt:                 K{m['debt']:,.2f}  ({m['debt_to_revenue_pct']}% of revenue)",
    ]

    if has_breakdown and largest_cat:
        lines.append(
            f"  Largest Expense:      {largest_cat} — "
            f"K{m['largest_expense_amount']:,.2f}  "
            f"({m['largest_expense_pct_of_expenses']}% of expenses)"
        )
        lines.append(
            "  EXPENSE BREAKDOWN STATUS: PROVIDED — "
            "DO NOT ask the user for their expense categories; they are already listed below. "
            "Cite specific categories and amounts directly."
        )
    else:
        lines.append(
            "  EXPENSE BREAKDOWN STATUS: NOT PROVIDED — "
            "if asked about expenses, ask the user for their top 3 categories. "
            "Never say 'unknown'."
        )

    lines += [
        "",
        "  NOTE — Two separate scores, do NOT confuse them:",
        f"  Business Health Score: {m['business_health_score']}/100  ({m['business_health_status']})",
        f"    → Measures overall financial fitness: profitability, margin strength, expense control.",
        f"  Loan Readiness Score:  {m['loan_readiness_score']}/100  ({m['loan_risk_level']} risk)",
        f"    → Measures borrowing safety: whether taking on debt is manageable given current profit and existing debt.",
        f"  Suggested Max Loan:    K{m['suggested_loan_amount']:,.2f}",
        f"    → Rule of thumb: {'1.0× monthly profit (existing debt present — more conservative)' if m['debt'] > 0 else '1.5× monthly profit (no existing debt)'}.",
        f"    → When explaining this amount, say: 'Based on 1.5× your monthly profit — a conservative rule to keep repayments manageable.'",
    ]

    if has_breakdown:
        lines.append("  Expense Breakdown:")
        for cat, data in m["expense_breakdown"].items():
            lines.append(
                f"    {cat}: K{data['amount']:,.2f}  "
                f"({data['pct_of_expenses']}% of expenses, "
                f"{data['pct_of_revenue']}% of revenue)"
            )
    return "\n".join(lines)


# ── Deterministic cashflow answer for spend-breakdown questions ───────────────

def _is_spend_breakdown_question(question: str) -> bool:
    q = question.lower()
    triggers = [
        "where is my money going",
        "where am i spending",
        "where is the money going",
        "break down my expenses",
        "expense breakdown",
        "what am i spending on",
    ]
    return any(t in q for t in triggers)


def _build_spend_breakdown_reply(metrics: dict[str, Any]) -> str:
    revenue = float(metrics.get("revenue") or 0.0)
    expenses = float(metrics.get("expenses") or 0.0)
    profit = float(metrics.get("profit") or 0.0)
    expense_ratio = float(metrics.get("expense_ratio_pct") or 0.0)
    breakdown = metrics.get("expense_breakdown") or {}

    if not breakdown:
        return (
            f"Your total expenses are {format_currency(expenses)} ({expense_ratio:.1f}% of revenue), "
            f"and profit is {format_currency(profit)}.\n\n"
            "I can make this much more useful if you share your top 2-3 expense categories "
            "(for example: rent, wages, stock)."
        )

    top = sorted(breakdown.items(), key=lambda x: x[1]["amount"], reverse=True)[:3]
    lines = [
        (
            f"Your money is mainly going to expenses of {format_currency(expenses)} "
            f"against revenue of {format_currency(revenue)} (expense ratio: {expense_ratio:.1f}%)."
        ),
        "",
        "Top expense drivers:",
    ]
    for i, (cat, data) in enumerate(top, start=1):
        lines.append(
            f"{i}. {cat} — {format_currency(data['amount'])} "
            f"({data['pct_of_expenses']:.1f}% of expenses, {data['pct_of_revenue']:.1f}% of revenue)"
        )
    return "\n".join(lines)


# ── Message builder (multi-turn history) ───────────────────────────────────────

def build_messages_for_agent(
    agent_name: str,
    user_question: str,
    report: dict[str, Any],
    metrics: dict[str, Any],
    session_messages: list[dict],
) -> list[dict]:
    """Build the full messages array to pass to the LLM, including conversation history."""
    base_system = AGENT_SYSTEMS.get(agent_name, AGENT_SYSTEMS["advisor"])
    metrics_block = _format_metrics_block(metrics)
    profile = report.get("business_profile", {})
    profile_lines = "  " + "\n  ".join(f"{k}: {v}" for k, v in profile.items() if v) if profile else "  Not provided"

    monthly_block = ""
    if agent_name == "market":
        raw_monthly = metrics.get("monthly_breakdown") or []
        if isinstance(raw_monthly, list) and len(raw_monthly) > 1:
            lines = ["MONTHLY BREAKDOWN DATA (cite specific months and figures for seasonality/trend answers):"]
            for entry in raw_monthly:
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
            monthly_block = "\n".join(lines) + "\n"

    # Forecast injection — if the user is asking a forward-looking question,
    # compute the projection in Python and hand the agent the exact figures.
    forecast_block = ""
    intent = detect_forecast_intent(user_question)
    if intent:
        brief = build_forecast_brief(report, intent["months"])
        if brief:
            forecast_block = (
                f"{brief['text_block']}\n\n"
                "FORECAST RESPONSE RULES:\n"
                "- The user asked a forward-looking question. Use the VERIFIED FORECAST above as ground truth.\n"
                "- Lead with the specific numbers for the requested horizon — never say the data 'cannot be determined'.\n"
                "- State the base-case revenue, expenses, profit, and margin for the final month, in plain Kwacha.\n"
                "- Mention briefly how optimistic/pessimistic scenarios differ if it adds value.\n"
                "- End with a one-line offer: 'Want me to show this on a chart?' so the user can ask for a graph.\n"
            )

    system_content = (
        f"{base_system}\n\n"
        f"{metrics_block}\n\n"
        + (f"{monthly_block}\n" if monthly_block else "")
        + (f"{forecast_block}\n" if forecast_block else "")
        + f"BUSINESS PROFILE:\n{profile_lines}\n\n"
        "IMPORTANT: If the user gives a short follow-up like 'yes', 'tell me more', or 'continue', "
        "elaborate naturally on your previous response — do NOT restart the analysis from scratch."
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject conversation history, keeping it manageable (last 8 turns)
    history_msgs = []
    for msg in session_messages:
        role = msg.get("role", "")
        if role == "user":
            history_msgs.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            if msg.get("type") == "initial_analysis":
                # Summarise the initial analysis as a short assistant turn
                summary = report.get("final_summary", "Initial analysis completed.")
                advisor_note = report.get("advisor", {}).get("what_this_means", "")
                history_msgs.append({
                    "role": "assistant",
                    "content": f"[Initial Analysis]\n{summary}\n{advisor_note}",
                })
            elif msg.get("content"):
                history_msgs.append({"role": "assistant", "content": msg["content"]})

    # Keep only the last 8 turns (4 Q&A pairs) to stay within context limits
    messages.extend(history_msgs[-8:])
    messages.append({"role": "user", "content": user_question})
    return messages


# ── Public API ────────────────────────────────────────────────────────────────

_LLM_ERROR_MSG = "I'm having trouble reaching the AI right now. Please try again in a moment."


def stream_followup_for_agent(
    agent_name: str,
    user_question: str,
    report: dict[str, Any],
    session_messages: list[dict],
) -> Generator[str, None, None]:
    """Stream a response from the given agent, with full conversation history injected."""
    if agent_name == "meta":
        yield _meta_response_for_question(user_question)
        return

    metrics = compute_financial_metrics(report)
    if agent_name == "cashflow" and _is_spend_breakdown_question(user_question):
        yield _build_spend_breakdown_reply(metrics)
        return
    messages = build_messages_for_agent(agent_name, user_question, report, metrics, session_messages)

    yielded = False
    try:
        for chunk in request_llm_stream_chat(messages, temperature=0.4, max_tokens=700):
            yield chunk
            yielded = True
    except Exception as e:
        logger.error("stream_followup_for_agent failed: %s", e)

    if not yielded:
        yield _LLM_ERROR_MSG


def answer_followup_question(
    question: str,
    report: dict[str, Any],
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Non-streaming path — always calls the LLM via the advisor agent."""
    metrics = compute_financial_metrics(report)
    session_msgs: list[dict] = []
    for h in (history or []):
        if h.get("question"):
            session_msgs.append({"role": "user", "content": h["question"]})
        if h.get("answer"):
            session_msgs.append({"role": "assistant", "content": h["answer"]})
    messages = build_messages_for_agent("advisor", question, report, metrics, session_msgs)
    result = request_llm_chat(messages, temperature=0.4, max_tokens=700)
    return {"answer": result or _LLM_ERROR_MSG, "used_llm": bool(result)}


# Keep the old signature so existing imports in app.py don't break
def stream_followup_answer(
    question: str,
    report: dict[str, Any],
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Legacy entry point — routes internally. Prefer stream_followup_for_agent directly."""
    session_msgs: list[dict] = []
    for h in (history or []):
        if h.get("question"):
            session_msgs.append({"role": "user", "content": h["question"]})
        if h.get("answer"):
            session_msgs.append({"role": "assistant", "content": h["answer"]})

    metrics = compute_financial_metrics(report)
    messages = build_messages_for_agent("advisor", question, report, metrics, session_msgs)
    yielded = False
    try:
        for chunk in request_llm_stream_chat(messages, temperature=0.4, max_tokens=700):
            yield chunk
            yielded = True
    except Exception as e:
        logger.error("stream_followup_answer failed: %s", e)
    if not yielded:
        yield _LLM_ERROR_MSG

from __future__ import annotations

import json
from typing import Any

from agents import load_prompt, request_llm_json

SYSTEM_PROMPT = """You are Duka AI's Financial Advisor Agent.

PERSONALITY: Strategic and action-oriented. You sound like a trusted local business mentor — someone who understands both the numbers and real-world conditions of running a small business. You give clear, prioritized recommendations based on actual data, not generic advice.

YOUR JOB: Provide specific, actionable business recommendations based on the financial data supplied.

RULES:
- Use ONLY the financial figures provided in the prompt — never invent amounts or recalculate percentages.
- Name specific expense categories from the user's actual breakdown, not generic categories.
- Give 2-3 numbered, prioritized actions whose expected impact is derived from the actual figures.
- Use Kwacha (K) naturally when amounts are known from the data. Do not force it.
- Do NOT include specific location names, store names, or market names unless the user mentioned them.
- Do NOT include example savings amounts (like "K450/month") unless they are computed from the actual data.
- Speak in simple English any business owner can understand.
- End responses naturally. Do not use templated endings.
- Return JSON only.

Return this JSON shape:
{
  "main_issue": "specific issue derived from the supplied figures",
  "what_this_means": "plain-English interpretation using the actual amounts",
  "doing_well": "what is working, with figures from the data",
  "needs_attention": "main risk naming the specific category and amount from the data",
  "best_next_move": "specific action with expected impact based on actual numbers",
  "this_weeks_action": "one concrete action with a target derived from the data",
  "growth_recommendation": "specific growth advice grounded in the actual figures",
  "next_actions": ["Prioritized action 1 with expected impact", "Prioritized action 2 with expected impact", "Prioritized action 3 with expected impact"],
  "reasoning": "brief reasoning"
}
"""


def analyze_financial_advice(
    user_input: str,
    parsed_data: dict[str, Any],
    cashflow_result: dict[str, Any],
) -> dict[str, Any]:
    context_bundle = parsed_data.get("_analysis_context", {})

    llm_result = request_llm_json(
        SYSTEM_PROMPT + "\n\n" + load_prompt("advisor_prompt.md"),
        (
            f"User input:\n{user_input}\n\n"
            f"Structured financial data:\n{json.dumps(parsed_data, indent=2)}\n\n"
            f"Cash flow analysis:\n{json.dumps(cashflow_result, indent=2)}\n\n"
            f"Analysis context:\n{json.dumps(context_bundle, indent=2)}\n\n"
            "Provide practical financial advice for this Zambian business owner."
        ),
    )
    llm_next_actions = llm_result.get("next_actions") if isinstance(llm_result.get("next_actions") if llm_result else None, list) else None

    return {
        "main_issue": (llm_result or {}).get("main_issue", ""),
        "advice": (llm_result or {}).get("what_this_means", ""),
        "what_this_means": (llm_result or {}).get("what_this_means", ""),
        "doing_well": (llm_result or {}).get("doing_well", ""),
        "needs_attention": (llm_result or {}).get("needs_attention", ""),
        "best_next_move": (llm_result or {}).get("best_next_move", ""),
        "this_weeks_action": (llm_result or {}).get("this_weeks_action", ""),
        "growth_recommendation": (llm_result or {}).get("growth_recommendation", ""),
        "next_actions": llm_next_actions[:5] if llm_next_actions else [],
        "reasoning": (llm_result or {}).get("reasoning", ""),
        "used_llm": bool(llm_result),
    }

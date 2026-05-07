from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
AMD_REQUEST_TIMEOUT_SECONDS = 300.0


_PROVIDER_BANNER_PRINTED = False


def _print_provider_banner() -> None:
    global _PROVIDER_BANNER_PRINTED
    if _PROVIDER_BANNER_PRINTED:
        return
    settings = get_settings()
    if settings.llm_provider == "amd":
        msg = f"[AMD] Running on AMD MI300X GPU - model: {settings.amd_model}"
    elif settings.llm_provider == "openai":
        msg = f"[OpenAI-compatible] model: {settings.model_name}"
    else:
        msg = f"[LLM] provider={settings.llm_provider} model={settings.amd_model or settings.model_name}"
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    _PROVIDER_BANNER_PRINTED = True


def load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    return prompt_path.read_text(encoding="utf-8")


def _extract_json_block(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def get_chat_model() -> Any | None:
    settings = get_settings()
    if not settings.llm_enabled:
        return None

    _print_provider_banner()

    try:
        import httpx
    except Exception:
        return None

    if settings.llm_provider == "amd":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            logger.error("langchain_openai import failed: %s", e)
            return None

        try:
            return ChatOpenAI(
                base_url=settings.amd_base_url,
                api_key=settings.amd_api_key,
                model=settings.amd_model,
                temperature=0.3,
                max_retries=4,
                request_timeout=AMD_REQUEST_TIMEOUT_SECONDS,
                http_client=httpx.Client(trust_env=False, timeout=AMD_REQUEST_TIMEOUT_SECONDS),
                http_async_client=httpx.AsyncClient(
                    trust_env=False, timeout=AMD_REQUEST_TIMEOUT_SECONDS
                ),
            )
        except Exception as e:
            logger.error("ChatOpenAI (AMD) init failed: %s", e)
            return None

    if settings.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            logger.error("langchain_openai import failed: %s", e)
            return None

        try:
            return ChatOpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=settings.model_name,
                temperature=0.3,
                max_retries=2,
                http_client=httpx.Client(trust_env=False, timeout=120.0),
                http_async_client=httpx.AsyncClient(trust_env=False, timeout=120.0),
            )
        except Exception as e:
            logger.error("ChatOpenAI (OpenAI-compatible) init failed: %s", e)
            return None

    return None


def request_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 450,
) -> Optional[str]:
    llm = get_chat_model()
    if llm is None:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error("LLM invoke failed: %s", e)
        print(f"LLM invoke failed: {e}")
        return f"Error: {str(e)}"

    message = getattr(response, "content", None)
    if isinstance(message, str):
        return message.strip()
    return None


def request_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 700,
) -> dict[str, Any] | None:
    response = request_llm(
        system_prompt,
        user_prompt + "\n\nReturn valid JSON only. Do not include markdown fences or extra commentary.",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response:
        return None
    return _extract_json_block(response)


def request_llm_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 600,
):
    """Generator that yields text chunks from LLM for streaming display."""
    llm = get_chat_model()
    if llm is None:
        return

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        for chunk in llm.stream(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                yield content
    except Exception as e:
        logger.error("LLM stream failed: %s", e)
        print(f"LLM stream failed: {e}")
        yield f"Error: {str(e)}"
        return


def request_llm_stream_chat(
    messages: list[dict],
    *,
    temperature: float = 0.4,
    max_tokens: int = 700,
):
    """Generator that streams LLM response for a full multi-turn conversation.

    ``messages`` is a list of dicts with keys ``role`` (system/user/assistant)
    and ``content`` (str).  This preserves conversation history so the model
    can reference earlier turns when handling short follow-ups like "yes" or
    "tell me more".
    """
    llm = get_chat_model()
    if llm is None:
        return

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        for chunk in llm.stream(lc_messages, temperature=temperature, max_tokens=max_tokens):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                yield content
    except Exception as e:
        logger.error("LLM stream_chat failed: %s", e)
        print(f"LLM stream_chat failed: {e}")
        yield f"Error: {str(e)}"
        return


def request_llm_chat(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 80,
) -> Optional[str]:
    """Non-streaming single response for a multi-turn conversation (used for routing)."""
    llm = get_chat_model()
    if llm is None:
        return None

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = llm.invoke(lc_messages, temperature=temperature, max_tokens=max_tokens)
        text = getattr(response, "content", None)
        return text.strip() if isinstance(text, str) else None
    except Exception as e:
        logger.error("LLM chat failed: %s", e)
        print(f"LLM chat failed: {e}")
        return f"Error: {str(e)}"

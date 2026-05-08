from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    openai_api_key: str
    openai_base_url: str
    amd_base_url: str
    amd_api_key: str
    amd_model: str
    model_name: str
    app_env: str
    tavily_api_key: str

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "amd":
            return bool(self.amd_base_url and self.amd_model)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_base_url and self.model_name)
        return False

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "amd").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        amd_base_url=os.getenv("AMD_BASE_URL", "http://165.245.131.144:8000/v1").strip(),
        amd_api_key=os.getenv("AMD_API_KEY", "duka-ai-key").strip(),
        amd_model=os.getenv("AMD_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip(),
        model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile").strip(),
        app_env=os.getenv("APP_ENV", "development").strip(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
    )

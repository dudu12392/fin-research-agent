"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LLM + SEC configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM / DeepSeek ──────────────────────────────────────────────
    openai_api_key: str = "your_key_here"
    openai_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # ── SEC EDGAR ───────────────────────────────────────────────────
    sec_user_agent: str = "Liang Chang 18845976355@163.com"

    # ── Application ─────────────────────────────────────────────────
    log_level: str = "INFO"
    max_retries: int = 3
    request_timeout: int = 30


settings = Settings()

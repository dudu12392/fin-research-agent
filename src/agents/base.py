"""Abstract base class for all agents in the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI

from core.config import settings
from core.logging import get_logger


class BaseAgent(ABC):
    """Abstract agent with shared LLM client and logging."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.1,
        )

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's core logic and return updated state."""
        ...

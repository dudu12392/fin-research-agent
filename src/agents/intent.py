"""Intent recognition agent — classifies user query into predefined intents."""

from __future__ import annotations

from typing import Any

import yaml
from pathlib import Path

from agents.base import BaseAgent

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class IntentAgent(BaseAgent):
    """Classify user queries into intents: company_overview, financial_analysis, etc."""

    def __init__(self) -> None:
        super().__init__("intent")
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPTS_DIR / "intent.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("system", "")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        self.logger.info("classifying_intent", query=query)
        response = await self.llm.ainvoke(
            [("system", self._system_prompt), ("human", query)]
        )
        intent = (response.content or "").strip().lower()
        state["intent"] = intent
        return state

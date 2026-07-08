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

    def classify(self, query: str) -> str:
        """Synchronous intent classification for RAG routing.

        Uses a dedicated prompt to return exactly one of 6 categories.
        """
        classify_prompt = (
            "Classify the user query into EXACTLY ONE of these categories.\n"
            "Reply with ONLY the category name, nothing else.\n\n"
            "single_fact — asks about ONE company and ONE financial number\n"
            "compare — compares two or more companies on the same metric\n"
            "trend — asks about changes over multiple years\n"
            "ambiguous — vague query missing specific metric or year\n"
            "chitchat — greeting, thanks, capability question\n"
            "out_of_scope — stock price prediction, macro economics, crypto, non-financial\n"
        )
        response = self.llm.invoke(
            [("system", classify_prompt), ("human", query)]
        )
        raw = (response.content or "").strip().lower()

        valid = {"single_fact", "compare", "trend", "ambiguous", "chitchat", "out_of_scope"}
        result = "single_fact"
        for cat in valid:
            if cat in raw:
                result = cat
                break

        # Heuristic post-correction: queries <10 chars with no year → ambiguous
        if result == "single_fact":
            clean = query.replace("?", "").replace("。", "").replace(" ", "")
            has_year = any(w.isdigit() and len(w) == 4 for w in query.split())
            if not has_year and len(clean) <= 10:
                result = "ambiguous"

        return result

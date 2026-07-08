"""Analysis & report-generation agent."""

from __future__ import annotations

import json
from typing import Any

import yaml
from pathlib import Path

from agents.base import BaseAgent

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class AnalystAgent(BaseAgent):
    """Analyze fetched data and generate a research report."""

    def __init__(self) -> None:
        super().__init__("analyst")
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPTS_DIR / "analyst.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("system", "")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        data = state.get("data", {})

        context = json.dumps(data, indent=2, ensure_ascii=False)
        user_msg = f"Query: {query}\nData:\n{context}"

        self.logger.info("analyzing", query=query)
        response = await self.llm.ainvoke(
            [("system", self._system_prompt), ("human", user_msg)]
        )
        state["report"] = response.content or ""
        return state

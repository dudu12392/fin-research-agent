"""Task-planner agent — decomposes a research query into executable steps."""

from __future__ import annotations

import json
from typing import Any

import yaml
from pathlib import Path

from agents.base import BaseAgent

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PlannerAgent(BaseAgent):
    """Break down a research question into a sequence of sub-tasks."""

    def __init__(self) -> None:
        super().__init__("planner")
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPTS_DIR / "planner.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("system", "")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        intent = state.get("intent", "")
        self.logger.info("planning_tasks", query=query, intent=intent)

        user_msg = f"Intent: {intent}\nQuery: {query}"
        response = await self.llm.ainvoke(
            [("system", self._system_prompt), ("human", user_msg)]
        )
        try:
            plan = json.loads(response.content or "{}")
        except json.JSONDecodeError:
            plan = {"steps": [{"action": "fallback", "params": {"query": query}}]}

        state["plan"] = plan
        return state

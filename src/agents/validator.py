"""Validation agent — verifies report quality and factual accuracy."""

from __future__ import annotations

import json
from typing import Any

import yaml
from pathlib import Path

from agents.base import BaseAgent

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class ValidatorAgent(BaseAgent):
    """Validate the generated report against source data."""

    def __init__(self) -> None:
        super().__init__("validator")
        self._system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = PROMPTS_DIR / "validator.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("system", "")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        report = state.get("report", "")
        data = state.get("data", {})
        self.logger.info("validating_report")

        context = json.dumps(data, indent=2, ensure_ascii=False)
        user_msg = f"Report:\n{report}\n\nSource Data:\n{context}"

        response = await self.llm.ainvoke(
            [("system", self._system_prompt), ("human", user_msg)]
        )
        state["validation"] = response.content or ""
        return state

"""Task planner — decomposes complex financial queries into atomic sub-questions."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI


class PlannerAgent:
    """Break compare/trend queries into per-company, per-year sub-queries."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            temperature=0.1,
            max_tokens=512,
        )

    def decompose(self, query: str, intent: str) -> list[dict[str, Any]]:
        """Decompose a complex query into atomic sub-queries.

        Args:
            query: User's original question.
            intent: Classified intent (compare / trend).

        Returns:
            List of dicts with keys: sub_query, target_company, target_year, target_metric.
        """
        system = (
            "You are a financial query decomposer. Break the user's question into "
            "independent atomic sub-queries. Each sub-query asks about ONE company "
            "and ONE year. Return ONLY a JSON array.\n\n"
            "Format:\n"
            '[{"sub_query": "...", "target_company": "AAPL", "target_year": 2025, "target_metric": "revenue"}, ...]\n\n'
            "For compare intents: decompose into one sub-query per company.\n"
            "For trend intents: decompose into one sub-query per year.\n"
            "For unknown years, use null. For unknown metrics, use 'summary'."
        )

        response = self.llm.invoke([("system", system), ("human", query)])
        content = response.content or "[]"

        try:
            # Strip markdown code fences
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return [{"sub_query": query, "target_company": "AAPL", "target_year": 2025, "target_metric": "summary"}]

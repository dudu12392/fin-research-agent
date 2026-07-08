"""LangGraph state graph — orchestrates the research agent pipeline."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from agents.intent import IntentAgent
from agents.planner import PlannerAgent
from agents.fetcher import FetcherAgent
from agents.analyst import AnalystAgent
from agents.validator import ValidatorAgent
from core.logging import get_logger

logger = get_logger("graph.research")


class ResearchState(TypedDict, total=False):
    """Shared state flowing through the agent graph."""

    query: str
    messages: Annotated[list, add_messages]
    intent: str
    plan: dict[str, Any]
    data: dict[str, Any]
    report: str
    validation: str
    error: str | None


class ResearchGraph:
    """Build and run the financial research pipeline as a LangGraph state graph."""

    def __init__(self) -> None:
        self.intent_agent = IntentAgent()
        self.planner_agent = PlannerAgent()
        self.fetcher_agent = FetcherAgent()
        self.analyst_agent = AnalystAgent()
        self.validator_agent = ValidatorAgent()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ResearchState)

        # ── Nodes ──────────────────────────────────────────────
        builder.add_node("intent", self._intent_node)
        builder.add_node("planner", self._planner_node)
        builder.add_node("fetcher", self._fetcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("validator", self._validator_node)

        # ── Edges ──────────────────────────────────────────────
        builder.set_entry_point("intent")
        builder.add_edge("intent", "planner")
        builder.add_edge("planner", "fetcher")
        builder.add_edge("fetcher", "analyst")
        builder.add_edge("analyst", "validator")
        builder.set_finish_point("validator")

        return builder.compile()

    async def _intent_node(self, state: ResearchState) -> ResearchState:
        result = await self.intent_agent.run(dict(state))
        return {**state, **result}  # type: ignore[return-value]

    async def _planner_node(self, state: ResearchState) -> ResearchState:
        result = await self.planner_agent.run(dict(state))
        return {**state, **result}  # type: ignore[return-value]

    async def _fetcher_node(self, state: ResearchState) -> ResearchState:
        result = await self.fetcher_agent.run(dict(state))
        return {**state, **result}  # type: ignore[return-value]

    async def _analyst_node(self, state: ResearchState) -> ResearchState:
        result = await self.analyst_agent.run(dict(state))
        return {**state, **result}  # type: ignore[return-value]

    async def _validator_node(self, state: ResearchState) -> ResearchState:
        result = await self.validator_agent.run(dict(state))
        return {**state, **result}  # type: ignore[return-value]

    async def run(self, query: str) -> dict[str, Any]:
        """Execute the full pipeline for a single research query."""
        logger.info("pipeline_start", query=query)
        initial_state: ResearchState = {"query": query, "messages": []}
        result = await self._graph.ainvoke(initial_state)
        logger.info("pipeline_complete")
        return result


# Singleton
research_graph = ResearchGraph()

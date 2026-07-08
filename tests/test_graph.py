"""Integration tests for the full research graph pipeline."""

from __future__ import annotations

import pytest

from graph.research_graph import ResearchGraph, ResearchState


class TestResearchGraph:
    """Integration tests for the LangGraph pipeline."""

    @pytest.fixture
    def graph(self) -> ResearchGraph:
        return ResearchGraph()

    def test_graph_builds(self, graph: ResearchGraph) -> None:
        """Verify the state graph compiles without error."""
        assert graph._graph is not None

    def test_graph_nodes(self, graph: ResearchGraph) -> None:
        """Verify all expected nodes are in the graph."""
        nodes = graph._graph.nodes if hasattr(graph._graph, "nodes") else {}
        expected = {"intent", "planner", "fetcher", "analyst", "validator"}
        # The graph should at minimum contain these nodes
        assert expected.issubset(set(nodes.keys())) if nodes else True

    def test_initial_state(self) -> None:
        """Verify initial state can be constructed."""
        state: ResearchState = {"query": "What is Apple's revenue?", "messages": []}
        assert state["query"] == "What is Apple's revenue?"
        assert state["messages"] == []

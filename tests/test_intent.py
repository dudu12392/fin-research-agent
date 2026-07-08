"""Tests for intent recognition agent."""

from __future__ import annotations

import pytest

from agents.intent import IntentAgent


class TestIntentAgent:
    """Unit tests for intent classification."""

    @pytest.fixture
    def agent(self) -> IntentAgent:
        return IntentAgent()

    def test_agent_initialization(self, agent: IntentAgent) -> None:
        """Verify agent initializes with correct name and has an LLM."""
        assert agent.name == "intent"
        assert agent.llm is not None

    def test_prompt_loaded(self, agent: IntentAgent) -> None:
        """Verify system prompt is loaded from YAML."""
        assert agent._system_prompt is not None
        assert len(agent._system_prompt) > 0
        assert "financial research intent" in agent._system_prompt.lower()

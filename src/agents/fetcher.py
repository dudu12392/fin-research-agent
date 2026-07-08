"""Data-fetching agent — retrieves SEC filings via edgartools."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from core.exceptions import (
    CompanyNotFoundError,
    ExtractionError,
    FilingNotFoundError,
)
from tools.sec_filings import get_company_facts, compare_companies


class FetcherAgent(BaseAgent):
    """Execute data-fetching steps defined by the planner."""

    def __init__(self) -> None:
        super().__init__("fetcher")

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan", {})
        steps = plan.get("steps", [])
        data: dict[str, Any] = {}

        for step in steps:
            action = step.get("action", "")
            params = step.get("params", {})
            self.logger.info("fetching_step", action=action, params=params)

            try:
                if action == "search_filings":
                    ticker = params.get("ticker", "")
                    form_type = params.get("form_type", "10-K")
                    facts = get_company_facts(ticker, form=form_type)
                    data.setdefault("facts", []).append(facts)

                elif action == "get_filing":
                    ticker = params.get("ticker", "")
                    form_type = params.get("form_type", "10-K")
                    facts = get_company_facts(ticker, form=form_type)
                    data["filing_data"] = facts

                elif action == "compare":
                    tickers = params.get("tickers", [])
                    form_type = params.get("form_type", "10-K")
                    comparisons = compare_companies(tickers, form=form_type)
                    data["comparisons"] = comparisons

                else:
                    self.logger.warning("unknown_action", action=action)

            except CompanyNotFoundError as e:
                self.logger.error("company_not_found", error=str(e))
                data.setdefault("errors", []).append(str(e))
            except FilingNotFoundError as e:
                self.logger.error("filing_not_found", error=str(e))
                data.setdefault("errors", []).append(str(e))
            except ExtractionError as e:
                self.logger.error("extraction_error", error=str(e))
                data.setdefault("errors", []).append(str(e))

        state["data"] = data
        return state

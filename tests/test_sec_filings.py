"""Tests for SEC filings data-fetching functions."""

from __future__ import annotations

import pytest

from tools.sec_filings import get_company_facts, compare_companies, search_financial_concept
from core.exceptions import CompanyNotFoundError, ExtractionError, FilingNotFoundError


class TestSECFilings:
    """Integration tests for SEC EDGAR queries."""

    @pytest.mark.asyncio
    async def test_get_apple_facts(self) -> None:
        """Verify basic fact extraction for Apple."""
        facts = get_company_facts("AAPL")
        assert isinstance(facts, dict)
        assert facts["ticker"] == "AAPL"
        assert "company_name" in facts
        assert "revenue" in facts
        assert "net_income" in facts
        assert "financial_metrics" in facts
        assert isinstance(facts["revenue"], float)
        assert isinstance(facts["financial_metrics"], dict)

    @pytest.mark.asyncio
    async def test_get_10k_facts(self) -> None:
        """Verify extraction with explicit 10-K form."""
        facts = get_company_facts("MSFT", form="10-K")
        assert facts["ticker"] == "MSFT"
        assert "fiscal_year" in facts
        assert "filing_date" in facts

    @pytest.mark.asyncio
    async def test_invalid_ticker(self) -> None:
        """Verify proper error for invalid ticker."""
        with pytest.raises(CompanyNotFoundError):
            get_company_facts("ZZZINVALID")

    @pytest.mark.asyncio
    async def test_facts_have_keys(self) -> None:
        """Verify required keys are present."""
        facts = get_company_facts("AAPL")
        required_keys = {
            "ticker", "company_name", "fiscal_year", "filing_date",
            "revenue", "net_income", "total_assets", "total_liabilities",
            "operating_income", "free_cash_flow", "financial_metrics",
        }
        assert required_keys.issubset(set(facts.keys()))

    @pytest.mark.asyncio
    async def test_compare_companies(self) -> None:
        """Verify multi-company comparison works."""
        results = compare_companies(["AAPL", "MSFT"], form="10-K")
        assert isinstance(results, list)
        assert len(results) >= 1  # at least one should succeed
        for r in results:
            assert "ticker" in r
            assert r["ticker"] in ("AAPL", "MSFT")

    @pytest.mark.asyncio
    async def test_search_financial_concept(self) -> None:
        """Verify concept search returns matching keys."""
        matches = search_financial_concept("AAPL", "revenue")
        assert isinstance(matches, list)
        assert len(matches) >= 1
        assert any("revenue" in m.lower() for m in matches)

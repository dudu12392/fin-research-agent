"""SEC EDGAR data access layer — extraction, comparison, and concept search."""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
from edgar import Company, set_identity
from tenacity import retry, stop_after_attempt, wait_exponential

from core.exceptions import (
    CompanyNotFoundError,
    ExtractionError,
    FilingNotFoundError,
)
from core.logging import get_logger

# ── Identity (must be set at module top) ──────────────────────────
set_identity(os.getenv("SEC_USER_AGENT", "Liang Chang 18845976355@163.com"))

logger = get_logger("tools.sec_filings")


# ═══════════════════════════════════════════════════════════════════
# 1. get_company_facts
# ═══════════════════════════════════════════════════════════════════

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def get_company_facts(ticker: str, form: str = "10-K") -> dict[str, Any]:
    """Extract key financial facts from a company's latest SEC filing.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT').
        form: Filing form type (default '10-K').

    Returns:
        Dictionary with keys:
            ticker, company_name, fiscal_year, filing_date,
            revenue, net_income, total_assets, total_liabilities,
            operating_income, free_cash_flow, financial_metrics.
    """
    start_time = time.perf_counter()
    ticker_upper = ticker.upper()

    logger.info(
        "extraction_start",
        ticker=ticker_upper,
        form=form,
    )

    # ── Resolve company ────────────────────────────────────────
    try:
        company = Company(ticker_upper)
    except Exception as e:
        logger.error("company_not_found", ticker=ticker_upper, error=str(e))
        raise CompanyNotFoundError(f"Company not found: {ticker_upper}") from e

    company_name = _get_company_name(company, ticker_upper)

    # ── Get latest filing ──────────────────────────────────────
    try:
        filings = company.get_filings(form=form)
        filing = filings.latest(1)
    except Exception as e:
        logger.error("filing_not_found", ticker=ticker_upper, form=form, error=str(e))
        raise FilingNotFoundError(
            f"No {form} filing found for {ticker_upper}"
        ) from e

    # ── Unwrap to TenK / TenQ object ───────────────────────────
    try:
        tenk = filing.obj()
    except Exception as e:
        logger.error("obj_unwrap_failed", ticker=ticker_upper, error=str(e))
        raise ExtractionError(
            f"Failed to parse {form} object for {ticker_upper}"
        ) from e

    financials = tenk.financials

    # ── Get built-in metrics dict ──────────────────────────────
    try:
        metrics = financials.get_financial_metrics()
    except Exception as e:
        logger.error("metrics_extraction_failed", ticker=ticker_upper, error=str(e))
        raise ExtractionError(
            f"Failed to extract financial metrics for {ticker_upper}: {e}"
        ) from e

    if not isinstance(metrics, dict):
        raise ExtractionError(f"Unexpected metrics type for {ticker_upper}: {type(metrics)}")

    # ── Extract top-level fields ───────────────────────────────
    revenue = float(metrics.get("revenue", 0.0))
    net_income = float(metrics.get("net_income", 0.0))
    total_assets = float(metrics.get("total_assets", 0.0))
    total_liabilities = float(metrics.get("total_liabilities", 0.0))
    operating_income = float(metrics.get("operating_income", 0.0))

    # ── Cash flow: try getter, fallback to statement DataFrame ─
    operating_cash_flow = _get_operating_cash_flow(financials)
    capital_expenditures = float(metrics.get("capital_expenditures", 0.0))
    free_cash_flow = operating_cash_flow - capital_expenditures

    # ── Fiscal year & filing date ──────────────────────────────
    fiscal_year = _extract_fiscal_year(tenk)
    filing_date = str(getattr(tenk, "filing_date", ""))

    # ── Enrich metrics with derived ratios ─────────────────────
    _enrich_metrics(metrics, revenue, net_income, total_assets, total_liabilities,
                    operating_cash_flow, capital_expenditures)

    elapsed = round(time.perf_counter() - start_time, 3)
    logger.info(
        "extraction_complete",
        ticker=ticker_upper,
        form=form,
        revenue=revenue,
        net_income=net_income,
        elapsed_seconds=elapsed,
    )

    return {
        "ticker": ticker_upper,
        "company_name": company_name,
        "fiscal_year": fiscal_year,
        "filing_date": filing_date,
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "operating_income": operating_income,
        "free_cash_flow": free_cash_flow,
        "financial_metrics": metrics,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. compare_companies
# ═══════════════════════════════════════════════════════════════════

def compare_companies(tickers: list[str], form: str = "10-K") -> list[dict[str, Any]]:
    """Batch-extract facts for multiple companies, skipping failures gracefully.

    Args:
        tickers: List of stock ticker symbols.
        form: Filing form type (default '10-K').

    Returns:
        List of result dicts (same shape as get_company_facts).
        Companies that fail extraction are omitted from the list.
    """
    total = len(tickers)
    logger.info("compare_companies_start", tickers=tickers, total=total, form=form)

    results: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for ticker in tickers:
        try:
            facts = get_company_facts(ticker, form=form)
            results.append(facts)
            success_count += 1
        except (CompanyNotFoundError, FilingNotFoundError, ExtractionError) as e:
            logger.warning(
                "company_skipped",
                ticker=ticker.upper(),
                reason=type(e).__name__,
                error=str(e),
            )
            fail_count += 1

    logger.info(
        "compare_companies_done",
        requested=total,
        success=success_count,
        failed=fail_count,
    )
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. search_financial_concept
# ═══════════════════════════════════════════════════════════════════

def search_financial_concept(ticker: str, keyword: str) -> list[str]:
    """Search financial metrics of a company for keys matching a keyword (case-insensitive).

    Args:
        ticker: Stock ticker symbol.
        keyword: Search term (e.g. 'revenue', 'debt', 'margin').

    Returns:
        List of metric key names whose keys contain the keyword.
    """
    logger.info("concept_search", ticker=ticker.upper(), keyword=keyword)

    facts = get_company_facts(ticker)
    metrics: dict[str, Any] = facts.get("financial_metrics", {})
    keyword_lower = keyword.lower()

    matches = [key for key in metrics if keyword_lower in key.lower()]
    matches.sort()

    logger.info(
        "concept_search_done",
        ticker=ticker.upper(),
        keyword=keyword,
        match_count=len(matches),
    )
    return matches


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _get_company_name(company: Any, ticker: str) -> str:
    """Extract company name from the Company object."""
    for attr in ("name", "company_name", "title"):
        val = getattr(company, attr, None)
        if val and isinstance(val, str) and val.strip() and val.strip() != ticker:
            return str(val).strip()
    return ticker


def _get_operating_cash_flow(financials: Any) -> float:
    """Get operating cash flow — try getter first, fallback to cash flow statement."""
    # Try direct getter
    try:
        val = financials.get_operating_cash_flow()
        if val is not None:
            return float(val)
    except Exception:
        pass

    # Fallback: parse cash flow statement DataFrame
    try:
        cfs = financials.cash_flow_statement()
        df: pd.DataFrame = cfs.to_dataframe()
        # Find the OCF row by concept
        target = "us-gaap_NetCashProvidedByUsedInOperatingActivities"
        row = df[df["concept"] == target]
        if not row.empty:
            # Pick the first value column (latest fiscal year)
            value_cols = [c for c in df.columns if c not in (
                "concept", "label", "standard_concept", "level", "abstract",
                "dimension", "is_breakdown", "dimension_axis", "dimension_member",
                "dimension_member_label", "dimension_label", "balance", "weight",
                "preferred_sign", "parent_concept", "parent_abstract_concept",
            )]
            for col in value_cols:
                val = row.iloc[0][col]
                if pd.notna(val):
                    return float(val)
    except Exception:
        pass

    return 0.0


def _extract_fiscal_year(tenk: Any) -> int:
    """Extract fiscal year from the filing object."""
    for attr in ("fiscal_year", "year", "period_end_year"):
        val = getattr(tenk, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    filing_date = getattr(tenk, "filing_date", None)
    if filing_date is not None:
        try:
            return int(str(filing_date)[:4])
        except (TypeError, ValueError):
            pass

    return 0


def _enrich_metrics(
    metrics: dict[str, Any],
    revenue: float,
    net_income: float,
    total_assets: float,
    total_liabilities: float,
    operating_cash_flow: float,
    capital_expenditures: float,
) -> None:
    """Add derived financial ratios to the metrics dict."""
    # Gross margin (if we have cost_of_revenue or can infer)
    if "cost_of_revenue" in metrics and revenue > 0:
        metrics["gross_margin"] = round(
            (revenue - float(metrics["cost_of_revenue"])) / revenue * 100, 2
        )

    # Net margin
    if revenue > 0:
        metrics["net_margin"] = round(net_income / revenue * 100, 2)

    # ROA
    if total_assets > 0:
        metrics["roa"] = round(net_income / total_assets * 100, 2)

    # ROE
    equity = float(metrics.get("stockholders_equity", 0.0))
    if equity > 0:
        metrics["roe"] = round(net_income / equity * 100, 2)

    # Debt ratio
    if total_assets > 0:
        metrics["debt_ratio"] = round(total_liabilities / total_assets * 100, 2)

    # Interest coverage (if we had EBIT and interest expense)
    if operating_cash_flow > 0 and capital_expenditures >= 0:
        metrics["free_cash_flow"] = operating_cash_flow - capital_expenditures
    elif "free_cash_flow" not in metrics or metrics.get("free_cash_flow") is None:
        metrics["free_cash_flow"] = operating_cash_flow - capital_expenditures

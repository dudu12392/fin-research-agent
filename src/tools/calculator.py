"""Financial metrics calculator — gross margin, YoY growth, etc."""

from __future__ import annotations


def gross_margin(revenue: float, cost_of_revenue: float) -> float:
    """Calculate gross margin percentage.

    Args:
        revenue: Total revenue.
        cost_of_revenue: Cost of goods sold / cost of revenue.

    Returns:
        Gross margin as a percentage (0-100).
    """
    if revenue == 0:
        return 0.0
    return round((revenue - cost_of_revenue) / revenue * 100, 2)


def net_margin(net_income: float, revenue: float) -> float:
    """Calculate net profit margin percentage."""
    if revenue == 0:
        return 0.0
    return round(net_income / revenue * 100, 2)


def yoy_growth(current: float, previous: float) -> float:
    """Calculate year-over-year growth rate.

    Args:
        current: Current period value.
        previous: Previous period value.

    Returns:
        Growth rate as a percentage (e.g. 15.5 for 15.5% growth).
    """
    if previous == 0:
        return 0.0
    return round((current - previous) / abs(previous) * 100, 2)


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """Calculate current ratio (liquidity metric)."""
    if current_liabilities == 0:
        return float("inf")
    return round(current_assets / current_liabilities, 2)


def debt_to_equity(total_debt: float, total_equity: float) -> float:
    """Calculate debt-to-equity ratio."""
    if total_equity == 0:
        return float("inf")
    return round(total_debt / total_equity, 2)


def roe(net_income: float, total_equity: float) -> float:
    """Calculate Return on Equity (ROE) as percentage."""
    if total_equity == 0:
        return 0.0
    return round(net_income / total_equity * 100, 2)

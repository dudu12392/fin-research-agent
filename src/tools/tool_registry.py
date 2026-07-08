"""Tool registry — Pydantic-typed financial data tools for LLM function calling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tools.sec_filings import get_company_facts
from tools.calculator import gross_margin, net_margin, yoy_growth, roe, current_ratio


# ═══════════════════════════════════════════════════════════════════
# Tool schemas (Pydantic)
# ═══════════════════════════════════════════════════════════════════

class SearchFinancialDataInput(BaseModel):
    company: str = Field(..., description="Stock ticker, e.g. AAPL, MSFT")
    year: int = Field(..., description="Fiscal year, e.g. 2025")
    metric: str = Field(default="summary", description="Metric name: revenue, net_income, gross_margin, etc.")


class CalculateRatioInput(BaseModel):
    numerator: float = Field(..., description="Numerator value")
    denominator: float = Field(..., description="Denominator value")
    name: str = Field(..., description="Ratio name, e.g. gross_margin, roe, current_ratio")


class GetCompanyInfoInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. AAPL")


# ═══════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════

def search_financial_data(company: str, year: int, metric: str = "summary") -> dict[str, Any]:
    """Retrieve financial data for a company from its 10-K filing."""
    try:
        facts = get_company_facts(company)
        if str(facts.get("fiscal_year", "")) != str(year):
            return {"status": "warning", "message": f"Returning {facts.get('fiscal_year')} data (requested {year})", "data": facts}
        return {"status": "ok", "data": facts}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def calculate_ratio(numerator: float, denominator: float, name: str) -> dict[str, Any]:
    """Calculate a financial ratio."""
    if denominator == 0:
        return {"status": "error", "message": "Denominator is zero", "value": None}

    if name == "gross_margin":
        value = round((numerator) / denominator * 100, 2) if numerator else 0
    elif name == "net_margin":
        value = net_margin(numerator, denominator)
    elif name == "roe":
        value = roe(numerator, denominator)
    elif name == "current_ratio":
        value = current_ratio(numerator, denominator)
    else:
        value = round(numerator / denominator, 4)

    return {"status": "ok", "name": name, "value": value, "numerator": numerator, "denominator": denominator}


def get_company_info(ticker: str) -> dict[str, Any]:
    """Get basic company information from SEC filings."""
    try:
        facts = get_company_facts(ticker)
        return {
            "status": "ok",
            "ticker": facts.get("ticker", ticker),
            "company_name": facts.get("company_name", ticker),
            "fiscal_year": facts.get("fiscal_year"),
            "revenue": facts.get("revenue"),
            "net_income": facts.get("net_income"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════
# OpenAI function calling tool definitions
# ═══════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_financial_data",
            "description": "Search a company's SEC 10-K financial data by ticker and fiscal year. Returns revenue, net income, assets, liabilities, and other metrics.",
            "parameters": SearchFinancialDataInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ratio",
            "description": "Calculate a financial ratio given numerator and denominator. Supports gross_margin, net_margin, roe, current_ratio.",
            "parameters": CalculateRatioInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get basic information about a company from its latest SEC filing.",
            "parameters": GetCompanyInfoInput.model_json_schema(),
        },
    },
]

TOOL_MAP = {
    "search_financial_data": search_financial_data,
    "calculate_ratio": calculate_ratio,
    "get_company_info": get_company_info,
}


# ═══════════════════════════════════════════════════════════════════
# Tool selector — LLM-powered tool routing
# ═══════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Routes user queries to the correct tool via LLM function-calling.

    Uses DeepSeek's OpenAI-compatible function calling to select a tool
    and extract structured parameters from natural language.
    """

    def __init__(self) -> None:
        import os
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            temperature=0.0,
        ).bind_tools(TOOL_DEFINITIONS, tool_choice="auto")

        self.tool_map = TOOL_MAP

    def select_tool(self, query: str) -> dict | None:
        """Analyze query and return the best tool + extracted parameters.

        Args:
            query: Natural-language user question.

        Returns:
            {"name": "search_financial_data", "parameters": {...}} or None.
        """
        try:
            response = self.llm.invoke([("human", query)])
            if hasattr(response, "tool_calls") and response.tool_calls:
                call = response.tool_calls[0]
                return {
                    "name": call["name"],
                    "parameters": call["args"],
                }
        except Exception:
            pass
        return None

    def execute(self, query: str) -> dict:
        """Full tool execution: select → invoke → return result."""
        tool_call = self.select_tool(query)
        if not tool_call:
            return {"status": "error", "message": "No tool matched"}

        name = tool_call["name"]
        params = tool_call["parameters"]
        func = self.tool_map.get(name)
        if func:
            return func(**params)
        return {"status": "error", "message": f"Unknown tool: {name}"}

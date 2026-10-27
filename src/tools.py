"""Backward-compatible imports for the Finance Assistant finance tools."""

from data.finance_api import (
    FUNCTION_TOOLS,
    FinanceToolError,
    analyze_portfolio,
    execute_tool_call,
    get_company_overview,
    get_daily_stock_prices,
    get_financial_news,
    get_stock_quote,
)

__all__ = [
    "FUNCTION_TOOLS",
    "FinanceToolError",
    "analyze_portfolio",
    "execute_tool_call",
    "get_company_overview",
    "get_daily_stock_prices",
    "get_financial_news",
    "get_stock_quote",
]

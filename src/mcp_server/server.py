"""Streamable HTTP MCP server exposing Finance Assistant tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.settings import get_settings
from data.finance_api import (
    analyze_portfolio as _analyze_portfolio,
    get_company_overview as _get_company_overview,
    get_daily_stock_prices as _get_daily_stock_prices,
    get_financial_news as _get_financial_news,
    get_stock_quote as _get_stock_quote,
)
from data.knowledge_service import search_knowledge as _search_knowledge
from utils.logging import configure_logging


configure_logging()
logger = logging.getLogger(__name__)
mcp = FastMCP(get_settings().app_name)


@mcp.tool()
def get_stock_quote(symbol: str) -> dict[str, Any]:
    """Get the latest available quote for a stock ticker."""
    return _get_stock_quote(symbol)


@mcp.tool()
def get_company_overview(symbol: str) -> dict[str, Any]:
    """Get company profile, valuation data, and key financial ratios."""
    return _get_company_overview(symbol)


@mcp.tool()
def get_daily_stock_prices(
    symbol: str,
    outputsize: str = "compact",
) -> dict[str, Any]:
    """Get daily historical price and volume data for a stock."""
    return _get_daily_stock_prices(symbol, outputsize)


@mcp.tool()
def get_financial_news(
    tickers: list[str] | None = None,
    topics: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get recent financial news and sentiment, optionally filtered."""
    return _get_financial_news(tickers, topics, time_from, time_to, limit)


@mcp.tool()
def analyze_portfolio(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize current portfolio value and allocation; not investment advice."""
    return _analyze_portfolio(holdings)


@mcp.tool()
def search_knowledge(
    query: str,
    doc_types: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search finance knowledge documents using dense and sparse hybrid retrieval."""
    return _search_knowledge(query, doc_types, limit)


def main() -> None:
    """Run the MCP server over Streamable HTTP for MCP clients."""
    logger.info("Finance Assistant MCP server ready at http://127.0.0.1:8000/mcp")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

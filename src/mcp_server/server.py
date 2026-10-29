"""Streamable HTTP MCP server exposing Finance Assistant tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

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


class PortfolioHoldingInput(BaseModel):
    """Validated shape for a portfolio position passed to the MCP tool."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(
        min_length=1,
        max_length=20,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Bare stock ticker, for example AAPL; do not include a company name or other text.",
    )
    quantity: float = Field(
        gt=0,
        description="Number of shares held; must be greater than zero.",
    )
    average_cost: float | None = Field(
        default=None,
        ge=0,
        description="Optional average purchase price per share.",
    )


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
def analyze_portfolio(holdings: list[PortfolioHoldingInput]) -> dict[str, Any]:
    """Analyze holdings using symbol, quantity, and optional average_cost only."""
    return _analyze_portfolio(
        [holding.model_dump(exclude_none=True) for holding in holdings]
    )


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

"""Alpha Vantage tools for the Finance Assistant's market and news agents."""

from __future__ import annotations

import json
import ssl
import threading
import time
from copy import deepcopy
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from core.errors import FinanceAssistantError
from core.settings import get_settings

_API_URL = "https://www.alphavantage.co/query"
_SETTINGS = get_settings()
_REQUEST_TIMEOUT_SECONDS = _SETTINGS.finance_api_timeout_seconds
_MAX_ATTEMPTS = _SETTINGS.finance_api_max_attempts
_MAX_RETRY_DELAY_SECONDS = _SETTINGS.finance_api_max_retry_delay_seconds
_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


class FinanceToolError(FinanceAssistantError):
    """Raised when an external finance data request fails."""


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def _api_key() -> str:
    api_key = get_settings().alpha_vantage_api_key or ""
    if not api_key:
        raise FinanceToolError(
            "ALPHA_VANTAGE_API_KEY is not set. Add an Alpha Vantage API key "
            "to the environment before calling market-data tools."
        )
    return api_key


def _validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized or len(normalized) > 20:
        raise ValueError("symbol must contain between 1 and 20 characters")
    if not all(character.isalnum() or character in ".-_" for character in normalized):
        raise ValueError("symbol contains unsupported characters")
    return normalized


def _cache_key(
    function_name: str, params: dict[str, str]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return function_name, tuple(sorted(params.items()))


def _cached_response(
    function_name: str, params: dict[str, str]
) -> dict[str, Any] | None:
    key = _cache_key(function_name, params)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            return None
        expiry, response = cached
        if expiry <= now:
            del _CACHE[key]
            return None
        return deepcopy(response)


def _store_response(
    function_name: str,
    params: dict[str, str],
    response: dict[str, Any],
    ttl_seconds: int,
) -> None:
    key = _cache_key(function_name, params)
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic() + ttl_seconds, deepcopy(response))


def _retry_delay(response: HTTPError, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), _MAX_RETRY_DELAY_SECONDS)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                delay = (retry_time - datetime.now(retry_time.tzinfo)).total_seconds()
                return min(max(delay, 0.0), _MAX_RETRY_DELAY_SECONDS)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(2**attempt, _MAX_RETRY_DELAY_SECONDS)


def _request_alpha_vantage(
    function_name: str,
    params: dict[str, str],
    *,
    ttl_seconds: int,
) -> dict[str, Any]:
    cached = _cached_response(function_name, params)
    if cached is not None:
        return cached

    query_params = {
        "function": function_name,
        **params,
        "apikey": _api_key(),
    }
    request = Request(
        f"{_API_URL}?{urlencode(query_params)}",
        headers={"User-Agent": "FinanceAssistance/1.0"},
    )
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urlopen(
                request,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                context=_ssl_context(),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay(exc, attempt))
                    continue
                raise FinanceToolError(
                    f"Alpha Vantage returned HTTP {exc.code} after "
                    f"{_MAX_ATTEMPTS} attempts."
                ) from exc
            raise FinanceToolError(
                f"Alpha Vantage returned HTTP {exc.code}."
            ) from exc
        except (TimeoutError, URLError, ssl.SSLCertVerificationError) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise FinanceToolError(
                    "TLS certificate verification failed for Alpha Vantage. "
                    "Check the system CA certificates or set SSL_CERT_FILE to "
                    "a trusted CA bundle."
                ) from exc
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(min(2**attempt, _MAX_RETRY_DELAY_SECONDS))
                continue
            raise FinanceToolError(
                f"Could not reach Alpha Vantage after {_MAX_ATTEMPTS} attempts: "
                f"{reason}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinanceToolError(
                "Alpha Vantage returned a response that was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise FinanceToolError("Alpha Vantage returned an unexpected response.")
        if any(key in payload for key in ("Error Message", "Information", "Note")):
            message = (
                payload.get("Error Message")
                or payload.get("Information")
                or payload.get("Note")
            )
            raise FinanceToolError(f"Alpha Vantage API error: {message}")

        _store_response(function_name, params, payload, ttl_seconds)
        return payload

    raise FinanceToolError("Alpha Vantage request failed unexpectedly.")


def get_stock_quote(symbol: str) -> dict[str, Any]:
    """Get the latest available quote for a stock or supported ticker."""
    normalized = _validate_symbol(symbol)
    response = _request_alpha_vantage(
        "GLOBAL_QUOTE",
        {"symbol": normalized},
        ttl_seconds=60,
    )
    quote = response.get("Global Quote")
    if not isinstance(quote, dict) or not quote:
        raise FinanceToolError(f"No quote data was returned for {normalized}.")
    return {"symbol": normalized, "quote": quote}


def get_company_overview(symbol: str) -> dict[str, Any]:
    """Get company profile, valuation, and key financial ratios."""
    normalized = _validate_symbol(symbol)
    response = _request_alpha_vantage(
        "OVERVIEW",
        {"symbol": normalized},
        ttl_seconds=6 * 60 * 60,
    )
    if not response or not response.get("Symbol"):
        raise FinanceToolError(f"No company overview was returned for {normalized}.")
    return response


def get_daily_stock_prices(
    symbol: str,
    outputsize: str = "compact",
) -> dict[str, Any]:
    """Get recent or full daily stock price and volume history."""
    normalized = _validate_symbol(symbol)
    if outputsize not in {"compact", "full"}:
        raise ValueError("outputsize must be 'compact' or 'full'")
    return _request_alpha_vantage(
        "TIME_SERIES_DAILY",
        {"symbol": normalized, "outputsize": outputsize},
        ttl_seconds=15 * 60,
    )


def get_financial_news(
    tickers: list[str] | None = None,
    topics: list[str] | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get recent Alpha Vantage financial news and sentiment metadata."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    params: dict[str, str] = {"limit": str(limit), "sort": "LATEST"}
    if tickers:
        params["tickers"] = ",".join(_validate_symbol(ticker) for ticker in tickers)
    if topics:
        normalized_topics = [topic.strip() for topic in topics if topic.strip()]
        if not normalized_topics:
            raise ValueError("topics must contain at least one non-empty topic")
        params["topics"] = ",".join(normalized_topics)
    for name, value in (("time_from", time_from), ("time_to", time_to)):
        if value is not None:
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
            params[name] = value.strip()

    return _request_alpha_vantage(
        "NEWS_SENTIMENT",
        params,
        ttl_seconds=5 * 60,
    )


def analyze_portfolio(
    holdings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize supplied holdings using current quotes; this is not investment advice."""
    if not holdings:
        raise ValueError("holdings must contain at least one position")

    positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    has_cost_basis = False

    for holding in holdings:
        if not isinstance(holding, dict):
            raise TypeError("Each holding must be an object")
        symbol = _validate_symbol(str(holding.get("symbol", "")))
        quantity = holding.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            raise ValueError(f"{symbol} quantity must be numeric")
        if quantity <= 0:
            raise ValueError(f"{symbol} quantity must be greater than zero")

        quote_result = get_stock_quote(symbol)
        quote = quote_result["quote"]
        try:
            current_price = float(quote["05. price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinanceToolError(
                f"The quote for {symbol} did not contain a valid current price."
            ) from exc

        market_value = current_price * float(quantity)
        position: dict[str, Any] = {
            "symbol": symbol,
            "quantity": float(quantity),
            "current_price": current_price,
            "market_value": market_value,
        }
        total_market_value += market_value

        average_cost = holding.get("average_cost")
        if average_cost is not None:
            if isinstance(average_cost, bool) or not isinstance(
                average_cost, (int, float)
            ):
                raise ValueError(f"{symbol} average_cost must be numeric")
            if average_cost < 0:
                raise ValueError(f"{symbol} average_cost cannot be negative")
            cost_basis = float(average_cost) * float(quantity)
            gain_loss = market_value - cost_basis
            position.update(
                {
                    "average_cost": float(average_cost),
                    "cost_basis": cost_basis,
                    "gain_loss": gain_loss,
                    "gain_loss_percent": (
                        gain_loss / cost_basis * 100 if cost_basis else None
                    ),
                }
            )
            total_cost_basis += cost_basis
            has_cost_basis = True
        positions.append(position)

    for position in positions:
        position["allocation_percent"] = (
            position["market_value"] / total_market_value * 100
            if total_market_value
            else 0.0
        )

    result: dict[str, Any] = {
        "positions": positions,
        "total_market_value": total_market_value,
        "disclaimer": (
            "Informational portfolio summary based on the latest available quotes; "
            "not investment advice."
        ),
    }
    if has_cost_basis:
        result["total_cost_basis"] = total_cost_basis
        result["total_gain_loss"] = total_market_value - total_cost_basis
    return result


_SYMBOL_PARAMETER = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Ticker symbol, for example AAPL or MSFT.",
        }
    },
    "required": ["symbol"],
    "additionalProperties": False,
}

FUNCTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_portfolio",
            "description": (
                "Estimate portfolio market value, allocation, and optional gain/loss "
                "using current stock quotes. Informational only, not investment advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "quantity": {"type": "number", "exclusiveMinimum": 0},
                                "average_cost": {
                                    "type": "number",
                                    "minimum": 0,
                                    "description": (
                                        "Optional average purchase price per share."
                                    ),
                                },
                            },
                            "required": ["symbol", "quantity"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["holdings"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get the latest available stock quote for market analysis.",
            "parameters": _SYMBOL_PARAMETER,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": (
                "Get a company's profile, sector, market capitalization, valuation "
                "ratios, and key financial metrics."
            ),
            "parameters": _SYMBOL_PARAMETER,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_prices",
            "description": (
                "Get daily historical prices and volume for market trend analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": _SYMBOL_PARAMETER["properties"]["symbol"],
                    "outputsize": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "description": (
                            "compact returns recent history; full requests the full "
                            "available time series."
                        ),
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_news",
            "description": (
                "Search recent financial news with sentiment metadata for the News "
                "Synthesizer agent. Filter by ticker, topic, or time range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional ticker symbols, such as ['AAPL'].",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional Alpha Vantage news topics.",
                    },
                    "time_from": {
                        "type": "string",
                        "description": "Optional start time in YYYYMMDDTHHMM format.",
                    },
                    "time_to": {
                        "type": "string",
                        "description": "Optional end time in YYYYMMDDTHHMM format.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 50,
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "get_stock_quote": get_stock_quote,
    "get_company_overview": get_company_overview,
    "get_daily_stock_prices": get_daily_stock_prices,
    "get_financial_news": get_financial_news,
    "analyze_portfolio": analyze_portfolio,
}


def execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a model-selected function tool call to its implementation."""
    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ValueError(f"Unknown finance tool: {name}")
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object")
    return function(**arguments)

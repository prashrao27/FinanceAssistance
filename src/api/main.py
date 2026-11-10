"""FastAPI endpoints for chat, portfolio analysis, and market data."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Path as PathParam, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from core.conversation_store import ConversationStore
from core.errors import FinanceAssistantError
from core.settings import get_settings
from data.finance_api import (
    analyze_portfolio,
    get_company_overview,
    get_daily_stock_prices,
    get_stock_quote,
)
from workflow.assistant import answer_question, build_assistant_graph

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=10_000)
    user_id: str = Field(default="local-user", min_length=1, max_length=128)
    thread_id: str = Field(default="default", min_length=1, max_length=128)


class PortfolioHolding(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9._-]+$")
    quantity: float = Field(gt=0, allow_inf_nan=False)
    average_cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PortfolioAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holdings: list[PortfolioHolding] = Field(min_length=1, max_length=50)


def _cors_origins() -> list[str]:
    configured = os.environ.get("FINANCE_ASSISTANT_CORS_ORIGINS")
    if configured is None:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = app.state.store or ConversationStore(
        settings.conversation_db_path
    )
    app.state.graph_lock = asyncio.Lock()
    yield


def create_app(
    *,
    store: ConversationStore | None = None,
    graph: Any | None = None,
    debug: bool | None = None,
) -> FastAPI:
    """Create the UI API; optional service injection keeps endpoint tests isolated."""
    api_debug = get_settings().api_debug if debug is None else debug
    app = FastAPI(
        title="Finance Assistant API",
        description=(
            "Conversational finance assistance, portfolio analysis, and market data."
        ),
        version="1.0.0",
        debug=api_debug,
        lifespan=_lifespan,
    )
    app.state.store = store
    app.state.graph = graph
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def log_debug_requests(request: Request, call_next: Any) -> Any:
        if not app.debug:
            return await call_next(request)
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "API request failed: %s %s",
                request.method,
                request.url.path,
            )
            raise
        logger.info(
            "API request %s %s completed with %d in %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started_at,
        )
        return response

    @app.exception_handler(FinanceAssistantError)
    async def finance_service_error(
        _request: Request,
        exc: FinanceAssistantError,
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    async def get_graph() -> Any:
        if app.state.graph is None:
            async with app.state.graph_lock:
                if app.state.graph is None:
                    app.state.graph = await build_assistant_graph()
        return app.state.graph

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(body: ChatRequest) -> dict[str, str]:
        graph_instance = await get_graph()
        answer = await answer_question(
            body.message,
            user_id=body.user_id,
            thread_id=body.thread_id,
            graph=graph_instance,
            store=app.state.store,
        )
        return {
            "user_id": body.user_id,
            "thread_id": body.thread_id,
            "message": body.message,
            "answer": answer,
        }

    @app.get("/chat/history")
    async def chat_history(
        user_id: str = Query(
            default="local-user",
            min_length=1,
            max_length=128,
            pattern=r".*\S.*",
        ),
        thread_id: str = Query(
            default="default",
            min_length=1,
            max_length=128,
            pattern=r".*\S.*",
        ),
        limit: int | None = Query(default=None, ge=1, le=500),
    ) -> dict[str, Any]:
        turns = await run_in_threadpool(
            app.state.store.get_history,
            user_id,
            thread_id,
            limit=limit,
        )
        return {"user_id": user_id, "thread_id": thread_id, "turns": turns}

    @app.delete("/chat/history")
    async def clear_chat_history(
        user_id: str = Query(
            default="local-user",
            min_length=1,
            max_length=128,
            pattern=r".*\S.*",
        ),
        thread_id: str = Query(
            default="default",
            min_length=1,
            max_length=128,
            pattern=r".*\S.*",
        ),
    ) -> dict[str, Any]:
        deleted = await run_in_threadpool(
            app.state.store.clear_history,
            user_id,
            thread_id,
        )
        return {"user_id": user_id, "thread_id": thread_id, "deleted_turns": deleted}

    @app.post("/portfolio/analysis")
    async def portfolio_analysis(
        body: PortfolioAnalysisRequest,
    ) -> dict[str, Any]:
        holdings = [holding.model_dump(exclude_none=True) for holding in body.holdings]
        result = await run_in_threadpool(analyze_portfolio, holdings)
        return result

    @app.get("/market/quote/{symbol}")
    async def market_quote(
        symbol: str = PathParam(
            min_length=1,
            max_length=20,
            pattern=r"^[A-Za-z0-9._-]+$",
        ),
    ) -> dict[str, Any]:
        return await run_in_threadpool(get_stock_quote, symbol)

    @app.get("/market/overview/{symbol}")
    async def market_overview(
        symbol: str = PathParam(
            min_length=1,
            max_length=20,
            pattern=r"^[A-Za-z0-9._-]+$",
        ),
    ) -> dict[str, Any]:
        return await run_in_threadpool(get_company_overview, symbol)

    @app.get("/market/history/{symbol}")
    async def market_history(
        symbol: str = PathParam(
            min_length=1,
            max_length=20,
            pattern=r"^[A-Za-z0-9._-]+$",
        ),
        outputsize: Literal["compact", "full"] = "compact",
    ) -> dict[str, Any]:
        response = await run_in_threadpool(
            get_daily_stock_prices,
            symbol,
            outputsize,
        )
        series = response.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise FinanceAssistantError(
                f"No daily price history was returned for {symbol.strip().upper()}."
            )
        try:
            prices = [
                {
                    "date": date,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["5. volume"]),
                }
                for date, values in series.items()
                if isinstance(date, str) and isinstance(values, dict)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise FinanceAssistantError(
                f"Daily price history for {symbol.strip().upper()} was incomplete."
            ) from exc
        prices.sort(key=lambda price: price["date"])
        if not prices:
            raise FinanceAssistantError(
                f"No daily price history was returned for {symbol.strip().upper()}."
            )
        return {
            "symbol": symbol.strip().upper(),
            "metadata": response.get("Meta Data", {}),
            "prices": prices,
        }

    return app


app = create_app()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from api.main import create_app
from core.conversation_store import ConversationStore
from data.finance_api import FinanceToolError


class FinanceAssistantApiTests(unittest.IsolatedAsyncioTestCase):
    def test_debug_option_configures_fastapi_debug_mode(self) -> None:
        self.assertTrue(create_app(debug=True).debug)
        self.assertFalse(create_app(debug=False).debug)

    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        store = ConversationStore(
            Path(self.temporary_directory.name) / "conversations.sqlite3"
        )
        self.graph = object()
        self.app = create_app(store=store, graph=self.graph)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.temporary_directory.cleanup()

    async def test_health_and_chat_use_the_workflow(self) -> None:
        self.assertEqual((await self.client.get("/health")).json(), {"status": "ok"})
        with patch(
            "api.main.answer_question",
            new=AsyncMock(return_value="Assistant response"),
        ) as answer_question:
            response = await self.client.post(
                "/chat",
                json={
                    "message": "What is diversification?",
                    "user_id": "user-1",
                    "thread_id": "thread-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Assistant response")
        self.assertEqual(answer_question.await_args.kwargs["graph"], self.graph)
        self.assertEqual(answer_question.await_args.kwargs["user_id"], "user-1")

    async def test_chat_history_can_be_read_and_cleared(self) -> None:
        self.app.state.store.append_turn(
            "user-1", "thread-1", "Question", "Answer"
        )

        response = await self.client.get(
            "/chat/history",
            params={"user_id": "user-1", "thread_id": "thread-1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["turns"][0]["answer"], "Answer")

        response = await self.client.delete(
            "/chat/history",
            params={"user_id": "user-1", "thread_id": "thread-1"},
        )
        self.assertEqual(response.json()["deleted_turns"], 1)

    async def test_portfolio_and_market_endpoints_return_dashboard_data(self) -> None:
        with (
            patch(
                "api.main.analyze_portfolio",
                return_value={
                    "positions": [{"symbol": "AAPL", "allocation_percent": 100}]
                },
            ) as portfolio,
            patch(
                "api.main.get_stock_quote",
                return_value={"symbol": "AAPL", "quote": {"05. price": "100"}},
            ),
            patch(
                "api.main.get_company_overview",
                return_value={"Symbol": "AAPL", "Name": "Apple Inc."},
            ),
            patch(
                "api.main.get_daily_stock_prices",
                return_value={
                    "Meta Data": {},
                    "Time Series (Daily)": {
                        "2025-01-02": {
                            "1. open": "2",
                            "2. high": "3",
                            "3. low": "1",
                            "4. close": "2.5",
                            "5. volume": "20",
                        },
                        "2025-01-01": {
                            "1. open": "1",
                            "2. high": "2",
                            "3. low": "0.5",
                            "4. close": "1.5",
                            "5. volume": "10",
                        },
                    },
                },
            ),
        ):
            portfolio_response = await self.client.post(
                "/portfolio/analysis",
                json={"holdings": [{"symbol": "AAPL", "quantity": 2}]},
            )
            quote_response = await self.client.get("/market/quote/AAPL")
            overview_response = await self.client.get("/market/overview/AAPL")
            history_response = await self.client.get("/market/history/AAPL")

        self.assertEqual(portfolio_response.status_code, 200)
        self.assertEqual(portfolio_response.json()["positions"][0]["symbol"], "AAPL")
        portfolio.assert_called_once_with([{"symbol": "AAPL", "quantity": 2.0}])
        self.assertEqual(quote_response.json()["quote"]["05. price"], "100")
        self.assertEqual(overview_response.json()["Name"], "Apple Inc.")
        history = history_response.json()["prices"]
        self.assertEqual(
            [point["date"] for point in history],
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(history[0]["close"], 1.5)

    async def test_invalid_symbols_and_holdings_are_rejected(self) -> None:
        self.assertEqual(
            (await self.client.get("/market/quote/A%20A")).status_code,
            422,
        )
        self.assertEqual(
            (
                await self.client.post(
                "/portfolio/analysis",
                json={"holdings": [{"symbol": "Apple Inc.", "quantity": 1}]},
                )
            ).status_code,
            422,
        )

    async def test_market_provider_errors_are_reported_as_bad_gateway(self) -> None:
        with patch(
            "api.main.get_stock_quote",
            side_effect=FinanceToolError("Market data provider unavailable."),
        ):
            response = await self.client.get("/market/quote/AAPL")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Market data provider unavailable.",
        )


if __name__ == "__main__":
    unittest.main()

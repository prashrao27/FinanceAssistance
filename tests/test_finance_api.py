from __future__ import annotations

import unittest
from unittest.mock import patch

from data.finance_api import analyze_portfolio
from mcp_server.server import mcp


class AnalyzePortfolioTests(unittest.TestCase):
    def test_rejects_non_ticker_symbol_with_actionable_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "bare stock ticker"):
            analyze_portfolio(
                [{"symbol": "Apple shares in my portfolio", "quantity": 2}]
            )

    def test_analyzes_ticker_and_calculates_allocation(self) -> None:
        with patch(
            "data.finance_api.get_stock_quote",
            return_value={"quote": {"05. price": "100"}},
        ):
            result = analyze_portfolio(
                [
                    {
                        "symbol": "aapl",
                        "quantity": 2,
                        "average_cost": 80,
                    }
                ]
            )

        self.assertEqual(result["positions"][0]["symbol"], "AAPL")
        self.assertEqual(result["positions"][0]["allocation_percent"], 100)
        self.assertEqual(result["positions"][0]["gain_loss"], 40)

    def test_mcp_schema_requires_bare_ticker_and_forbids_extra_fields(self) -> None:
        tool = next(
            tool
            for tool in mcp._tool_manager.list_tools()
            if tool.name == "analyze_portfolio"
        )
        holding_schema = tool.parameters["$defs"]["PortfolioHoldingInput"]

        self.assertEqual(holding_schema["properties"]["symbol"]["maxLength"], 20)
        self.assertEqual(
            holding_schema["properties"]["symbol"]["pattern"],
            r"^[A-Za-z0-9._-]+$",
        )
        self.assertFalse(holding_schema["additionalProperties"])
        self.assertIn("symbol", holding_schema["required"])
        self.assertIn("quantity", holding_schema["required"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agents.router import RouteDecision
from agents.specialists import AgentType, SPECIALISTS
from workflow import assistant


class FakeRouter:
    def __init__(self, agent: AgentType) -> None:
        self.agent = agent

    async def ainvoke(self, messages: list[Any]) -> RouteDecision:
        return RouteDecision(agent=self.agent.value, rationale="test route")


class FakeModel:
    def __init__(self, agent: AgentType) -> None:
        self.agent = agent

    def with_structured_output(self, schema: type[RouteDecision]) -> FakeRouter:
        return FakeRouter(self.agent)

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content="Grounded finance education answer.")


class FakeAgent:
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Specialist tool-backed answer.")]}


class AssistantWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retrieval_calls: list[dict[str, Any]] = []

        async def retrieve(
            query: str,
            doc_types: list[str] | None = None,
            limit: int = 5,
        ) -> str:
            self.retrieval_calls.append(
                {"query": query, "doc_types": doc_types, "limit": limit}
            )
            return '{"results":[{"text":"Retrieved test context"}]}'

        def no_op(value: str = "") -> str:
            return value

        self.tools = [
            StructuredTool.from_function(
                coroutine=retrieve,
                name="search_knowledge",
                description="Search finance knowledge.",
            )
        ]
        self.tools.extend(
            StructuredTool.from_function(
                func=no_op,
                name=tool_name,
                description="Test MCP tool.",
            )
            for tool_name in {
                tool
                for specialist in SPECIALISTS.values()
                for tool in specialist.mcp_tools
            }
        )

    def test_rag_then_routed_specialist(self) -> None:
        async def run() -> dict[str, Any]:
            graph = await assistant.build_assistant_graph(
                model=FakeModel(AgentType.TAX_EDUCATION),
                mcp_tools=self.tools,
            )
            return await graph.ainvoke(
                {
                    "query": "Explain tax-advantaged accounts",
                    "route": "",
                    "retrieved_context": "",
                    "answer": "",
                }
            )

        with patch.object(
            assistant,
            "create_agent",
            return_value=FakeAgent(),
        ) as create_agent:
            result = asyncio.run(run())

        self.assertEqual(result["route"], AgentType.TAX_EDUCATION.value)
        self.assertEqual(result["answer"], "Grounded finance education answer.")
        self.assertEqual(
            self.retrieval_calls,
            [
                {
                    "query": "Explain tax-advantaged accounts",
                    "doc_types": ["TaxEducation"],
                    "limit": 5,
                }
            ],
        )
        create_agent.assert_called()

    def test_specialist_tool_assignments_are_scoped(self) -> None:
        self.assertEqual(
            SPECIALISTS[AgentType.NEWS_SYNTHESIZER].mcp_tools,
            frozenset({"get_financial_news"}),
        )
        self.assertEqual(
            SPECIALISTS[AgentType.MARKET_ANALYSIS].mcp_tools,
            frozenset(
                {
                    "get_stock_quote",
                    "get_company_overview",
                    "get_daily_stock_prices",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()

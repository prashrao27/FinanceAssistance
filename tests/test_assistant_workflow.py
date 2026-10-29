from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
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
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> RouteDecision:
        self.messages = messages
        return RouteDecision(agent=self.agent.value, rationale="test route")


class FakeModel:
    def __init__(self, agent: AgentType) -> None:
        self.agent = agent
        self.router: FakeRouter | None = None
        self.invocations: list[list[Any]] = []

    def with_structured_output(self, schema: type[RouteDecision]) -> FakeRouter:
        self.router = FakeRouter(self.agent)
        return self.router

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(messages)
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
            model = FakeModel(AgentType.TAX_EDUCATION)
            graph = await assistant.build_assistant_graph(
                model=model,
                mcp_tools=self.tools,
            )
            result = await graph.ainvoke(
                {
                    "query": "Explain tax-advantaged accounts",
                    "route": "",
                    "retrieved_context": "",
                    "answer": "",
                    "conversation_history": [
                        {
                            "query": "What is a Roth account?",
                            "answer": "It is a retirement account with tax treatment.",
                        }
                    ],
                    "user_profile": {
                        "learning_level": "beginner",
                        "explanation_preferences": "Use plain language",
                        "financial_goals": ["Save for retirement"],
                    },
                }
            )
            return result, model

        with patch.object(
            assistant,
            "create_agent",
            return_value=FakeAgent(),
        ) as create_agent:
            result, model = asyncio.run(run())

        self.assertEqual(result["route"], AgentType.TAX_EDUCATION.value)
        self.assertEqual(result["answer"], "Grounded finance education answer.")
        self.assertIn("What is a Roth account?", model.router.messages[1].content)
        specialist_messages = model.invocations[-1]
        self.assertEqual(specialist_messages[1].content, "What is a Roth account?")
        self.assertEqual(
            specialist_messages[2].content,
            "It is a retirement account with tax treatment.",
        )
        self.assertIn("Learning level: beginner", specialist_messages[-1].content)
        self.assertIn("Save for retirement", specialist_messages[-1].content)
        self.assertIn("Explain tax-advantaged accounts", specialist_messages[-1].content)
        self.assertEqual(
            self.retrieval_calls[0]["query"].splitlines()[-1],
            "Current question: Explain tax-advantaged accounts",
        )
        self.assertEqual(self.retrieval_calls[0]["doc_types"], ["TaxEducation"])
        self.assertEqual(self.retrieval_calls[0]["limit"], 5)
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

    def test_answer_question_sets_trace_metadata(self) -> None:
        class RecordingGraph:
            config: dict[str, Any] | None = None
            state: dict[str, Any] | None = None

            async def ainvoke(
                self,
                state: dict[str, Any],
                *,
                config: dict[str, Any],
            ) -> dict[str, str]:
                self.state = state
                self.config = config
                return {"answer": "Test answer."}

        with tempfile.TemporaryDirectory() as directory:
            store = assistant.ConversationStore(
                Path(directory) / "assistant.sqlite3"
            )
            store.update_profile(
                "local-user",
                learning_level="beginner",
                explanation_preferences="Use plain language",
                financial_goals=["Save for a home"],
            )
            graph = RecordingGraph()
            answer = asyncio.run(
                assistant.answer_question(
                    "  Explain diversification  ",
                    graph=graph,
                    store=store,
                )
            )
            first_state = graph.state
            saved_turns = store.get_history("local-user", "default")
            store.update_profile(
                "local-user",
                learning_level="intermediate",
            )
            second_answer = asyncio.run(
                assistant.answer_question(
                    "What about tax-advantaged accounts?",
                    graph=graph,
                    store=store,
                )
            )

        self.assertEqual(answer, "Test answer.")
        self.assertEqual(first_state["query"], "Explain diversification")
        self.assertEqual(
            saved_turns[0]["query"],
            "Explain diversification",
        )
        self.assertEqual(
            graph.state["conversation_history"][0]["answer"],
            "Test answer.",
        )
        self.assertEqual(
            graph.state["user_profile"],
            {
                "learning_level": "intermediate",
                "explanation_preferences": "Use plain language",
                "financial_goals": ["Save for a home"],
            },
        )
        self.assertEqual(second_answer, "Test answer.")
        self.assertEqual(graph.config["run_name"], "finance-assistant")
        self.assertEqual(graph.config["tags"], ["finance-assistant"])
        self.assertEqual(
            graph.config["metadata"],
            {"workflow": "route-retrieve-answer"},
        )


if __name__ == "__main__":
    unittest.main()

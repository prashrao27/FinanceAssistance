from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agents.router import RouteDecision
from agents.specialists import AgentType, SPECIALISTS
from workflow import assistant

if not __package__:
    sys.path.insert(0, str(Path(__file__).parent))

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

    def with_structured_output(
        self,
        schema: type[RouteDecision],
        **kwargs: Any,
    ) -> FakeRouter:
        self.router = FakeRouter(self.agent)
        return self.router

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(messages)
        return AIMessage(
            content=(
                "Grounded finance education answer.\n\n"
                "Sources:\n- [S1] model-generated source"
            )
        )


class FakeAgent:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages or [
            AIMessage(content="Specialist tool-backed answer.")
        ]

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": self.messages}


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
            return (
                '{"results":[{"text":"Retrieved test context","metadata":'
                '{"document_title":"Personal Finance Guide",'
                '"source_file":"finance-guide.pdf","page_start":3,'
                '"page_end":4,"section_path":["Saving","Budget"]}}]}'
            )

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
            model = FakeModel(AgentType.FINANCE_QA)
            intent_model = FakeModel(AgentType.TAX_EDUCATION)
            graph = await assistant.build_assistant_graph(
                model=model,
                router_model=intent_model,
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
            return result, model, intent_model

        with patch.object(
            assistant,
            "create_agent",
            return_value=FakeAgent(),
        ) as create_agent:
            result, model, intent_model = asyncio.run(run())

        self.assertEqual(result["route"], AgentType.TAX_EDUCATION.value)
        self.assertTrue(
            result["answer"].startswith("Grounded finance education answer.")
        )
        self.assertIn(
            "Sources:\n- [S1] Personal Finance Guide, PDF: finance-guide.pdf, "
            "pp. 3-4, section: Saving > Budget",
            result["answer"],
        )
        self.assertEqual(result["answer"].count("Sources:"), 1)
        self.assertNotIn("model-generated source", result["answer"])
        self.assertIn(
            "What is a Roth account?",
            intent_model.router.messages[1].content,
        )
        specialist_messages = model.invocations[-1]
        self.assertEqual(specialist_messages[1].content, "What is a Roth account?")
        self.assertEqual(
            specialist_messages[2].content,
            "It is a retirement account with tax treatment.",
        )
        self.assertIn("Learning level: beginner", specialist_messages[-1].content)
        self.assertIn("Save for retirement", specialist_messages[-1].content)
        self.assertIn("Explain tax-advantaged accounts", specialist_messages[-1].content)
        self.assertIn("[S1] Personal Finance Guide", specialist_messages[-1].content)
        self.assertIn("Cite PDF-supported claims", specialist_messages[-1].content)
        self.assertEqual(
            self.retrieval_calls[0]["query"].splitlines()[-1],
            "Explain tax-advantaged accounts",
        )
        self.assertEqual(self.retrieval_calls[0]["doc_types"], ["TaxEducation"])
        self.assertEqual(self.retrieval_calls[0]["limit"], 5)
        create_agent.assert_called()

    def test_response_lists_pdf_and_external_api_sources(self) -> None:
        async def run() -> dict[str, Any]:
            model = FakeModel(AgentType.MARKET_ANALYSIS)
            graph = await assistant.build_assistant_graph(
                model=model,
                router_model=FakeModel(AgentType.MARKET_ANALYSIS),
                mcp_tools=self.tools,
            )
            return await graph.ainvoke(
                {
                    "query": "Get the latest NVDA stock quote",
                    "route": "",
                    "retrieved_context": "",
                    "source_attributions": [],
                    "answer": "",
                    "conversation_history": [],
                    "user_profile": {},
                }
            )

        with patch.object(
            assistant,
            "create_agent",
            return_value=FakeAgent(
                [
                    ToolMessage(
                        content='{"symbol":"NVDA","price":100}',
                        tool_call_id="quote-1",
                        name="get_stock_quote",
                    ),
                    AIMessage(content="NVDA's latest quote is $100."),
                ]
            ),
        ):
            result = asyncio.run(run())

        self.assertIn("PDF: finance-guide.pdf", result["answer"])
        self.assertIn(
            "External API: Alpha Vantage (stock quote)",
            result["answer"],
        )
        self.assertEqual(result["answer"].count("Sources:"), 1)

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

    def test_retrieval_source_filename_falls_back_to_document_type(self) -> None:
        context, sources = assistant._format_retrieval_context(
            {
                "results": [
                    {
                        "text": "Tax rules depend on the tax year.",
                        "metadata": {"DocType": "TaxEducation"},
                    }
                ]
            }
        )

        self.assertIn("PDF: tax_education.pdf", context)
        self.assertEqual(
            sources,
            ["[S1] PDF: tax_education.pdf"],
        )

    def test_retrieval_source_filename_uses_specialist_doc_type_without_metadata(
        self,
    ) -> None:
        context, sources = assistant._format_retrieval_context(
            {
                "results": [
                    {
                        "text": "Tax rules depend on the tax year.",
                    }
                ]
            },
            fallback_doc_type="TaxEducation",
        )

        self.assertIn("PDF: tax_education.pdf", context)
        self.assertEqual(
            sources,
            ["[S1] PDF: tax_education.pdf"],
        )

    def test_retrieval_context_limits_results_and_excerpt_length(self) -> None:
        context, sources = assistant._format_retrieval_context(
            {
                "results": [
                    {
                        "text": f"Relevant excerpt {index}. " + ("x" * 2_000),
                        "metadata": {
                            "source_file": f"source-{index}.pdf",
                            "DocType": "TaxEducation",
                        },
                    }
                    for index in range(6)
                ]
            }
        )

        self.assertEqual(len(sources), assistant._MAX_RETRIEVAL_RESULTS_IN_PROMPT)
        self.assertIn("[S4] PDF: source-3.pdf", context)
        self.assertNotIn("source-4.pdf", context)
        self.assertLessEqual(
            len(context),
            assistant._MAX_RETRIEVAL_RESULTS_IN_PROMPT
            * (assistant._MAX_RETRIEVAL_EXCERPT_CHARS + 100),
        )

    def test_history_prompt_keeps_only_recent_truncated_turns(self) -> None:
        history = [
            {"query": f"question {index}", "answer": f"answer {index}"}
            for index in range(4)
        ]
        history[-1]["answer"] = "a" * (assistant._MAX_HISTORY_MESSAGE_CHARS + 100)

        messages = assistant._history_messages(history)

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].content, "question 2")
        self.assertEqual(messages[-1].content[-3:], "...")
        self.assertLessEqual(
            len(messages[-1].content),
            assistant._MAX_HISTORY_MESSAGE_CHARS,
        )

    def test_retrieval_query_uses_prior_question_only_for_follow_up(self) -> None:
        history = [
            {
                "query": "Explain Roth IRA contributions",
                "answer": "Long answer that should not be sent into retrieval.",
            }
        ]

        focused = assistant._retrieval_query("Explain traditional IRAs", history)
        follow_up = assistant._retrieval_query("How does that compare?", history)

        self.assertEqual(focused, "Explain traditional IRAs")
        self.assertIn("Current question: How does that compare?", follow_up)
        self.assertIn("Related prior question: Explain Roth IRA contributions", follow_up)
        self.assertNotIn("Long answer", follow_up)

    def test_retrieval_formatter_unwraps_mcp_text_blocks(self) -> None:
        context, sources = assistant._format_retrieval_context(
            [
                {
                    "type": "text",
                    "text": (
                        '{"query":"tax account types","results":[{"text":'
                        '"Tax rules depend on the tax year.","metadata":'
                        '{"DocType":"TaxEducation","page_start":5}}]}'
                    ),
                }
            ],
            fallback_doc_type="TaxEducation",
        )

        self.assertIn("Tax rules depend on the tax year.", context)
        self.assertIn("PDF: tax_education.pdf", context)
        self.assertIn("p. 5", context)
        self.assertEqual(
            sources,
            ["[S1] PDF: tax_education.pdf, p. 5"],
        )

    def test_builds_intent_router_with_openrouter_settings(self) -> None:
        settings = SimpleNamespace(
            router_model="~typesafe/jev-latest",
            openrouter_api_key="test-openrouter-key",
            openrouter_base_url="https://openrouter.ai/api/v1",
        )
        router = assistant._build_intent_router_model(settings)
        self.assertIsInstance(router, assistant._JevIntentRouter)
        self.assertEqual(router.model, "~typesafe/jev-latest")
        self.assertEqual(
            router.endpoint,
            "https://openrouter.ai/api/alpha/decisions",
        )

    def test_jev_intent_router_posts_decisions_and_selects_highest_score(self) -> None:
        response = httpx.Response(
            200,
            json={
                "answers": {
                    specialist.name.value: {
                        "type": "noul",
                        "noul": (
                            0.91
                            if specialist.name is AgentType.PORTFOLIO_ANALYSIS
                            else 0.1
                        ),
                    }
                    for specialist in SPECIALISTS.values()
                }
            },
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response

        async def run() -> RouteDecision:
            router = assistant._JevIntentRouter(
                model="~typesafe/jev-latest",
                api_key="test-openrouter-key",
                base_url="https://openrouter.ai",
            )
            with patch.object(
                assistant.httpx,
                "AsyncClient",
                return_value=client,
            ) as client_factory:
                decision = await router.ainvoke(
                    [
                        SystemMessage(content="Route the question."),
                        HumanMessage(content="Analyze my holdings."),
                    ]
                )
            client_factory.assert_called_once_with(timeout=30.0)
            return decision

        decision = asyncio.run(run())
        request = client.post.await_args
        self.assertEqual(decision.agent, AgentType.PORTFOLIO_ANALYSIS.value)
        self.assertEqual(
            request.args[0],
            "https://openrouter.ai/api/alpha/decisions",
        )
        self.assertEqual(
            request.kwargs["json"]["model"],
            "~typesafe/jev-latest",
        )
        self.assertEqual(
            set(request.kwargs["json"]["questions"]),
            {specialist.name.value for specialist in SPECIALISTS.values()},
        )
        self.assertEqual(
            request.kwargs["headers"]["Authorization"],
            "Bearer test-openrouter-key",
        )

    def test_missing_openrouter_key_has_actionable_error(self) -> None:
        settings = SimpleNamespace(
            router_model="~typesafe/jev-latest",
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
        )
        with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
            assistant._build_intent_router_model(settings)

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

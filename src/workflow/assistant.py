"""LangGraph workflow routing questions to MCP-backed specialist agents."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.router import ROUTER_INSTRUCTIONS, RouteDecision
from agents.specialists import AgentType, SPECIALISTS
from core.settings import get_settings
from utils.logging import configure_logging


logger = logging.getLogger(__name__)

_RETRIEVAL_TOOL_NAME = "search_knowledge"


class AssistantState(TypedDict):
    query: str
    route: str
    retrieved_context: str
    answer: str


def _tool_name(tool: BaseTool) -> str:
    """Remove an optional MCP server prefix from an adapter tool name."""
    return tool.name.rsplit("__", maxsplit=1)[-1]


def _format_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _answer_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Specialist agent returned no messages.")
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return _format_tool_result(content)


async def build_assistant_graph(
    *,
    model: Any | None = None,
    mcp_tools: list[BaseTool] | None = None,
) -> Any:
    """Connect to MCP, build the specialist agents, and compile the LangGraph."""
    settings = get_settings()
    if model is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or the process "
                "environment before running the assistant."
            )
        model = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    if mcp_tools is None:
        mcp_client = MultiServerMCPClient(
            {
                "finance": {
                    "transport": "streamable_http",
                    "url": settings.mcp_server_url,
                }
            }
        )
        mcp_tools = await mcp_client.get_tools()

    tools_by_name: dict[str, BaseTool] = {}
    for tool in mcp_tools:
        name = _tool_name(tool)
        if name in tools_by_name:
            raise ValueError(f"Multiple MCP tools are registered as {name!r}")
        tools_by_name[name] = tool

    retrieval_tool = tools_by_name.get(_RETRIEVAL_TOOL_NAME)
    if retrieval_tool is None:
        raise RuntimeError(
            f"MCP server is missing the required {_RETRIEVAL_TOOL_NAME!r} tool."
        )

    router = model.with_structured_output(RouteDecision)
    specialist_nodes: dict[str, Any] = {}
    for specialist in SPECIALISTS.values():
        missing_tools = specialist.mcp_tools - tools_by_name.keys()
        if missing_tools:
            raise RuntimeError(
                f"MCP server is missing tools for {specialist.name.value}: "
                f"{', '.join(sorted(missing_tools))}"
            )
        specialist_tools = [
            tools_by_name[name] for name in sorted(specialist.mcp_tools)
        ]
        if specialist_tools:
            agent = create_agent(
                model,
                specialist_tools,
                system_prompt=specialist.system_prompt,
            )
        else:
            agent = None

        async def run_specialist(
            state: AssistantState,
            *,
            definition: Any = specialist,
            compiled_agent: Any = agent,
        ) -> dict[str, str]:
            contextual_query = (
                "Retrieved knowledge context:\n"
                f"{state['retrieved_context']}\n\n"
                f"User question:\n{state['query']}"
            )
            if compiled_agent is None:
                response = await model.ainvoke(
                    [
                        SystemMessage(content=definition.system_prompt),
                        HumanMessage(content=contextual_query),
                    ]
                )
                answer = response.content
                if not isinstance(answer, str):
                    answer = _format_tool_result(answer)
            else:
                response = await compiled_agent.ainvoke(
                    {"messages": [HumanMessage(content=contextual_query)]}
                )
                answer = _answer_text(response)
            return {"answer": answer}

        specialist_nodes[specialist.name.value] = run_specialist

    async def route_query(state: AssistantState) -> dict[str, str]:
        decision = await router.ainvoke(
            [
                SystemMessage(content=ROUTER_INSTRUCTIONS),
                HumanMessage(content=state["query"]),
            ]
        )
        if isinstance(decision, RouteDecision):
            agent_name = decision.agent
        elif isinstance(decision, dict):
            agent_name = decision["agent"]
        else:
            raise TypeError("Workflow router returned an invalid route decision")
        if agent_name not in SPECIALISTS:
            raise ValueError(f"Workflow router selected unknown agent: {agent_name}")
        logger.info("Routed question to %s", agent_name)
        return {"route": str(agent_name)}

    async def retrieve_context(state: AssistantState) -> dict[str, str]:
        specialist = SPECIALISTS[AgentType(state["route"])]
        result = await retrieval_tool.ainvoke(
            {
                "query": state["query"],
                "doc_types": (
                    list(specialist.doc_types) if specialist.doc_types is not None else None
                ),
                "limit": 5,
            }
        )
        context = _format_tool_result(result)
        logger.info(
            "Retrieved knowledge context for %s (%d characters)",
            specialist.name.value,
            len(context),
        )
        return {"retrieved_context": context}

    graph = StateGraph(AssistantState)
    graph.add_node("router", route_query)
    graph.add_node("retrieve_context", retrieve_context)
    for name, node in specialist_nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieve_context")
    graph.add_conditional_edges(
        "retrieve_context",
        lambda state: state["route"],
        {name: name for name in specialist_nodes},
    )
    for name in specialist_nodes:
        graph.add_edge(name, END)
    return graph.compile()


async def answer_question(query: str, *, graph: Any | None = None) -> str:
    """Route one question through RAG and the selected specialist."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    assistant = graph or await build_assistant_graph()
    state = await assistant.ainvoke(
        {
            "query": query.strip(),
            "route": "",
            "retrieved_context": "",
            "answer": "",
        }
    )
    return state["answer"]


async def main() -> None:
    """Read one question from stdin and print the specialist response."""
    configure_logging()
    query = input("Finance question: ").strip()
    if not query:
        raise ValueError("Enter a non-empty finance question.")
    print(await answer_question(query))


if __name__ == "__main__":
    asyncio.run(main())

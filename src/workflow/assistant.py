"""LangGraph workflow routing questions to MCP-backed specialist agents."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit

import httpx
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from pydantic import BaseModel, Field

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.router import ROUTER_INSTRUCTIONS, RouteDecision
from agents.specialists import AgentType, SPECIALISTS
from core.conversation_store import ConversationStore, UserProfile
from core.settings import get_settings
from utils.logging import configure_logging

logger = logging.getLogger(__name__)

_RETRIEVAL_TOOL_NAME = "search_knowledge"


class _JevAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class _JevResponse(BaseModel):
    answers: dict[str, _JevAnswer]


class AssistantState(TypedDict):
    query: str
    route: str
    retrieved_context: str
    answer: str
    conversation_history: list[dict[str, str]]
    user_profile: dict[str, Any]


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


def _history_messages(history: list[dict[str, str]]) -> list[Any]:
    messages = []
    for turn in history:
        messages.extend(
            [
                HumanMessage(content=turn["query"]),
                AIMessage(content=turn["answer"]),
            ]
        )
    return messages


def _profile_context(profile_data: dict[str, Any]) -> str:
    if not profile_data:
        return ""
    profile = UserProfile(
        learning_level=profile_data.get("learning_level"),
        explanation_preferences=profile_data.get("explanation_preferences"),
        financial_goals=tuple(profile_data.get("financial_goals", ())),
    )
    sections = ["Saved user profile (user-provided context):"]
    if profile.learning_level:
        sections.append(f"- Learning level: {profile.learning_level}")
    if profile.explanation_preferences:
        sections.append(
            f"- Explanation preferences: {profile.explanation_preferences}"
        )
    if profile.financial_goals:
        sections.append(f"- Financial goals: {'; '.join(profile.financial_goals)}")
    return "\n".join(sections) if len(sections) > 1 else ""


def _build_intent_router_model(settings: Any) -> Any:
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add an OpenRouter API key "
            "to .env to use the configured intent-routing model."
        )
    return _JevIntentRouter(
        model=settings.router_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


class _JevIntentRouter:
    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        parsed_url = urlsplit(base_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("OPENROUTER_BASE_URL must be an absolute URL")
        self.endpoint = (
            f"{parsed_url.scheme}://{parsed_url.netloc}/api/alpha/decisions"
        )

    async def ainvoke(self, messages: list[Any]) -> RouteDecision:
        state = {
            "routing_instructions": ROUTER_INSTRUCTIONS,
            "specialists": [
                {"name": name.value, "description": specialist.description}
                for name, specialist in SPECIALISTS.items()
            ],
            "messages": [
                {
                    "role": message.type,
                    "content": (
                        message.content
                        if isinstance(message.content, str)
                        else _format_tool_result(message.content)
                    ),
                }
                for message in messages
            ],
        }
        questions = {
            specialist.name.value: {
                "type": "noul",
                "instructions": (
                    "Estimate whether this specialist is the single best fit for "
                    "the user's current request, considering the routing "
                    "instructions and conversation. "
                    f"Specialist: {specialist.name.value}. "
                    f"Description: {specialist.description}"
                ),
            }
            for specialist in SPECIALISTS.values()
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "state": state,
                    "questions": questions,
                },
            )
        if response.is_error:
            raise RuntimeError(
                "OpenRouter Decisions API returned "
                f"HTTP {response.status_code}: {response.text}"
            )

        result = _JevResponse.model_validate(response.json())
        missing_answers = questions.keys() - result.answers.keys()
        if missing_answers:
            raise RuntimeError(
                "OpenRouter Decisions API did not answer routing questions for: "
                f"{', '.join(sorted(missing_answers))}"
            )
        selected_agent = max(
            questions,
            key=lambda agent: result.answers[agent].noul,
        )
        confidence = result.answers[selected_agent].noul
        return RouteDecision(
            agent=selected_agent,
            rationale=(
                f"Jev selected {selected_agent} with a routing score of "
                f"{confidence:.3f}."
            ),
        )


async def build_assistant_graph(
    *,
    model: Any | None = None,
    router_model: Any | None = None,
    mcp_tools: list[BaseTool] | None = None,
) -> Any:
    """Connect to MCP, build the specialist agents, and compile the LangGraph."""
    settings = get_settings()
    injected_model = model is not None
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

    if router_model is None:
        if injected_model:
            router_model = model
        else:
            router_model = _build_intent_router_model(settings)

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

    router = (
        router_model
        if isinstance(router_model, _JevIntentRouter)
        else router_model.with_structured_output(
            RouteDecision,
            method="json_schema",
            strict=True,
        )
    )
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
            profile_context = _profile_context(state["user_profile"])
            contextual_query = (
                f"{profile_context}\n\n" if profile_context else ""
            ) + (
                "Retrieved knowledge context:\n"
                f"{state['retrieved_context']}\n\n"
                f"User question:\n{state['query']}"
            )
            conversation_messages = _history_messages(
                state["conversation_history"]
            )
            if compiled_agent is None:
                response = await model.ainvoke(
                    [
                        SystemMessage(content=definition.system_prompt),
                        *conversation_messages,
                        HumanMessage(content=contextual_query),
                    ]
                )
                answer = response.content
                if not isinstance(answer, str):
                    answer = _format_tool_result(answer)
            else:
                response = await compiled_agent.ainvoke(
                    {
                        "messages": [
                            *conversation_messages,
                            HumanMessage(content=contextual_query),
                        ]
                    }
                )
                answer = _answer_text(response)
            return {"answer": answer}

        specialist_nodes[specialist.name.value] = run_specialist

    async def route_query(state: AssistantState) -> dict[str, str]:
        decision = await router.ainvoke(
            [
                SystemMessage(content=ROUTER_INSTRUCTIONS),
                *_history_messages(state["conversation_history"]),
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
        recent_context = "\n".join(
            f"Previous question: {turn['query']}\n"
            f"Previous answer: {turn['answer']}"
            for turn in state["conversation_history"][-2:]
        )
        retrieval_query = (
            f"{recent_context}\nCurrent question: {state['query']}"
            if recent_context
            else state["query"]
        )
        retrieval_arguments = {
            "query": retrieval_query,
            "doc_types": (
                list(specialist.doc_types) if specialist.doc_types is not None else None
            ),
            "limit": 5,
        }
        result = await retrieval_tool.ainvoke(retrieval_arguments)
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


async def answer_question(
    query: str,
    *,
    user_id: str = "local-user",
    thread_id: str = "default",
    graph: Any | None = None,
    store: ConversationStore | None = None,
) -> str:
    """Answer in a persistent user conversation and save successful turns."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    normalized_query = query.strip()
    settings = get_settings()
    conversation_store = store or ConversationStore(settings.conversation_db_path)
    history = conversation_store.get_history(
        user_id,
        thread_id,
        limit=settings.conversation_history_turns,
    )
    profile = conversation_store.get_profile(user_id)
    assistant = graph or await build_assistant_graph()
    with tracing_context(
        enabled=settings.langsmith_tracing,
        project_name=settings.langsmith_project,
    ):
        state = await assistant.ainvoke(
            {
                "query": normalized_query,
                "route": "",
                "retrieved_context": "",
                "answer": "",
                "conversation_history": history,
                "user_profile": {
                    "learning_level": profile.learning_level,
                    "explanation_preferences": profile.explanation_preferences,
                    "financial_goals": list(profile.financial_goals),
                },
            },
            config=RunnableConfig(
                run_name="finance-assistant",
                tags=["finance-assistant"],
                metadata={"workflow": "route-retrieve-answer"},
            ),
        )
    answer = state["answer"]
    conversation_store.append_turn(user_id, thread_id, normalized_query, answer)
    return answer


async def main() -> None:
    """Run an interactive CLI with persistent local history and profile commands."""
    configure_logging()
    user_id = input("User ID [local-user]: ").strip() or "local-user"
    thread_id = input("Conversation ID [default]: ").strip() or "default"
    settings = get_settings()
    store = ConversationStore(settings.conversation_db_path)
    graph: Any | None = None

    while True:
        entry = input(
            "Finance question (/profile, /profile set, /history, "
            "/clear-history, /quit): "
        ).strip()
        if entry in {"/quit", "/exit"}:
            return
        if entry == "/profile":
            profile = store.get_profile(user_id)
            print(f"Learning level: {profile.learning_level or '(not set)'}")
            print(
                "Explanation preferences: "
                f"{profile.explanation_preferences or '(not set)'}"
            )
            print(
                "Financial goals: "
                f"{'; '.join(profile.financial_goals) or '(not set)'}"
            )
            continue
        if entry == "/profile set":
            profile = store.get_profile(user_id)
            learning_level = input(
                f"Learning level [{profile.learning_level or 'not set'}] "
                "(blank keeps, 'clear' removes): "
            ).strip()
            preferences = input(
                f"Explanation preferences "
                f"[{profile.explanation_preferences or 'not set'}] "
                "(blank keeps, 'clear' removes): "
            ).strip()
            goals = input(
                f"Financial goals [{'; '.join(profile.financial_goals) or 'not set'}] "
                "(comma-separated, blank keeps, 'clear' removes): "
            ).strip()
            normalized_goals = (
                []
                if goals.lower() == "clear"
                else [goal.strip() for goal in goals.split(",")]
                if goals
                else None
            )
            store.update_profile(
                user_id,
                learning_level=(
                    "" if learning_level.lower() == "clear" else learning_level or None
                ),
                explanation_preferences=(
                    "" if preferences.lower() == "clear" else preferences or None
                ),
                financial_goals=normalized_goals,
            )
            print("Profile saved.")
            continue
        if entry == "/profile delete":
            if input("Delete the saved profile? [y/N]: ").strip().lower() == "y":
                print(
                    "Profile deleted."
                    if store.delete_profile(user_id)
                    else "No profile was saved."
                )
            continue
        if entry == "/history":
            turns = store.get_history(user_id, thread_id)
            if not turns:
                print("No saved turns in this conversation.")
            for turn in turns:
                print(f"\nYou: {turn['query']}\nAssistant: {turn['answer']}")
            continue
        if entry == "/clear-history":
            if input("Delete this conversation's history? [y/N]: ").strip().lower() == "y":
                deleted = store.clear_history(user_id, thread_id)
                print(f"Deleted {deleted} saved turn(s).")
            continue
        if not entry:
            continue
        if graph is None:
            graph = await build_assistant_graph()
        print(
            await answer_question(
                entry,
                user_id=user_id,
                thread_id=thread_id,
                graph=graph,
                store=store,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())

"""Create a routing evaluation dataset and compare Jev with an OpenAI router."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import Client, aevaluate, tracing_context
from pydantic import BaseModel, Field

from agents.specialists import AgentType
from core.settings import get_settings
from workflow.assistant import (
    _build_intent_router_model,
    build_assistant_graph,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "finance-assistant-routing-v1"
DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "finance_router_v1.json"
)
VALID_ROUTES = {specialist.value for specialist in AgentType}


class _AnswerQuality(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str


def load_evaluation_examples(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load and validate the versioned evaluation examples."""
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON list: {path}")
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Evaluation example {index} must be a JSON object")
        if not isinstance(example.get("query"), str) or not example["query"].strip():
            raise ValueError(f"Evaluation example {index} needs a non-empty query")
        if example.get("route") not in VALID_ROUTES:
            raise ValueError(f"Evaluation example {index} has an invalid route")
        if not isinstance(example.get("answer_criteria"), str):
            raise ValueError(
                f"Evaluation example {index} needs answer_criteria text"
            )
    return examples


def create_or_load_dataset(client: Client) -> Any:
    """Create the versioned LangSmith dataset once, or reuse the existing one."""
    existing = next(
        iter(client.list_datasets(dataset_name=DATASET_NAME, limit=1)),
        None,
    )
    if existing is not None:
        return existing

    examples = load_evaluation_examples()
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Synthetic finance questions with expected specialist routes and "
            "answer-quality criteria. Version 1."
        ),
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"query": example["query"]},
                "outputs": {
                    "route": example["route"],
                    "answer_criteria": example["answer_criteria"],
                },
                "metadata": {"source": "curated-synthetic", "version": 1},
            }
            for example in examples
        ],
    )
    return dataset


def route_accuracy(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the selected specialist matches the expected route."""
    expected = reference_outputs.get("route")
    actual = outputs.get("route")
    return {
        "key": "route_accuracy",
        "score": float(actual == expected),
        "comment": (
            f"query={inputs.get('query', '')}; expected={expected}; actual={actual}"
        ),
    }


def answer_present(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Flag pipeline failures that return no useful answer text."""
    answer = outputs.get("answer")
    is_present = isinstance(answer, str) and bool(answer.strip())
    return {
        "key": "answer_present",
        "score": float(is_present),
        "comment": (
            f"query={inputs.get('query', '')}; "
            f"expected_route={reference_outputs.get('route')}; "
            + ("answer is non-empty." if is_present else "answer is empty.")
        ),
    }


def make_answer_quality_evaluator(judge: Any) -> Any:
    """Build an LLM-as-judge evaluator that scores the answer against the rubric."""

    async def answer_quality(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        result = await judge.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are evaluating a finance-education assistant answer. "
                        "Score from 1 to 5 for satisfying the supplied criteria, "
                        "accuracy based on the available output, clarity, and "
                        "appropriate caution. Do not reward invented market facts "
                        "or personalized investment, tax, or legal advice. "
                        "Return a concise reason."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": inputs["query"],
                            "selected_specialist": outputs.get("route"),
                            "answer": outputs.get("answer"),
                            "criteria": reference_outputs["answer_criteria"],
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        return {
            "key": "answer_quality",
            "score": result.score / 5,
            "comment": result.reasoning,
        }

    return answer_quality


def _require(value: str | None, variable: str) -> str:
    if not value:
        raise RuntimeError(f"{variable} is required to run LangSmith experiments.")
    return value


async def run_experiments() -> tuple[Any, Any]:
    """Create/reuse the dataset, then run experiments A and B against it."""
    settings = get_settings()
    langsmith_key = _require(settings.langsmith_api_key, "LANGSMITH_API_KEY")
    openai_key = _require(settings.openai_api_key, "OPENAI_API_KEY")
    _require(settings.openrouter_api_key, "OPENROUTER_API_KEY")

    client = Client(
        api_key=langsmith_key,
        api_url=settings.langsmith_endpoint,
    )
    dataset = create_or_load_dataset(client)
    answer_model = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=openai_key,
    )
    judge = answer_model.with_structured_output(
        _AnswerQuality,
        method="json_schema",
        strict=True,
    )
    mcp_client = MultiServerMCPClient(
        {
            "finance": {
                "transport": "streamable_http",
                "url": settings.mcp_server_url,
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()
    router_a = _build_intent_router_model(settings)
    router_b = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=openai_key,
    )
    graphs = {
        "A-jev-decisions": await build_assistant_graph(
            model=answer_model,
            router_model=router_a,
            mcp_tools=mcp_tools,
        ),
        "B-openai-structured-router": await build_assistant_graph(
            model=answer_model,
            router_model=router_b,
            mcp_tools=mcp_tools,
        ),
    }
    evaluators = [
        route_accuracy,
        answer_present,
        make_answer_quality_evaluator(judge),
    ]

    async def evaluate_variant(name: str, graph: Any) -> Any:
        async def target(inputs: dict[str, Any]) -> dict[str, str]:
            with tracing_context(
                enabled=True,
                client=client,
                project_name=settings.langsmith_project,
                tags=["finance-assistant-evaluation", name],
            ):
                state = await graph.ainvoke(
                    {
                        "query": inputs["query"],
                        "route": "",
                        "retrieved_context": "",
                        "answer": "",
                        "conversation_history": [],
                        "user_profile": {},
                    }
                )
            return {"route": state["route"], "answer": state["answer"]}

        return await aevaluate(
            target,
            data=dataset.name,
            evaluators=evaluators,
            experiment_prefix=name,
            description=(
                "Finance assistant routing A/B evaluation. Both variants use "
                f"the same specialist model ({settings.llm_model}), retrieval, "
                "tools, dataset, and answer-quality evaluator."
            ),
            metadata={
                "router_variant": name,
                "answer_model": settings.llm_model,
                "dataset_version": 1,
            },
            max_concurrency=1,
            client=client,
        )

    results_a = await evaluate_variant("A-jev-decisions", graphs["A-jev-decisions"])
    results_b = await evaluate_variant(
        "B-openai-structured-router",
        graphs["B-openai-structured-router"],
    )
    await results_a.wait()
    await results_b.wait()
    comparison_url = await results_a.get_comparison_url()
    if comparison_url:
        parsed_url = urlsplit(comparison_url)
        query = dict(parse_qsl(parsed_url.query))
        query["selectedSessions"] = (
            f"{results_a.experiment_id},{results_b.experiment_id}"
        )
        comparison_url = urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                urlencode(query),
                parsed_url.fragment,
            )
        )
    print(f"Experiment A: {results_a.experiment_name} - {results_a.url}")
    print(f"Experiment B: {results_b.experiment_name} - {results_b.url}")
    print(f"Comparison: {comparison_url}")
    return results_a, results_b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Jev-vs-OpenAI LangSmith agent evaluation experiments."
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Create or reuse the LangSmith dataset without running experiments.",
    )
    args = parser.parse_args()
    settings = get_settings()
    client = Client(
        api_key=_require(settings.langsmith_api_key, "LANGSMITH_API_KEY"),
        api_url=settings.langsmith_endpoint,
    )
    if args.dataset_only:
        dataset = create_or_load_dataset(client)
        print(f"LangSmith dataset ready: {dataset.name}")
        return
    asyncio.run(run_experiments())


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from pydantic import BaseModel

from agents.specialists import AgentType
from evaluation import langsmith_experiments as evaluation


class LangSmithEvaluationTests(unittest.TestCase):
    def test_curated_dataset_covers_each_specialist(self) -> None:
        examples = evaluation.load_evaluation_examples()
        routes = {example["route"] for example in examples}
        self.assertEqual(routes, {specialist.value for specialist in AgentType})
        self.assertEqual(len(examples), 12)

    def test_invalid_dataset_route_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(
                '[{"query":"Finance question","route":"unknown",'
                '"answer_criteria":"Explain clearly"}]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid route"):
                evaluation.load_evaluation_examples(path)

    def test_create_or_load_dataset_reuses_existing_dataset(self) -> None:
        client = Mock()
        existing = Mock(name="existing dataset")
        client.list_datasets.return_value = iter([existing])

        dataset = evaluation.create_or_load_dataset(client)

        self.assertIs(dataset, existing)
        client.create_dataset.assert_not_called()
        client.create_examples.assert_not_called()

    def test_create_or_load_dataset_uploads_curated_examples(self) -> None:
        client = Mock()
        client.list_datasets.return_value = iter([])
        dataset = Mock()
        dataset.id = "dataset-id"
        client.create_dataset.return_value = dataset

        result = evaluation.create_or_load_dataset(client)

        self.assertIs(result, dataset)
        examples = client.create_examples.call_args.kwargs["examples"]
        self.assertEqual(len(examples), 12)
        self.assertEqual(
            examples[0]["outputs"]["route"],
            AgentType.FINANCE_QA.value,
        )
        self.assertEqual(examples[0]["metadata"]["version"], 1)

    def test_route_accuracy_and_answer_presence_evaluators(self) -> None:
        references = {"route": AgentType.TAX_EDUCATION.value}
        accurate = evaluation.route_accuracy(
            {"query": "Explain a Roth IRA"},
            {"route": AgentType.TAX_EDUCATION.value},
            references,
        )
        inaccurate = evaluation.route_accuracy(
            {"query": "Explain a Roth IRA"},
            {"route": AgentType.FINANCE_QA.value},
            references,
        )
        present = evaluation.answer_present({}, {"answer": "General guidance"}, {})
        missing = evaluation.answer_present({}, {"answer": " "}, {})

        self.assertEqual(accurate["score"], 1.0)
        self.assertEqual(inaccurate["score"], 0.0)
        self.assertEqual(present["score"], 1.0)
        self.assertEqual(missing["score"], 0.0)

    def test_answer_quality_evaluator_uses_reference_rubric(self) -> None:
        class FakeJudgeResult(BaseModel):
            score: int = 4
            reasoning: str = "Clear general explanation."

        class FakeJudge:
            messages: list[Any] | None = None

            async def ainvoke(self, messages: list[Any]) -> FakeJudgeResult:
                self.messages = messages
                return FakeJudgeResult()

        judge = FakeJudge()
        evaluator = evaluation.make_answer_quality_evaluator(judge)
        result = asyncio.run(
            evaluator(
                {"query": "Explain Roth IRAs"},
                {"route": AgentType.TAX_EDUCATION.value, "answer": "General facts."},
                {"answer_criteria": "Explain general tax treatment."},
            )
        )

        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["comment"], "Clear general explanation.")
        self.assertIn("Explain general tax treatment.", judge.messages[-1].content)


if __name__ == "__main__":
    unittest.main()

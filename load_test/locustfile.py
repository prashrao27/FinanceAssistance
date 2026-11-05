"""Locust load test for concurrent Finance Assistant workflow requests."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import gevent
from locust import User, between, events, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workflow.assistant import answer_question, build_assistant_graph

_QUESTION = os.environ.get(
    "LOCUST_WORKFLOW_QUESTION",
    "Explain compound interest in simple terms.",
)


class _WorkflowRunner:
    """Keep the assistant graph and its async clients on one dedicated loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._graph: Any | None = None
        self._ready: Future[None] = Future()
        self._start_lock = threading.Lock()
        self._thread_started = False
        self._thread = threading.Thread(
            target=self._run,
            name="locust-workflow",
            daemon=True,
        )

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._graph = loop.run_until_complete(build_assistant_graph())
        except BaseException as error:
            self._ready.set_exception(error)
            loop.close()
            return

        self._ready.set_result(None)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _wait(self, future: Future[Any]) -> Any:
        return gevent.get_hub().threadpool.spawn(future.result).get()

    def answer(self, query: str, user_id: str, thread_id: str) -> str:
        with self._start_lock:
            if not self._thread_started:
                self._thread.start()
                self._thread_started = True

        self._wait(self._ready)
        if self._loop is None or self._graph is None:
            raise RuntimeError("The workflow runner did not initialize correctly.")

        result = asyncio.run_coroutine_threadsafe(
            answer_question(
                query,
                user_id=user_id,
                thread_id=thread_id,
                graph=self._graph,
            ),
            self._loop,
        )
        return self._wait(result)

    def stop(self) -> None:
        if (
            self._loop is not None
            and self._ready.done()
            and self._ready.exception() is None
        ):
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


_runner: _WorkflowRunner | None = None
_runner_lock = threading.Lock()


def _get_runner() -> _WorkflowRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = _WorkflowRunner()
    return _runner


def _stop_runner(**_: Any) -> None:
    if _runner is not None:
        _runner.stop()


events.quitting.add_listener(_stop_runner)


class FinanceAssistantWorkflowUser(User):
    """A virtual user asking the assistant questions in an isolated thread."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.user_id: str = f"locust-{uuid.uuid4().hex}"
        self.thread_id: str = "load-test"

    @task
    def ask_assistant(self) -> None:
        with self.environment.events.request.measure(
            "WORKFLOW",
            "answer_question",
            context={"user_id": self.user_id},
        ) as request:
            answer = _get_runner().answer(
                _QUESTION,
                user_id=self.user_id,
                thread_id=self.thread_id,
            )
            if not answer.strip():
                raise RuntimeError("The assistant returned an empty answer.")
            request["response_length"] = len(answer.encode("utf-8"))

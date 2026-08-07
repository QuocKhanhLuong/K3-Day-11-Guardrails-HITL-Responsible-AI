"""Minimal OpenAI runtime adapter used by all lab agents.

The assignment keeps Google ADK classes installed for starter/public-test
compatibility, but live model calls go through the OpenAI Responses API.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace
from typing import Awaitable, Callable
from uuid import uuid4

from openai import AsyncOpenAI

InputCheck = Callable[[str], tuple[bool, str | None]]
OutputFilter = Callable[[str], str]


def get_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"


def get_openai_judge_model() -> str:
    return os.environ.get("OPENAI_JUDGE_MODEL", get_openai_model()).strip() or get_openai_model()


@dataclass(frozen=True)
class OpenAIAgent:
    name: str
    instruction: str
    model: str | None = None


class OpenAIRunner:
    """Small runner exposing the same high-level contract used by the lab helpers."""

    def __init__(
        self,
        *,
        agent: OpenAIAgent,
        app_name: str,
        plugins: list | None = None,
        input_check: InputCheck | None = None,
        output_filter: OutputFilter | None = None,
        client: AsyncOpenAI | None = None,
    ):
        self.agent = agent
        self.app_name = app_name
        self.plugins = plugins or []
        self.input_check = input_check
        self.output_filter = output_filter
        self.client = client or AsyncOpenAI()

    async def _run_input_plugins(self, text: str) -> tuple[bool, str | None]:
        if self.input_check is not None:
            blocked, message = self.input_check(text)
            if blocked:
                return True, message

        for plugin in self.plugins:
            evaluate = getattr(plugin, "evaluate", None)
            if callable(evaluate):
                blocked, _reason, message = evaluate(text)
                if blocked:
                    return True, message or "I cannot process that request."
        return False, None

    async def _run_output_plugins(self, text: str) -> str:
        candidate = text
        for plugin in self.plugins:
            evaluate_text = getattr(plugin, "evaluate_text", None)
            if callable(evaluate_text):
                result = await evaluate_text(candidate)
                if result.get("blocked"):
                    return (
                        "I cannot provide that response safely. "
                        "Please rephrase your VinBank banking question."
                    )
                candidate = result.get("text", candidate)
        if self.output_filter is not None:
            candidate = self.output_filter(candidate)
        return candidate

    async def chat(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
        user_id: str = "student",
    ):
        del user_id
        blocked, message = await self._run_input_plugins(user_message)
        session = SimpleNamespace(id=session_id or uuid4().hex)
        if blocked:
            return message or "I cannot process that request.", session

        response = await self.client.responses.create(
            model=self.agent.model or get_openai_model(),
            instructions=self.agent.instruction,
            input=user_message,
            store=False,
        )
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return await self._run_output_plugins(text), session

"""Base class for all four agents.

Handles the cross-cutting concerns so each concrete agent file only
contains its prompt-assembly and output-parsing logic:
  - calling the LLM backend with (fixed system prompt, dynamic context)
  - retrying on transient backend errors
  - repairing/retrying once on malformed JSON output
  - recording timing + cache telemetry onto the PipelineState
"""
from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from core.cache_manager import CacheManager
from core.exceptions import LLMBackendError, MalformedAgentOutputError
from core.llm_backend import LLMBackend
from core.types import AgentTiming, PipelineState

logger = logging.getLogger("platform.agents")

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    name: str = "base_agent"
    system_prompt: str = ""

    def __init__(self, backend: LLMBackend, cache_manager: CacheManager) -> None:
        self.backend = backend
        self.cache_manager = cache_manager

    async def _call(
        self,
        dynamic_context: str,
        state: PipelineState,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        start = time.perf_counter()
        try:
            resp = await self._call_with_retry(dynamic_context, temperature, max_tokens)
        except LLMBackendError as exc:
            state.errors.append(f"{self.name}: backend error: {exc}")
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        timing = AgentTiming(
            agent=self.name,
            latency_ms=latency_ms,
            prompt_tokens=resp.prompt_tokens,
            cached_prompt_tokens=resp.cached_prompt_tokens,
            generated_tokens=resp.completion_tokens,
        )
        state.add_timing(timing)
        self.cache_manager.record(self.name, resp.prompt_tokens, resp.cached_prompt_tokens)
        logger.info(
            "agent=%s session=%s latency_ms=%.1f prompt_tokens=%d cached=%d hit_ratio=%.2f",
            self.name,
            state.session_id,
            latency_ms,
            resp.prompt_tokens,
            resp.cached_prompt_tokens,
            timing.cache_hit_ratio,
        )
        return resp.text

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.3, max=6),
        retry=retry_if_exception_type(LLMBackendError),
    )
    async def _call_with_retry(self, dynamic_context: str, temperature: float, max_tokens: int):
        return await self.backend.complete(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            dynamic_context=dynamic_context,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        # tolerate models that wrap output in ```json fences despite instructions
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    async def _parse_or_repair(self, raw: str, model_cls: type[T], state: PipelineState, dynamic_context: str) -> T:
        try:
            data = self._parse_json(raw)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("agent=%s malformed output, attempting repair: %s", self.name, exc)
            repair_prompt = (
                dynamic_context
                + "\n\nYour previous output was not valid JSON matching the required schema. "
                "Re-emit ONLY the corrected strict JSON object, nothing else.\n"
                f"Previous output was: {raw[:500]}"
            )
            try:
                raw2 = await self._call(repair_prompt, state, temperature=0.0)
                data = self._parse_json(raw2)
                return model_cls.model_validate(data)
            except Exception as exc2:  # noqa: BLE001
                raise MalformedAgentOutputError(self.name, raw, exc2) from exc2

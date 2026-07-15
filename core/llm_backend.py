"""
LLM backend abstraction.

Two implementations:

* MockLLMBackend  - deterministic, in-process, zero external dependencies.
  Used for unit tests, CI, and local development on machines without a
  GPU. It *simulates* LMCache's prefix-cache-hit accounting (see
  core/cache_manager.py) so the rest of the pipeline, and its tests, can
  exercise the exact same "cache_hit_ratio" code paths that production
  does.

* VLLMLMCacheBackend - talks to a real vLLM OpenAI-compatible server
  that has the LMCache KV-cache connector enabled (see
  deploy/lmcache_config.yaml and README.md for the server-side setup).
  vLLM performs prefix-cache matching transparently: as long as the
  bytes/tokens at the start of the prompt are identical to a
  previously-served request, LMCache serves those KV tensors from its
  CPU/disk offload tier instead of recomputing them on the GPU. This
  class's only job is to (a) always place the fixed, shared system
  prompt first in the prompt so it forms a stable prefix, and
  (b) read the cached-token count back out of the response so we can
  report real cache-hit metrics.
"""
from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config.settings import Settings
from core.exceptions import LLMBackendError, LLMRateLimitError

logger = logging.getLogger("platform.llm_backend")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    async def complete(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        dynamic_context: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Run one completion.

        `system_prompt` is the fixed, agent-specific instruction block
        (identical across all sessions -- the cacheable prefix).
        `dynamic_context` is everything session/turn-specific
        (retrieved articles, conversation history, the draft response,
        etc.) that is appended AFTER the system prompt so the shared
        prefix stays intact and matchable.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------
class MockLLMBackend(LLMBackend):
    """A deterministic fake model good enough to drive the full pipeline
    end-to-end (including realistic-shaped JSON per agent) without any
    network or GPU dependency, plus a simulated LMCache prefix cache so
    cache-hit-ratio reporting is exercised honestly in tests."""

    def __init__(self, settings: Settings, fail_rate: float = 0.0) -> None:
        self.settings = settings
        self.fail_rate = fail_rate
        # prefix_hash -> token_count already "computed" (simulates LMCache's
        # store of previously-seen prefixes)
        self._prefix_cache: dict[str, int] = {}

    @staticmethod
    def _approx_tokens(text: str) -> int:
        return max(1, len(text.split()))

    def _prefix_hash(self, agent_name: str, system_prompt: str) -> str:
        # In real LMCache the hash is over token IDs of the shared prefix.
        # Here we hash agent_name + system_prompt text, which is exactly
        # the invariant portion of the prompt across all sessions.
        return hashlib.sha256(f"{agent_name}:{system_prompt}".encode()).hexdigest()

    async def complete(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        dynamic_context: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.perf_counter()

        if self.fail_rate and random.random() < self.fail_rate:
            raise LLMBackendError(f"[mock] simulated transient failure for {agent_name}")

        # simulate network + prefill + decode latency
        await asyncio.sleep(0.01)

        sys_tokens = self._approx_tokens(system_prompt)
        dyn_tokens = self._approx_tokens(dynamic_context)
        prompt_tokens = sys_tokens + dyn_tokens

        key = self._prefix_hash(agent_name, system_prompt)
        cached = self._prefix_cache.get(key, 0)
        if cached == 0:
            # first time this agent's fixed prompt is seen -> full compute,
            # then LMCache would persist it for reuse.
            self._prefix_cache[key] = sys_tokens
        cached_prompt_tokens = min(cached, sys_tokens)

        text = self._fake_output(agent_name, dynamic_context)
        completion_tokens = self._approx_tokens(text)
        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    # -- deterministic fake JSON generators, one per agent -----------------
    def _fake_output(self, agent_name: str, dynamic_context: str) -> str:
        low = dynamic_context.lower()
        if agent_name == "intent_classifier":
            return self._fake_intent(low)
        if agent_name == "knowledge_retriever":
            return self._fake_retrieval(dynamic_context)
        if agent_name == "response_generator":
            return self._fake_generation(dynamic_context)
        if agent_name == "quality_checker":
            return self._fake_quality(dynamic_context)
        if agent_name == "kb_summarizer":
            return self._fake_summary(dynamic_context)
        raise ValueError(f"unknown agent {agent_name}")

    @staticmethod
    def _fake_summary(dynamic_context: str) -> str:
        import re

        text = dynamic_context.split("ARTICLE TEXT:", 1)[-1].strip()
        # strip internal-only annotations, same rule the real summarizer
        # prompt enforces
        text = re.sub(r"\{\{.*?\}\}", "", text).strip()
        sentences = text.split(". ")
        return ". ".join(sentences[:2]).strip().rstrip(".") + "."

    def _fake_intent(self, low: str) -> str:
        if any(w in low for w in ("refund", "invoice", "charge", "bill", "price")):
            intent, needs_kb = "billing_inquiry", True
        elif any(w in low for w in ("error", "crash", "bug", "not working", "broken")):
            intent, needs_kb = "technical_support", True
        elif any(w in low for w in ("password", "email", "close my account", "upgrade", "downgrade")):
            intent, needs_kb = "account_management", True
        elif any(w in low for w in ("track", "shipment", "delivery", "order status")):
            intent, needs_kb = "order_status", True
        elif any(w in low for w in ("angry", "furious", "unacceptable", "terrible")):
            intent, needs_kb = "complaint", True
        elif any(w in low for w in ("hello", "hi there", "hey", "good morning")):
            intent, needs_kb = "general_chat", False
        else:
            intent, needs_kb = "product_information", True

        sentiment = "negative" if any(w in low for w in ("angry", "furious", "unacceptable", "frustrated")) else "neutral"
        urgency = "high" if any(w in low for w in ("urgent", "immediately", "asap", "again")) else "low"
        return json.dumps(
            {
                "intent": intent,
                "sentiment": sentiment,
                "urgency": urgency,
                "confidence": 0.91,
                "requires_knowledge_base": needs_kb,
                "reasoning": f"Matched keywords for {intent}.",
            }
        )

    def _fake_retrieval(self, dynamic_context: str) -> str:
        try:
            payload = json.loads(dynamic_context.split("CANDIDATE_ARTICLES_JSON:", 1)[1])
        except Exception:
            payload = []
        if not payload:
            return json.dumps({"sufficient_context": False, "selected_articles": [], "gaps": "No candidate articles retrieved."})
        selected = [
            {
                "article_id": a["article_id"],
                "title": a["title"],
                "relevant_excerpt": a["text"][:180],
                "relevance_rank": i + 1,
            }
            for i, a in enumerate(payload[:3])
        ]
        return json.dumps({"sufficient_context": True, "selected_articles": selected, "gaps": ""})

    def _fake_generation(self, dynamic_context: str) -> str:
        try:
            block = dynamic_context.split("SELECTED_ARTICLES_JSON:", 1)[1]
            block = block.split("CONVERSATION_HISTORY:", 1)[0]
            articles = json.loads(block)
        except Exception:
            articles = []
        if not articles:
            return json.dumps(
                {
                    "response_text": "I don't have enough verified information to answer that precisely yet, "
                    "so I'm looping in a specialist who can confirm the details with you shortly.",
                    "citations_used": [],
                    "escalate_to_human": True,
                }
            )
        cite_ids = [a["article_id"] for a in articles]
        excerpt = articles[0]["relevant_excerpt"].split(".")[0]
        text = f"{excerpt}. [source: {cite_ids[0]}]"
        if len(cite_ids) > 1:
            text += f" You can also review a related note here. [source: {cite_ids[1]}]"
        return json.dumps({"response_text": text, "citations_used": cite_ids, "escalate_to_human": False})

    def _fake_quality(self, dynamic_context: str) -> str:
        has_citation = "[source:" in dynamic_context
        grounding = 0.9 if has_citation else 0.3
        scores = {
            "grounding": grounding,
            "relevance": 0.88,
            "tone": 0.85,
            "completeness": 0.8,
            "safety": 0.95,
        }
        overall = round(sum(scores.values()) / len(scores), 2)
        if min(scores.values()) < 0.4:
            overall = min(overall, 0.4)
        passed = overall >= 0.75 and scores["grounding"] >= 0.6 and scores["safety"] >= 0.8
        reasons = [] if passed else ["Response contains claims without a verifiable citation."]
        return json.dumps({**scores, "overall_score": overall, "pass": passed, "failure_reasons": reasons})


# --------------------------------------------------------------------------
# Production backend: vLLM server with LMCache connector
# --------------------------------------------------------------------------
class VLLMLMCacheBackend(LLMBackend):
    """Talks to vLLM's OpenAI-compatible `/v1/chat/completions` endpoint.

    The heavy lifting (prefix hashing, CPU/disk KV offload, reuse across
    requests) happens entirely server-side inside vLLM + the LMCache
    connector -- see deploy/lmcache_config.yaml. This client's
    responsibility is narrow but important:

      1. Always send the fixed system prompt as the first message so it
         forms a stable, matchable token prefix.
      2. Keep the per-agent system prompt string byte-for-byte identical
         across calls (config/agent_prompts.py already guarantees this).
      3. Surface `prompt_tokens_details.cached_tokens` (vLLM reports this
         the same way OpenAI's usage extension does) so the pipeline can
         log/alert on real cache-hit ratios in production.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Imported lazily so `pip install openai` isn't required in mock mode.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        retry=retry_if_exception_type(LLMBackendError),
    )
    async def complete(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        dynamic_context: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.settings.vllm_model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": dynamic_context},
                ],
                extra_body={
                    # Stable per-agent cache namespace hint; harmless if the
                    # server ignores it, but lets LMCache-aware deployments
                    # partition prefix caches per agent role explicitly.
                    "cache_salt": agent_name,
                },
            )
        except Exception as exc:  # openai raises various subclasses
            status = getattr(exc, "status_code", None)
            if status == 429:
                raise LLMRateLimitError(str(exc)) from exc
            raise LLMBackendError(str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0

        text = resp.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )


def build_llm_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend_mode.value == "vllm":
        return VLLMLMCacheBackend(settings)
    return MockLLMBackend(settings)

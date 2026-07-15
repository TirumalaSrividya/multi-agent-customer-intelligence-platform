"""Sequential multi-agent orchestrator.

Flow: IntentClassifier -> KnowledgeRetriever -> ResponseGenerator -> QualityChecker

If the quality gate fails, the pipeline regenerates the response (up to
`max_quality_retries` times) before giving up and escalating to a human,
rather than ever shipping a response that failed grounding/safety checks.
"""
from __future__ import annotations

import logging

from agents.intent_classifier import IntentClassifierAgent
from agents.knowledge_retriever import KnowledgeRetrieverAgent
from agents.quality_checker import QualityCheckerAgent
from agents.response_generator import ResponseGeneratorAgent
from config.settings import Settings
from core.exceptions import CapacityExceededError, MalformedAgentOutputError, PlatformError
from core.session_manager import SessionManager
from core.types import PipelineResponse, PipelineState, Role

logger = logging.getLogger("platform.pipeline")

FALLBACK_MESSAGE = (
    "Thanks for your patience -- I want to make sure you get fully accurate information, "
    "so I'm connecting you with a specialist who will follow up shortly."
)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        intent_agent: IntentClassifierAgent,
        retriever_agent: KnowledgeRetrieverAgent,
        generator_agent: ResponseGeneratorAgent,
        quality_agent: QualityCheckerAgent,
        session_manager: SessionManager,
    ) -> None:
        self.settings = settings
        self.intent_agent = intent_agent
        self.retriever_agent = retriever_agent
        self.generator_agent = generator_agent
        self.quality_agent = quality_agent
        self.session_manager = session_manager

    async def run(self, session_id: str, user_message: str) -> PipelineResponse:
        await self.session_manager.get_or_create(session_id)
        history = await self.session_manager.get_history(session_id)

        state = PipelineState(session_id=session_id, user_message=user_message, history=history)

        try:
            await self.intent_agent.run(state)
            await self.retriever_agent.run(state)
            await self._generate_with_quality_gate(state)
        except CapacityExceededError:
            raise
        except (PlatformError, Exception) as exc:  # noqa: BLE001
            logger.exception("pipeline failed for session=%s: %s", session_id, exc)
            state.errors.append(f"pipeline: unhandled error: {exc}")
            state.escalated = True
            if state.generation is None:
                from core.types import GenerationResult

                state.generation = GenerationResult(response_text=FALLBACK_MESSAGE, citations_used=[], escalate_to_human=True)
            if state.quality is None:
                from core.types import QualityResult

                state.quality = QualityResult(
                    grounding=0, relevance=0, tone=0, completeness=0, safety=0,
                    overall_score=0, passed=False, failure_reasons=[str(exc)],
                )

        await self.session_manager.append_turn(session_id, Role.USER, user_message)
        await self.session_manager.append_turn(session_id, Role.ASSISTANT, state.generation.response_text)

        return PipelineResponse(
            session_id=session_id,
            request_id=state.request_id,
            response_text=state.generation.response_text,
            intent=state.intent.intent if state.intent else "unknown",
            escalated=state.escalated or state.generation.escalate_to_human,
            quality_score=state.quality.overall_score if state.quality else 0.0,
            latency_ms=round(state.total_latency_ms, 1),
            cache_hit_ratio=state.overall_cache_hit_ratio,
            citations=state.generation.citations_used,
        )

    async def _generate_with_quality_gate(self, state: PipelineState) -> None:
        threshold = self.settings.quality_score_threshold
        max_retries = self.settings.max_quality_retries

        while True:
            try:
                await self.generator_agent.run(state)
                await self.quality_agent.run(state)
            except MalformedAgentOutputError as exc:
                state.errors.append(str(exc))
                state.escalated = True
                if state.generation is None:
                    from core.types import GenerationResult

                    state.generation = GenerationResult(response_text=FALLBACK_MESSAGE, citations_used=[], escalate_to_human=True)
                return

            assert state.quality is not None
            if state.quality.passed or state.quality.overall_score >= threshold:
                return

            state.quality_attempts += 1
            logger.warning(
                "session=%s quality_attempt=%d score=%.2f below threshold=%.2f reasons=%s",
                state.session_id,
                state.quality_attempts,
                state.quality.overall_score,
                threshold,
                state.quality.failure_reasons,
            )
            if state.quality_attempts >= max_retries:
                state.escalated = True
                state.generation.escalate_to_human = True
                state.generation.response_text = FALLBACK_MESSAGE
                return
            # loop again: regenerate a fresh response and re-check

from __future__ import annotations

from config.agent_prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
from agents.base import BaseAgent
from core.types import IntentResult, PipelineState


class IntentClassifierAgent(BaseAgent):
    name = "intent_classifier"
    system_prompt = INTENT_CLASSIFIER_SYSTEM_PROMPT

    async def run(self, state: PipelineState) -> IntentResult:
        # Dynamic context is deliberately just the user's message -- it is
        # appended AFTER the fixed system prompt so the (much larger, 400
        # token) instruction block stays a clean, cacheable prefix that is
        # identical on every single call to this agent, across all
        # sessions.
        dynamic_context = f"CUSTOMER_MESSAGE:\n{state.user_message}"
        raw = await self._call(dynamic_context, state, temperature=0.0, max_tokens=200)
        result = await self._parse_or_repair(raw, IntentResult, state, dynamic_context)
        state.intent = result
        return result

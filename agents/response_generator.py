from __future__ import annotations

import json

from config.agent_prompts import RESPONSE_GENERATOR_SYSTEM_PROMPT
from agents.base import BaseAgent
from core.types import GenerationResult, PipelineState


class ResponseGeneratorAgent(BaseAgent):
    name = "response_generator"
    system_prompt = RESPONSE_GENERATOR_SYSTEM_PROMPT

    async def run(self, state: PipelineState) -> GenerationResult:
        if state.retrieval is None:
            raise ValueError("response_generator requires knowledge retrieval to run first")

        articles_json = json.dumps([a.model_dump() for a in state.retrieval.selected_articles])
        history_json = json.dumps([t.model_dump(mode="json") for t in state.history])

        # Ordering: fixed system prompt -> selected articles (stable for
        # this turn) -> conversation history (its earlier-turn prefix is
        # identical to last call within the same session) -> newest
        # message last, since it is the one part guaranteed unique to
        # this exact call.
        dynamic_context = (
            f"INTENT: {state.intent.intent if state.intent else 'unknown'}\n"
            f"SENTIMENT: {state.intent.sentiment if state.intent else 'neutral'}\n"
            f"URGENCY: {state.intent.urgency if state.intent else 'low'}\n"
            f"SUFFICIENT_CONTEXT: {state.retrieval.sufficient_context}\n"
            f"SELECTED_ARTICLES_JSON:{articles_json}\n"
            f"CONVERSATION_HISTORY:{history_json}\n\n"
            f"CUSTOMER_MESSAGE:\n{state.user_message}"
        )
        raw = await self._call(dynamic_context, state, temperature=0.3, max_tokens=500)
        result = await self._parse_or_repair(raw, GenerationResult, state, dynamic_context)
        state.generation = result
        return result

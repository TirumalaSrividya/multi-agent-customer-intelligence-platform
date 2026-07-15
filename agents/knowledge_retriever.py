from __future__ import annotations

import json

from config.agent_prompts import KNOWLEDGE_RETRIEVER_SYSTEM_PROMPT
from agents.base import BaseAgent
from core.types import PipelineState, RetrievalResult
from retrieval.vector_store import KnowledgeBase


class KnowledgeRetrieverAgent(BaseAgent):
    name = "knowledge_retriever"
    system_prompt = KNOWLEDGE_RETRIEVER_SYSTEM_PROMPT

    def __init__(self, backend, cache_manager, knowledge_base: KnowledgeBase, top_k: int = 4):
        super().__init__(backend, cache_manager)
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    async def run(self, state: PipelineState) -> RetrievalResult:
        if state.intent is None:
            raise ValueError("knowledge_retriever requires intent classification to run first")

        if not state.intent.requires_knowledge_base:
            result = RetrievalResult(sufficient_context=True, selected_articles=[], gaps="")
            state.retrieval = result
            return result

        candidates = self.knowledge_base.search(state.user_message, top_k=self.top_k)

        # Dynamic context: user request + intent + candidate articles from
        # the vector DB, appended after the fixed instruction block so that
        # block stays a clean cacheable prefix.
        dynamic_context = (
            f"CUSTOMER_REQUEST:\n{state.user_message}\n\n"
            f"CLASSIFIED_INTENT: {state.intent.intent}\n\n"
            f"CANDIDATE_ARTICLES_JSON:{json.dumps(candidates)}"
        )
        raw = await self._call(dynamic_context, state, temperature=0.0, max_tokens=600)
        result = await self._parse_or_repair(raw, RetrievalResult, state, dynamic_context)
        state.retrieval = result
        return result

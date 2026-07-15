from __future__ import annotations

import json

from config.agent_prompts import QUALITY_CHECKER_SYSTEM_PROMPT
from agents.base import BaseAgent
from core.types import PipelineState, QualityResult


class QualityCheckerAgent(BaseAgent):
    name = "quality_checker"
    system_prompt = QUALITY_CHECKER_SYSTEM_PROMPT

    async def run(self, state: PipelineState) -> QualityResult:
        if state.generation is None:
            raise ValueError("quality_checker requires a generated response first")

        articles_json = json.dumps([a.model_dump() for a in (state.retrieval.selected_articles if state.retrieval else [])])
        history_json = json.dumps([t.model_dump(mode="json") for t in state.history])

        dynamic_context = (
            f"SELECTED_ARTICLES_JSON:{articles_json}\n"
            f"CONVERSATION_HISTORY:{history_json}\n"
            f"MODEL_RESPONSE:{state.generation.response_text}\n"
            f"CITATIONS_USED:{json.dumps(state.generation.citations_used)}"
        )
        raw = await self._call(dynamic_context, state, temperature=0.0, max_tokens=300)
        data = self._parse_json_safe(raw, dynamic_context, state)
        result = QualityResult(
            grounding=data["grounding"],
            relevance=data["relevance"],
            tone=data["tone"],
            completeness=data["completeness"],
            safety=data["safety"],
            overall_score=data["overall_score"],
            passed=data["pass"],
            failure_reasons=data.get("failure_reasons", []),
        )
        state.quality = result
        return result

    def _parse_json_safe(self, raw: str, dynamic_context: str, state: PipelineState) -> dict:
        try:
            return self._parse_json(raw)
        except Exception:
            # Quality checker output failing to parse must never crash the
            # pipeline (this is the *last* gate) -- fail safe by treating
            # it as a failed quality check so the pipeline escalates
            # instead of silently shipping an unverified response.
            state.errors.append("quality_checker: malformed output, failing safe (treated as fail)")
            return {
                "grounding": 0.0,
                "relevance": 0.0,
                "tone": 0.0,
                "completeness": 0.0,
                "safety": 0.0,
                "overall_score": 0.0,
                "pass": False,
                "failure_reasons": ["quality checker output could not be parsed"],
            }

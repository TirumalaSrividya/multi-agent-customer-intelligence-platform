"""Typed data contracts shared by every agent and the orchestrator.

Keeping these in one module (instead of ad-hoc dicts) is what lets the
pipeline validate state transitions and lets tests assert on exact
shapes instead of guessing at dict keys.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role
    content: str
    timestamp: float = Field(default_factory=time.time)


class IntentResult(BaseModel):
    intent: str
    sentiment: str
    urgency: str
    confidence: float
    requires_knowledge_base: bool
    reasoning: str = ""


class RetrievedArticle(BaseModel):
    article_id: str
    title: str
    relevant_excerpt: str
    relevance_rank: int


class RetrievalResult(BaseModel):
    sufficient_context: bool
    selected_articles: list[RetrievedArticle] = Field(default_factory=list)
    gaps: str = ""


class GenerationResult(BaseModel):
    response_text: str
    citations_used: list[str] = Field(default_factory=list)
    escalate_to_human: bool = False


class QualityResult(BaseModel):
    grounding: float
    relevance: float
    tone: float
    completeness: float
    safety: float
    overall_score: float
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


class AgentTiming(BaseModel):
    agent: str
    latency_ms: float
    prompt_tokens: int
    cached_prompt_tokens: int
    generated_tokens: int

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return round(self.cached_prompt_tokens / self.prompt_tokens, 4)


class PipelineState(BaseModel):
    """The single object that flows through the sequential agent pipeline.
    Each agent reads what it needs and writes its own section; nothing is
    mutated out from under a previous agent."""

    session_id: str
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_message: str
    history: list[Turn] = Field(default_factory=list)

    intent: IntentResult | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    quality: QualityResult | None = None

    quality_attempts: int = 0
    escalated: bool = False
    timings: list[AgentTiming] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_timing(self, t: AgentTiming) -> None:
        self.timings.append(t)

    @property
    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.timings)

    @property
    def overall_cache_hit_ratio(self) -> float:
        total = sum(t.prompt_tokens for t in self.timings)
        cached = sum(t.cached_prompt_tokens for t in self.timings)
        return round(cached / total, 4) if total else 0.0


class PipelineResponse(BaseModel):
    session_id: str
    request_id: str
    response_text: str
    intent: str
    escalated: bool
    quality_score: float
    latency_ms: float
    cache_hit_ratio: float
    citations: list[str]

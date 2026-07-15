from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    response: str
    intent: str
    escalated: bool
    quality_score: float
    latency_ms: float
    cache_hit_ratio: float
    citations: list[str]


class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    backend_mode: str


class CacheStatsResponse(BaseModel):
    overall_hit_ratio: float
    total_prompt_tokens: int
    total_cached_tokens: int
    calls: int
    per_agent: dict

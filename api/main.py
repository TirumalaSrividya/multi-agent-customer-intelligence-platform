from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agents import (
    IntentClassifierAgent,
    KnowledgeRetrieverAgent,
    QualityCheckerAgent,
    ResponseGeneratorAgent,
)
from api.schemas import CacheStatsResponse, ChatRequest, ChatResponse, HealthResponse
from config.settings import get_settings
from core.cache_manager import CacheManager
from core.exceptions import CapacityExceededError, PlatformError
from core.llm_backend import build_llm_backend
from core.pipeline import Pipeline
from core.session_manager import SessionManager
from retrieval.vector_store import KnowledgeBase

from core.logging_setup import configure_logging

configure_logging(get_settings())
logger = logging.getLogger("platform.api")


class AppState:
    pipeline: Pipeline
    session_manager: SessionManager
    cache_manager: CacheManager
    settings: object


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = build_llm_backend(settings)
    cache_manager = CacheManager(settings)
    session_manager = SessionManager(settings)

    kb = KnowledgeBase()
    try:
        n = await kb.ingest_and_summarize_from_file("retrieval/sample_kb.json", backend)
        logger.info("ingested, summarized, and indexed %d knowledge base articles", n)
    except Exception:
        logger.exception(
            "full ingest+summarize pipeline failed; falling back to ingest-without-summary"
        )
        try:
            kb.ingest_from_file("retrieval/sample_kb.json")
        except Exception:
            logger.exception("failed to ingest default knowledge base; retrieval will return no results")

    intent_agent = IntentClassifierAgent(backend, cache_manager)
    retriever_agent = KnowledgeRetrieverAgent(backend, cache_manager, kb, top_k=settings.kb_top_k)
    generator_agent = ResponseGeneratorAgent(backend, cache_manager)
    quality_agent = QualityCheckerAgent(backend, cache_manager)

    pipeline = Pipeline(
        settings, intent_agent, retriever_agent, generator_agent, quality_agent, session_manager
    )

    app_state.pipeline = pipeline
    app_state.session_manager = session_manager
    app_state.cache_manager = cache_manager
    app_state.settings = settings

    logger.info("platform started, backend_mode=%s", settings.llm_backend_mode)
    yield
    logger.info("platform shutting down")


app = FastAPI(title="Multi-Agent Customer Intelligence Platform", lifespan=lifespan)


@app.exception_handler(CapacityExceededError)
async def capacity_handler(request: Request, exc: CapacityExceededError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PlatformError)
async def platform_error_handler(request: Request, exc: PlatformError):
    logger.exception("unhandled platform error")
    return JSONResponse(status_code=500, content={"detail": "internal platform error"})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = await app_state.pipeline.run(req.session_id, req.message)
    except CapacityExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(
        session_id=result.session_id,
        request_id=result.request_id,
        response=result.response_text,
        intent=result.intent,
        escalated=result.escalated,
        quality_score=result.quality_score,
        latency_ms=result.latency_ms,
        cache_hit_ratio=result.cache_hit_ratio,
        citations=result.citations,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        active_sessions=app_state.session_manager.active_session_count(),
        backend_mode=app_state.settings.llm_backend_mode.value,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(**app_state.cache_manager.stats.snapshot())

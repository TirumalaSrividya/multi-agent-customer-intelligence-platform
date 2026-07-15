from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agents import (
    IntentClassifierAgent,
    KnowledgeRetrieverAgent,
    QualityCheckerAgent,
    ResponseGeneratorAgent,
)
from config.settings import Settings
from core.cache_manager import CacheManager
from core.llm_backend import MockLLMBackend
from core.pipeline import Pipeline
from core.session_manager import SessionManager
from retrieval.vector_store import KnowledgeBase


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_backend_mode="mock", max_concurrent_sessions=10, max_history_turns=4)


@pytest.fixture
def backend(settings) -> MockLLMBackend:
    return MockLLMBackend(settings)


@pytest.fixture
def cache_manager(settings) -> CacheManager:
    return CacheManager(settings)


@pytest.fixture
def knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.ingest_from_file(Path(__file__).resolve().parent.parent / "retrieval" / "sample_kb.json")
    return kb


@pytest.fixture
def pipeline(settings, backend, cache_manager, knowledge_base) -> Pipeline:
    session_manager = SessionManager(settings)
    return Pipeline(
        settings,
        IntentClassifierAgent(backend, cache_manager),
        KnowledgeRetrieverAgent(backend, cache_manager, knowledge_base, top_k=settings.kb_top_k),
        ResponseGeneratorAgent(backend, cache_manager),
        QualityCheckerAgent(backend, cache_manager),
        session_manager,
    )

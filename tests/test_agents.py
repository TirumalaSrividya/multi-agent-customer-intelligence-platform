from __future__ import annotations

import pytest

from core.types import PipelineState, Role, Turn
from agents import IntentClassifierAgent, KnowledgeRetrieverAgent, ResponseGeneratorAgent, QualityCheckerAgent


@pytest.mark.asyncio
async def test_intent_classifier_billing(backend, cache_manager):
    agent = IntentClassifierAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="I was charged twice, I need a refund")
    result = await agent.run(state)
    assert result.intent == "billing_inquiry"
    assert result.requires_knowledge_base is True
    assert state.intent is result


@pytest.mark.asyncio
async def test_intent_classifier_greeting_skips_kb(backend, cache_manager):
    agent = IntentClassifierAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="hey there!")
    result = await agent.run(state)
    assert result.intent == "general_chat"
    assert result.requires_knowledge_base is False


@pytest.mark.asyncio
async def test_knowledge_retriever_finds_relevant_article(backend, cache_manager, knowledge_base):
    intent_agent = IntentClassifierAgent(backend, cache_manager)
    retriever = KnowledgeRetrieverAgent(backend, cache_manager, knowledge_base)
    state = PipelineState(session_id="s1", user_message="how do I reset my password?")
    await intent_agent.run(state)
    result = await retriever.run(state)
    assert result.sufficient_context is True
    assert any(a.article_id == "kb_002" for a in result.selected_articles)


@pytest.mark.asyncio
async def test_knowledge_retriever_skipped_for_general_chat(backend, cache_manager, knowledge_base):
    intent_agent = IntentClassifierAgent(backend, cache_manager)
    retriever = KnowledgeRetrieverAgent(backend, cache_manager, knowledge_base)
    state = PipelineState(session_id="s1", user_message="good morning!")
    await intent_agent.run(state)
    result = await retriever.run(state)
    assert result.selected_articles == []
    assert result.sufficient_context is True


@pytest.mark.asyncio
async def test_response_generator_cites_sources(backend, cache_manager, knowledge_base):
    intent_agent = IntentClassifierAgent(backend, cache_manager)
    retriever = KnowledgeRetrieverAgent(backend, cache_manager, knowledge_base)
    generator = ResponseGeneratorAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="how do I reset my password?")
    await intent_agent.run(state)
    await retriever.run(state)
    result = await generator.run(state)
    assert "[source:" in result.response_text
    assert result.citations_used, "expected at least one citation"


@pytest.mark.asyncio
async def test_response_generator_escalates_without_context(backend, cache_manager):
    from core.types import RetrievalResult

    generator = ResponseGeneratorAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="something obscure")
    state.intent = None
    state.retrieval = RetrievalResult(sufficient_context=False, selected_articles=[], gaps="no match")
    result = await generator.run(state)
    assert result.escalate_to_human is True
    assert result.citations_used == []


@pytest.mark.asyncio
async def test_quality_checker_passes_grounded_response(backend, cache_manager, knowledge_base):
    intent_agent = IntentClassifierAgent(backend, cache_manager)
    retriever = KnowledgeRetrieverAgent(backend, cache_manager, knowledge_base)
    generator = ResponseGeneratorAgent(backend, cache_manager)
    quality = QualityCheckerAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="how do I reset my password?")
    await intent_agent.run(state)
    await retriever.run(state)
    await generator.run(state)
    result = await quality.run(state)
    assert result.overall_score > 0
    assert result.grounding >= 0.6


@pytest.mark.asyncio
async def test_quality_checker_fails_ungrounded_response(backend, cache_manager):
    from core.types import GenerationResult, RetrievalResult

    quality = QualityCheckerAgent(backend, cache_manager)
    state = PipelineState(session_id="s1", user_message="what's your refund policy?")
    state.retrieval = RetrievalResult(sufficient_context=True, selected_articles=[])
    state.generation = GenerationResult(response_text="We always give a 100% refund no matter what.", citations_used=[])
    result = await quality.run(state)
    assert result.passed is False
    assert result.grounding < 0.6

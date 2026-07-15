from __future__ import annotations

import asyncio

import pytest

from core.exceptions import CapacityExceededError


@pytest.mark.asyncio
async def test_end_to_end_single_turn(pipeline):
    result = await pipeline.run("sess-1", "how do I reset my password?")
    assert result.session_id == "sess-1"
    assert result.response_text
    assert result.intent == "account_management"
    assert result.citations
    assert result.quality_score > 0


@pytest.mark.asyncio
async def test_multi_turn_history_persists(pipeline):
    await pipeline.run("sess-2", "hi there!")
    r2 = await pipeline.run("sess-2", "what plans do you offer?")
    history = await pipeline.session_manager.get_history("sess-2")
    assert len(history) == 4  # 2 user + 2 assistant turns
    assert r2.intent in ("product_information", "general_chat")


@pytest.mark.asyncio
async def test_history_truncated_to_max_turns(pipeline, settings):
    session_id = "sess-3"
    for i in range(settings.max_history_turns + 4):
        await pipeline.run(session_id, f"question number {i} about billing charge")
    history = await pipeline.session_manager.get_history(session_id)
    assert len(history) <= settings.max_history_turns


@pytest.mark.asyncio
async def test_no_context_escalates_to_human(pipeline):
    result = await pipeline.run("sess-4", "what is your opinion on quantum flux capacitors")
    # not in KB -> insufficient context -> escalation path
    assert result.escalated is True or result.citations == []


@pytest.mark.asyncio
async def test_capacity_limit_enforced(pipeline, settings):
    for i in range(settings.max_concurrent_sessions):
        await pipeline.run(f"cap-session-{i}", "hello")
    with pytest.raises(CapacityExceededError):
        await pipeline.run("one-too-many", "hello")


@pytest.mark.asyncio
async def test_concurrent_sessions_isolated(pipeline):
    async def one(i: int):
        return await pipeline.run(f"concurrent-{i}", "how do I track my order?")

    results = await asyncio.gather(*[one(i) for i in range(8)])
    assert len(results) == 8
    assert all(r.intent == "order_status" for r in results)
    session_ids = {r.session_id for r in results}
    assert len(session_ids) == 8


@pytest.mark.asyncio
async def test_cache_hit_ratio_improves_across_sessions(pipeline, cache_manager):
    # First call for each agent pays the full fixed-prompt cost; every
    # subsequent session hitting the same agent should show a non-zero
    # cache hit ratio on the shared system-prompt portion.
    await pipeline.run("cache-a", "how do I reset my password?")
    r2 = await pipeline.run("cache-b", "how do I reset my password?")
    assert r2.cache_hit_ratio > 0
    snapshot = cache_manager.stats.snapshot()
    assert snapshot["overall_hit_ratio"] > 0

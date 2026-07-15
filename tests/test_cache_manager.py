from __future__ import annotations

from core.cache_manager import CacheManager


def test_kv_bytes_per_token_matches_formula(settings):
    cm = CacheManager(settings)
    expected = 2 * settings.num_layers * settings.num_kv_heads * settings.head_dim * settings.kv_dtype_bytes
    assert cm.kv_bytes_per_token() == expected


def test_shared_prefix_uses_far_less_memory_than_naive(settings):
    cm = CacheManager(settings)
    naive = cm.estimate_naive_kv_gb(num_sessions=100, num_agents=4)
    shared = cm.estimate_shared_prefix_kv_gb(num_sessions=100, num_agents=4, active_concurrency=8)
    assert shared < naive
    assert naive / shared > 5  # order-of-magnitude reduction expected


def test_shared_prefix_fits_in_8gb_budget(settings):
    cm = CacheManager(settings)
    shared = cm.estimate_shared_prefix_kv_gb(num_sessions=100, num_agents=4, active_concurrency=8)
    assert shared <= 8.0


def test_stats_record_and_snapshot(settings):
    cm = CacheManager(settings)
    cm.record("intent_classifier", prompt_tokens=500, cached_tokens=400)
    cm.record("intent_classifier", prompt_tokens=500, cached_tokens=400)
    cm.record("response_generator", prompt_tokens=1000, cached_tokens=300)
    snap = cm.stats.snapshot()
    assert snap["calls"] == 3
    assert snap["total_prompt_tokens"] == 2000
    assert snap["total_cached_tokens"] == 1100
    assert snap["per_agent"]["intent_classifier"]["calls"] == 2
    assert 0 < snap["overall_hit_ratio"] <= 1

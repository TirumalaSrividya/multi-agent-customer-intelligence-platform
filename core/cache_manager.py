"""
CacheManager centralizes everything related to making LMCache's prefix
caching effective, plus the memory-budget math that justifies the
architecture.

WHY THIS MATTERS (the core challenge in the assignment)
---------------------------------------------------------------------------
Naively: 4 agents x 100 concurrent sessions x multi-turn history, each
request re-sending its full prompt (fixed instructions + growing
conversation), would mean the GPU recomputes attention over the *same*
few hundred tokens of agent instructions millions of times, and holds a
separate KV cache block for every session's full prompt (instructions +
history) at once. That blows through an 8GB GPU KV-cache budget almost
immediately -- see `estimate_naive_kv_gb` below for the actual numbers.

LMCache fixes this in two ways we deliberately exploit:

1. PREFIX SHARING ACROSS SESSIONS: every agent's system prompt is a
   fixed string (config/agent_prompts.py). If we always place that fixed
   text as the *first* tokens of every request for that agent, LMCache
   recognizes the token-ID prefix is identical across all 100+ sessions
   and reuses the already-computed KV tensors instead of recomputing
   them. This turns "4 x fixed-prompt-tokens" into effectively a single
   computation, shared by every session and every turn.

2. CROSS-TIER OFFLOAD FOR PER-SESSION STATE: the part of the prompt that
   *isn't* shareable (conversation history, retrieved articles, the
   draft response) is still session-specific, but LMCache can offload
   the KV blocks for turns that aren't the active one to CPU RAM / local
   disk and stream them back only when that session's next turn arrives,
   instead of permanently pinning every session's full history in GPU
   HBM. That is what makes 100 concurrent multi-turn sessions fit inside
   an 8GB *GPU-resident* budget -- the GPU budget only has to hold the
   working set of the handful of sessions actively being decoded right
   now, not all 100 at once.

This module implements (1) directly (prompt ordering + prefix-hash
bookkeeping used for our own hit-rate telemetry) and documents/configures
(2) via `deploy/lmcache_config.yaml`, which is consumed by the vLLM +
LMCache server process, not by this Python process.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field

from config.settings import Settings

logger = logging.getLogger("platform.cache_manager")


@dataclass
class CacheStats:
    """Aggregate, process-wide (all sessions, all agents) telemetry."""

    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    calls: int = 0
    per_agent: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, agent_name: str, prompt_tokens: int, cached_tokens: int) -> None:
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_cached_tokens += cached_tokens
            self.calls += 1
            bucket = self.per_agent.setdefault(agent_name, {"prompt_tokens": 0, "cached_tokens": 0, "calls": 0})
            bucket["prompt_tokens"] += prompt_tokens
            bucket["cached_tokens"] += cached_tokens
            bucket["calls"] += 1

    @property
    def hit_ratio(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return round(self.total_cached_tokens / self.total_prompt_tokens, 4)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "overall_hit_ratio": self.hit_ratio,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "calls": self.calls,
                "per_agent": {
                    name: {
                        **stats,
                        "hit_ratio": round(stats["cached_tokens"] / stats["prompt_tokens"], 4)
                        if stats["prompt_tokens"]
                        else 0.0,
                    }
                    for name, stats in self.per_agent.items()
                },
            }


class CacheManager:
    """Owns prompt composition rules and process-wide cache telemetry.

    Prompt composition rule (critical for LMCache prefix matching):
    ALWAYS: [fixed system prompt] -> [stable, slow-changing context]
            -> [fast-changing / unique-per-call context, LAST]

    e.g. for the Response Generator: system prompt, then retrieved
    articles (stable for the whole turn), then conversation history
    (grows by one turn at a time -> its *prefix*, i.e. the earlier
    turns, is still identical to the previous call within the same
    session, so even history gets partial reuse turn-over-turn), and
    finally the newest user message last.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stats = CacheStats()

    @staticmethod
    def prefix_fingerprint(*parts: str) -> str:
        """Stable fingerprint of a prompt prefix, used only for our own
        logging/debugging -- production cache matching is done inside
        vLLM/LMCache on actual token IDs, not this hash."""
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def record(self, agent_name: str, prompt_tokens: int, cached_tokens: int) -> None:
        self.stats.record(agent_name, prompt_tokens, cached_tokens)
        ratio = cached_tokens / prompt_tokens if prompt_tokens else 0.0
        logger.debug(
            "cache_call agent=%s prompt_tokens=%d cached_tokens=%d hit_ratio=%.2f",
            agent_name,
            prompt_tokens,
            cached_tokens,
            ratio,
        )

    # ---------------------------------------------------------------
    # Sizing math -- used by scripts/benchmark_cache.py and the README
    # to justify the 8GB budget claim with real numbers, not hand-waving.
    # ---------------------------------------------------------------
    def kv_bytes_per_token(self) -> int:
        s = self.settings
        # KV cache stores one Key and one Value vector per layer per token:
        # 2 (K & V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        return 2 * s.num_layers * s.num_kv_heads * s.head_dim * s.kv_dtype_bytes

    def estimate_naive_kv_gb(
        self,
        *,
        num_sessions: int = 100,
        num_agents: int = 4,
        avg_fixed_prompt_tokens: int = 400,
        avg_history_tokens: int = 800,
    ) -> float:
        """KV memory required WITHOUT any prefix sharing: every session
        holds its own copy of every agent's fixed prompt KV, plus its own
        growing history KV, all resident at once."""
        tokens_per_session = num_agents * (avg_fixed_prompt_tokens + avg_history_tokens)
        total_tokens = tokens_per_session * num_sessions
        total_bytes = total_tokens * self.kv_bytes_per_token()
        return round(total_bytes / (1024**3), 3)

    def estimate_shared_prefix_kv_gb(
        self,
        *,
        num_sessions: int = 100,
        num_agents: int = 4,
        avg_fixed_prompt_tokens: int = 400,
        avg_history_tokens: int = 800,
        active_concurrency: int = 8,
    ) -> float:
        """KV memory required WITH LMCache prefix sharing + offload:
        - the fixed system prompt KV is computed and GPU-resident ONCE per
          agent (shared across all sessions), not once per session.
        - only `active_concurrency` sessions' history KV needs to be
          GPU-resident at any instant; the rest lives in LMCache's
          CPU/disk offload tier and streams in on demand.
        """
        shared_prefix_tokens = num_agents * avg_fixed_prompt_tokens
        active_history_tokens = active_concurrency * num_agents * avg_history_tokens
        total_tokens = shared_prefix_tokens + active_history_tokens
        total_bytes = total_tokens * self.kv_bytes_per_token()
        return round(total_bytes / (1024**3), 3)

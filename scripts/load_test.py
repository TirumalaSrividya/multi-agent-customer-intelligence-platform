"""Simulate 100+ concurrent multi-turn sessions against the in-process
pipeline (mock backend by default) and report latency + cache-hit stats.

Usage:
    python scripts/load_test.py --sessions 120 --turns 3
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import (  # noqa: E402
    IntentClassifierAgent,
    KnowledgeRetrieverAgent,
    QualityCheckerAgent,
    ResponseGeneratorAgent,
)
from config.settings import get_settings  # noqa: E402
from core.cache_manager import CacheManager  # noqa: E402
from core.llm_backend import build_llm_backend  # noqa: E402
from core.pipeline import Pipeline  # noqa: E402
from core.session_manager import SessionManager  # noqa: E402
from retrieval.vector_store import KnowledgeBase  # noqa: E402

SAMPLE_MESSAGES = [
    "How do I reset my password?",
    "I was charged twice this month, can I get a refund?",
    "My integration sync keeps failing with an error.",
    "What's included in the Team plan?",
    "Where is my order, it hasn't arrived yet?",
    "This is unacceptable, I've asked three times already!",
    "Hi there, quick question about SSO setup.",
    "How do I close my account?",
]


async def run_session(pipeline: Pipeline, session_id: str, turns: int) -> list[float]:
    latencies = []
    for _ in range(turns):
        msg = random.choice(SAMPLE_MESSAGES)
        t0 = time.perf_counter()
        await pipeline.run(session_id, msg)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=120)
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    settings.max_concurrent_sessions = max(settings.max_concurrent_sessions, args.sessions)
    backend = build_llm_backend(settings)
    cache_manager = CacheManager(settings)
    session_manager = SessionManager(settings)
    kb = KnowledgeBase()
    kb.ingest_from_file("retrieval/sample_kb.json")

    pipeline = Pipeline(
        settings,
        IntentClassifierAgent(backend, cache_manager),
        KnowledgeRetrieverAgent(backend, cache_manager, kb, top_k=settings.kb_top_k),
        ResponseGeneratorAgent(backend, cache_manager),
        QualityCheckerAgent(backend, cache_manager),
        session_manager,
    )

    start = time.perf_counter()
    results = await asyncio.gather(
        *[run_session(pipeline, f"session_{i}", args.turns) for i in range(args.sessions)]
    )
    total_wall_s = time.perf_counter() - start

    all_latencies = [lat for r in results for lat in r]
    all_latencies.sort()

    def pct(p: float) -> float:
        idx = min(len(all_latencies) - 1, int(len(all_latencies) * p))
        return all_latencies[idx]

    print(f"Sessions: {args.sessions}, turns/session: {args.turns}, total requests: {len(all_latencies)}")
    print(f"Wall clock: {total_wall_s:.2f}s")
    print(f"Latency p50={pct(0.5):.1f}ms p95={pct(0.95):.1f}ms p99={pct(0.99):.1f}ms max={all_latencies[-1]:.1f}ms")
    print("Cache stats:", cache_manager.stats.snapshot())


if __name__ == "__main__":
    asyncio.run(main())

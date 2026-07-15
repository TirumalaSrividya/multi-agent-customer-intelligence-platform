"""Standalone knowledge-base ingestion script.

Runs the full ingest -> summarize -> index pipeline using the shared LLM
backend (mock by default, real vLLM/LMCache in production -- same
LLM_BACKEND_MODE env var as everything else).

Usage:
    python scripts/ingest_kb.py [path/to/articles.json]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from core.llm_backend import build_llm_backend  # noqa: E402
from retrieval.vector_store import KnowledgeBase  # noqa: E402


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "retrieval/sample_kb.json"
    settings = get_settings()
    backend = build_llm_backend(settings)
    kb = KnowledgeBase()

    count = await kb.ingest_and_summarize_from_file(path, backend)
    print(f"Ingested, summarized, and indexed {count} articles from {path}")

    # quick smoke query
    for q in ["how do I get a refund", "my sync is broken"]:
        results = kb.search(q, top_k=2)
        print(f"\nQuery: {q!r}")
        for r in results:
            print(f"  - {r['article_id']} ({r['title']}) score={r.get('score', 0):.3f}")


if __name__ == "__main__":
    asyncio.run(main())

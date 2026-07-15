from __future__ import annotations

from pathlib import Path

import pytest

from retrieval.summarizer import summarize_article, summarize_articles
from retrieval.vector_store import KnowledgeBase


@pytest.mark.asyncio
async def test_summarize_article_strips_internal_notes(backend):
    article = {
        "article_id": "kb_001",
        "title": "Refund Policy",
        "text": "Refunds take 5-7 days. {{internal note: escalate over $500}}",
    }
    summary = await summarize_article(backend, article)
    assert summary
    assert "{{" not in summary


@pytest.mark.asyncio
async def test_summarize_articles_batch(backend):
    articles = [
        {"article_id": "kb_001", "title": "Refund Policy", "text": "Refunds take 5-7 days."},
        {"article_id": "kb_002", "title": "Password Reset", "text": "Go to Settings > Security."},
    ]
    out = await summarize_articles(backend, articles)
    assert len(out) == 2
    assert all("summary" in a and a["summary"] for a in out)


@pytest.mark.asyncio
async def test_knowledge_base_full_ingest_summarize_index_pipeline(backend):
    kb = KnowledgeBase()
    path = Path(__file__).resolve().parent.parent / "retrieval" / "sample_kb.json"
    count = await kb.ingest_and_summarize_from_file(path, backend)
    assert count == 8
    results = kb.search("how do I reset my password", top_k=2)
    assert any(r["article_id"] == "kb_002" for r in results)

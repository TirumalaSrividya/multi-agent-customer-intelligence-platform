"""Knowledge base vector store.

Two implementations behind one interface:

* TfidfVectorStore - pure-Python/NumPy/sklearn TF-IDF cosine similarity.
  No network calls, no GPU, tiny memory footprint. This is the default
  so the whole platform runs and is testable on a laptop / CI runner
  with zero external services and no model downloads.

* ChromaVectorStore - persistent Chroma collection using a real sentence
  embedding model, for production quality semantic retrieval. Swap in
  via KnowledgeBase(store=ChromaVectorStore(...)); see README for setup
  (requires downloading an embedding model, hence not the default in
  this offline-friendly reference implementation).

Both raise KnowledgeBaseError on ingestion/query failure so the calling
agent can degrade gracefully (treat as "no context found") instead of
crashing the pipeline.
"""
from __future__ import annotations

import abc
import json
import logging
from pathlib import Path

from core.exceptions import KnowledgeBaseError

logger = logging.getLogger("platform.vector_store")


class VectorStore(abc.ABC):
    @abc.abstractmethod
    def ingest(self, articles: list[dict]) -> int:
        ...

    @abc.abstractmethod
    def query(self, text: str, top_k: int = 4) -> list[dict]:
        ...


class TfidfVectorStore(VectorStore):
    def __init__(self) -> None:
        self._articles: list[dict] = []
        self._vectorizer = None
        self._matrix = None

    def ingest(self, articles: list[dict]) -> int:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover
            raise KnowledgeBaseError("scikit-learn is required for TfidfVectorStore") from exc

        for a in articles:
            for field in ("article_id", "title", "text"):
                if field not in a:
                    raise KnowledgeBaseError(f"article missing required field '{field}': {a}")
        self._articles = articles
        corpus = [f"{a['title']} {a['text']}" for a in articles]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)
        logger.info("ingested %d articles into TfidfVectorStore", len(articles))
        return len(articles)

    def query(self, text: str, top_k: int = 4) -> list[dict]:
        if self._vectorizer is None or not self._articles:
            raise KnowledgeBaseError("vector store is empty; call ingest() first")
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:  # pragma: no cover
            raise KnowledgeBaseError("scikit-learn is required for TfidfVectorStore") from exc

        query_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            if sims[idx] <= 0:
                continue
            results.append({**self._articles[idx], "score": float(sims[idx])})
        return results


class ChromaVectorStore(VectorStore):
    """Production-grade store backed by Chroma + a sentence-transformer
    embedding model. Not used by default (requires a model download);
    see README 'Switching to production retrieval'."""

    def __init__(self, persist_dir: str, collection_name: str = "kb_articles") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise KnowledgeBaseError("chromadb is required for ChromaVectorStore") from exc
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def ingest(self, articles: list[dict]) -> int:
        try:
            self._collection.upsert(
                ids=[a["article_id"] for a in articles],
                documents=[f"{a['title']} {a['text']}" for a in articles],
                metadatas=[{"title": a["title"]} for a in articles],
            )
        except Exception as exc:  # noqa: BLE001
            raise KnowledgeBaseError(f"chroma ingest failed: {exc}") from exc
        return len(articles)

    def query(self, text: str, top_k: int = 4) -> list[dict]:
        try:
            res = self._collection.query(query_texts=[text], n_results=top_k)
        except Exception as exc:  # noqa: BLE001
            raise KnowledgeBaseError(f"chroma query failed: {exc}") from exc
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({"article_id": i, "title": meta.get("title", ""), "text": doc, "score": 1 - dist})
        return out


class KnowledgeBase:
    """Thin façade the agents/services use, independent of which
    VectorStore implementation is plugged in."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or TfidfVectorStore()

    def ingest_from_file(self, path: str | Path) -> int:
        """Ingest + index only (no summarization). Used for fast/offline
        paths -- e.g. tests -- where a live LLM backend isn't needed."""
        articles = self._load_articles(path)
        return self.store.ingest(articles)

    async def ingest_and_summarize_from_file(self, path: str | Path, backend) -> int:
        """Full pipeline: ingest -> summarize (via the shared LLM backend)
        -> index. Each article's LLM-generated `summary` is indexed
        alongside its raw text (summary weighted first in the searchable
        blob) so retrieval surfaces the concise, factual version, while
        the full original text remains available for exact quoting."""
        from retrieval.summarizer import summarize_articles

        articles = self._load_articles(path)
        summarized = await summarize_articles(backend, articles)
        for a in summarized:
            a["text"] = f"{a['summary']} {a['text']}"
        return self.store.ingest(summarized)

    @staticmethod
    def _load_articles(path: str | Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise KnowledgeBaseError(f"knowledge base file not found: {path}")
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"malformed knowledge base JSON: {exc}") from exc

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        try:
            return self.store.query(query, top_k=top_k)
        except KnowledgeBaseError:
            logger.exception("knowledge base query failed, returning empty result set")
            return []

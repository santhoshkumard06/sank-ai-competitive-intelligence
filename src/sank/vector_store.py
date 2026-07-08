"""Semantic matching: the one piece of this pipeline a keyword search
genuinely cannot replace.

Two implementations behind one interface:

  LocalVectorStore   - TF-IDF + cosine similarity, pure scikit-learn, zero
                        API keys, zero network calls. Real, legitimate,
                        explainable information-retrieval technique (the
                        same family used in classic search engines) — not
                        a toy. Good enough to prove the pipeline's wiring
                        end-to-end and to run in any sandbox or CI box.
                        Weaker than a transformer embedding at catching
                        matches that share zero vocabulary (e.g. "one-click
                        onboarding" vs "reduce setup friction").

  PineconeVectorStore - production swap-in. Takes an `embed_fn` you supply
                        (sentence-transformers, OpenAI, Voyage, Cohere —
                        anything that returns a fixed-size float vector),
                        so Sank never hard-codes which embedding model you
                        use. Verified against the current Pinecone Python
                        SDK (client-instance style: `Pinecone(api_key=...)`,
                        not the old deprecated `pinecone.init()`).

Swap which one the pipeline uses with one constructor call — agents.py and
pipeline.py are written against the `VectorStore` interface only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sank.exceptions import VectorStoreError
from sank.models import MatchResult, ReferenceItem

logger = logging.getLogger("sank.vector_store")


class VectorStore(ABC):
    """Index a reference corpus once, then find the best match for new text."""

    @abstractmethod
    def index_reference_corpus(self, items: list[ReferenceItem]) -> None:
        raise NotImplementedError

    @abstractmethod
    def find_best_match(self, text: str, top_k: int = 1) -> list[MatchResult]:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """TF-IDF based local store. No network, no API key, fully deterministic."""

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._items: list[ReferenceItem] = []
        self._matrix = None

    def index_reference_corpus(self, items: list[ReferenceItem]) -> None:
        if not items:
            raise VectorStoreError("Cannot index an empty reference corpus")
        self._items = items
        # min_df=1 so this works even on the small corpora typical of a
        # hackathon demo (a handful of roadmap items); a production corpus
        # of hundreds of items could safely raise this.
        self._vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        self._matrix = self._vectorizer.fit_transform([i.text for i in items])
        logger.info("Indexed %d reference items (local TF-IDF)", len(items))

    def find_best_match(self, text: str, top_k: int = 1) -> list[MatchResult]:
        if self._vectorizer is None or self._matrix is None:
            raise VectorStoreError(
                "index_reference_corpus() must be called before find_best_match()"
            )
        if not text.strip():
            raise VectorStoreError("Cannot match an empty string")

        query_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [
            MatchResult(
                reference_item_id=self._items[i].id,
                reference_item_text=self._items[i].text,
                # Float precision can push cosine similarity a hair past
                # 1.0 (e.g. 1.0000000000000002) — clamp before validation
                # rather than let a precision artifact fail a whole source.
                similarity=float(np.clip(sims[i], -1.0, 1.0)),
            )
            for i in top_indices
        ]


class PineconeVectorStore(VectorStore):
    """Production store backed by Pinecone. You supply the embedding function.

    Verify the exact SDK call shapes against current Pinecone docs at
    integration time — third-party SDKs evolve faster than this file will
    be updated. Pattern below is the modern client-instance API.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embed_fn: Callable[[str], list[float]],
        namespace: str = "sank",
        dimension: int = 384,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        import pinecone  # lazy import: PineconeVectorStore is never required
        # to be importable for users who only ever run LocalVectorStore.

        self._pc = pinecone.Pinecone(api_key=api_key)
        self._embed_fn = embed_fn
        self._namespace = namespace
        self._index_name = index_name

        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if index_name not in existing:
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info("Created Pinecone index %s", index_name)

        self._index = self._pc.Index(index_name)

    def index_reference_corpus(self, items: list[ReferenceItem]) -> None:
        if not items:
            raise VectorStoreError("Cannot index an empty reference corpus")
        try:
            vectors = [
                (item.id, self._embed_fn(item.text), {"text": item.text, **item.metadata})
                for item in items
            ]
            self._index.upsert(vectors=vectors, namespace=self._namespace)
        except Exception as exc:
            raise VectorStoreError(f"Pinecone upsert failed: {exc}") from exc
        logger.info("Indexed %d reference items (Pinecone: %s)", len(items), self._index_name)

    def find_best_match(self, text: str, top_k: int = 1) -> list[MatchResult]:
        if not text.strip():
            raise VectorStoreError("Cannot match an empty string")
        try:
            query_vec = self._embed_fn(text)
            result = self._index.query(
                vector=query_vec,
                top_k=top_k,
                namespace=self._namespace,
                include_metadata=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

        return [
            MatchResult(
                reference_item_id=match.id,
                reference_item_text=(match.metadata or {}).get("text", ""),
                similarity=max(-1.0, min(1.0, float(match.score))),
            )
            for match in result.matches
        ]

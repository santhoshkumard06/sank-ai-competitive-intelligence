"""Tests for sank.vector_store.

LocalVectorStore needs no mocking at all — it's pure scikit-learn/numpy.
PineconeVectorStore is tested against a mocked `pinecone.Pinecone` client
so the test asserts our code calls the real SDK's methods with the right
shapes, without needing a live Pinecone account.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sank.exceptions import VectorStoreError
from sank.vector_store import LocalVectorStore, PineconeVectorStore

try:
    import pinecone  # noqa: F401

    _HAS_PINECONE = True
except ImportError:
    _HAS_PINECONE = False


class TestLocalVectorStore:
    def test_raises_before_indexing(self):
        store = LocalVectorStore()
        with pytest.raises(VectorStoreError, match="index_reference_corpus"):
            store.find_best_match("anything")

    def test_rejects_empty_corpus(self):
        with pytest.raises(VectorStoreError, match="empty"):
            LocalVectorStore().index_reference_corpus([])

    def test_rejects_empty_query(self, sample_reference_items):
        store = LocalVectorStore()
        store.index_reference_corpus(sample_reference_items)
        with pytest.raises(VectorStoreError, match="empty string"):
            store.find_best_match("   ")

    def test_finds_the_obviously_relevant_match(self, sample_reference_items):
        store = LocalVectorStore()
        store.index_reference_corpus(sample_reference_items)
        results = store.find_best_match("New real-time collaboration features for teams")
        assert results[0].reference_item_id == "rm-002"

    def test_similarity_is_always_in_valid_range(self, sample_reference_items):
        """Regression test for the float-precision bug found during
        pipeline testing: cosine_similarity() can return values like
        1.0000000000000002, which must be clamped before this leaves
        the store."""
        store = LocalVectorStore()
        store.index_reference_corpus(sample_reference_items)
        # Querying with text identical to a reference item is the case
        # most likely to produce a similarity of exactly/almost 1.0.
        results = store.find_best_match(sample_reference_items[0].text, top_k=3)
        for r in results:
            assert -1.0 <= r.similarity <= 1.0

    def test_top_k_returns_results_in_descending_similarity_order(self, sample_reference_items):
        store = LocalVectorStore()
        store.index_reference_corpus(sample_reference_items)
        results = store.find_best_match("teams collaborating on projects", top_k=3)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)


@pytest.mark.skipif(not _HAS_PINECONE, reason="pinecone is an optional extra (pip install -e '.[pinecone]')")
class TestPineconeVectorStore:
    def _make_store(self, mock_pinecone_module, index_exists=False):
        # mock_pinecone_module IS the patched `pinecone.Pinecone` class
        # (we patched "pinecone.Pinecone" directly) — so the instance it
        # produces when called is `.return_value`, not `.Pinecone.return_value`.
        # Mixing those up makes every assertion silently check an unrelated,
        # disconnected mock that the real code never touches.
        mock_pc_instance = mock_pinecone_module.return_value
        mock_pc_instance.list_indexes.return_value = (
            [{"name": "sank-roadmap"}] if index_exists else []
        )
        mock_index = MagicMock()
        mock_pc_instance.Index.return_value = mock_index
        store = PineconeVectorStore(
            api_key="dummy",
            index_name="sank-roadmap",
            embed_fn=lambda text: [0.1, 0.2, 0.3],
            dimension=3,
        )
        return store, mock_pc_instance, mock_index

    def test_creates_index_when_missing(self):
        with patch("pinecone.Pinecone") as mock_pinecone:
            _, mock_pc_instance, _ = self._make_store(mock_pinecone, index_exists=False)
        mock_pc_instance.create_index.assert_called_once()

    def test_does_not_recreate_existing_index(self):
        with patch("pinecone.Pinecone") as mock_pinecone:
            _, mock_pc_instance, _ = self._make_store(mock_pinecone, index_exists=True)
        mock_pc_instance.create_index.assert_not_called()

    def test_index_reference_corpus_calls_upsert_with_embeddings(self, sample_reference_items):
        with patch("pinecone.Pinecone") as mock_pinecone:
            store, _, mock_index = self._make_store(mock_pinecone)
            store.index_reference_corpus(sample_reference_items)
        vectors_arg = mock_index.upsert.call_args.kwargs["vectors"]
        assert len(vectors_arg) == len(sample_reference_items)
        assert vectors_arg[0][1] == [0.1, 0.2, 0.3]  # our embed_fn's fixed output

    def test_find_best_match_parses_pinecone_response(self):
        fake_match = MagicMock(id="rm-002", score=0.91, metadata={"text": "Collaborate"})
        with patch("pinecone.Pinecone") as mock_pinecone:
            store, _, mock_index = self._make_store(mock_pinecone)
            mock_index.query.return_value = MagicMock(matches=[fake_match])
            results = store.find_best_match("teamwork")
        assert results[0].reference_item_id == "rm-002"
        assert results[0].similarity == 0.91

    def test_upsert_failure_wrapped_as_vector_store_error(self, sample_reference_items):
        with patch("pinecone.Pinecone") as mock_pinecone:
            store, _, mock_index = self._make_store(mock_pinecone)
            mock_index.upsert.side_effect = RuntimeError("network blip")
            with pytest.raises(VectorStoreError, match="upsert failed"):
                store.index_reference_corpus(sample_reference_items)

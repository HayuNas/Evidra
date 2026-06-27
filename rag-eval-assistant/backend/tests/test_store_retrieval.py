import tempfile
import unittest
from pathlib import Path

from app.rag.store import LocalDocumentStore
from app.schemas.models import DocumentChunk, RetrievalConfig


class TestStoreRetrieval(unittest.TestCase):
    def test_replace_source_persists_and_retrieves_top_match(self):
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "documents.json"
            store = LocalDocumentStore(store_path)
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.", "Finance", 1),
                    DocumentChunk("handbook.md:0002", "handbook.md", "Rotate access keys quarterly.", "Security", 2),
                ],
            )

            reloaded = LocalDocumentStore(store_path)
            results = reloaded.search("expense receipts", top_k=1)

            self.assertEqual(results[0].id, "handbook.md:0001")
            self.assertGreater(results[0].score, 0)

    def test_replace_source_removes_old_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source("a.txt", [DocumentChunk("a.txt:0001", "a.txt", "old text")])
            store.replace_source("a.txt", [DocumentChunk("a.txt:0001", "a.txt", "new text")])

            self.assertEqual([chunk.text for chunk in store.list_chunks()], ["new text"])

    def test_empty_query_returns_no_results(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source("a.txt", [DocumentChunk("a.txt:0001", "a.txt", "some text")])

            self.assertEqual(store.search("   "), [])

    def test_search_respects_top_k_and_score_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses."),
                    DocumentChunk("handbook.md:0002", "handbook.md", "Receipts may include hotel invoices."),
                ],
            )

            results = store.search("receipts", RetrievalConfig(top_k=1, score_threshold=0.1))

            self.assertEqual(len(results), 1)
            self.assertGreaterEqual(results[0].score, 0.1)

    def test_bm25_mode_retrieves_matching_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses."),
                    DocumentChunk("handbook.md:0002", "handbook.md", "Rotate access keys quarterly."),
                ],
            )

            results = store.search("quarterly key rotation", RetrievalConfig(mode="bm25", top_k=1))

            self.assertEqual(results[0].id, "handbook.md:0002")

    def test_embedding_search_and_hybrid_search_use_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source(
                "policy.md",
                [
                    DocumentChunk(id="a", source="policy.md", text="alpha", embedding=[1.0, 0.0]),
                    DocumentChunk(id="b", source="policy.md", text="beta", embedding=[0.0, 1.0]),
                ],
            )

            embedding_results = store.search(
                "unmatched",
                RetrievalConfig(mode="embedding", top_k=1),
                query_embedding=[0.9, 0.1],
            )
            hybrid_results = store.search(
                "beta",
                RetrievalConfig(mode="hybrid", top_k=1),
                query_embedding=[0.1, 0.9],
            )

            self.assertEqual(embedding_results[0].id, "a")
            self.assertEqual(hybrid_results[0].id, "b")

    def test_document_listing_chunks_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(Path(directory) / "documents.json")
            store.replace_source("a.md", [DocumentChunk(id="a1", source="a.md", text="alpha")])
            store.replace_source("b.md", [DocumentChunk(id="b1", source="b.md", text="beta")])

            self.assertEqual(
                store.list_documents(),
                [{"source": "a.md", "chunk_count": 1}, {"source": "b.md", "chunk_count": 1}],
            )
            self.assertEqual([chunk.id for chunk in store.list_chunks("a.md")], ["a1"])

            store.delete_source("a.md")
            self.assertEqual([doc["source"] for doc in store.list_documents()], ["b.md"])

            store.clear()
            self.assertEqual(store.list_documents(), [])


if __name__ == "__main__":
    unittest.main()

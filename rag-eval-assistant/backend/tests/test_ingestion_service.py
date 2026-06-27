import tempfile
import unittest
from pathlib import Path

from app.rag.store import LocalDocumentStore
from app.schemas.models import ChunkingConfig
from app.services.ingestion import DocumentIngestionService
from app.settings import Settings


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class TestIngestionService(unittest.TestCase):
    def test_ingests_supported_document_and_writes_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(project_root=root)
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            service = DocumentIngestionService(settings, store)

            result = service.ingest("handbook.md", b"# Finance\nReceipts are required.")

            self.assertEqual(result.source, "handbook.md")
            self.assertEqual(result.chunk_count, 1)
            self.assertTrue((settings.upload_dir / "handbook.md").exists())
            self.assertEqual(store.search("receipts")[0].source, "handbook.md")

    def test_ingest_uses_chunking_config(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            service = DocumentIngestionService(settings, store)

            result = service.ingest(
                "handbook.md",
                b"First paragraph.\n\nSecond paragraph.",
                ChunkingConfig(strategy="paragraph"),
            )

            self.assertEqual(result.chunk_count, 2)
            self.assertEqual([chunk.text for chunk in store.list_chunks()], ["First paragraph.", "Second paragraph."])

    def test_ingest_writes_embeddings_when_provider_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            service = DocumentIngestionService(settings, store, embedding_provider=FakeEmbeddingProvider())

            service.ingest(
                "handbook.md",
                b"First paragraph.\n\nSecond paragraph.",
                ChunkingConfig(strategy="paragraph"),
            )

            self.assertEqual([chunk.embedding for chunk in store.list_chunks()], [[0.0, 1.0], [1.0, 1.0]])

    def test_rejects_oversized_upload_before_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory), max_upload_bytes=4)
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            service = DocumentIngestionService(settings, store)

            with self.assertRaisesRegex(ValueError, "File is too large"):
                service.ingest("large.txt", b"12345")

            self.assertFalse((settings.upload_dir / "large.txt").exists())

    def test_parse_failure_does_not_replace_existing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            service = DocumentIngestionService(settings, store)
            service.ingest("notes.txt", b"original searchable text")

            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                service.ingest("notes.png", b"broken")

            self.assertEqual(store.search("original")[0].source, "notes.txt")


if __name__ == "__main__":
    unittest.main()

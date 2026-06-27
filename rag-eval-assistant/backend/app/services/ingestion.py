from __future__ import annotations

from pathlib import Path

from app.rag.store import LocalDocumentStore
from app.rag.text import chunk_text, extract_text
from app.schemas.models import ChunkingConfig, IngestedDocument
from app.services.model_providers import EmbeddingProvider
from app.settings import Settings


class DocumentIngestionService:
    def __init__(self, settings: Settings, store: LocalDocumentStore, embedding_provider: EmbeddingProvider | None = None):
        self.settings = settings
        self.store = store
        self.embedding_provider = embedding_provider

    def ingest(self, filename: str, content: bytes, chunking: ChunkingConfig | None = None) -> IngestedDocument:
        clean_name = Path(filename).name
        if len(content) > self.settings.max_upload_bytes:
            raise ValueError("File is too large.")

        text = extract_text(clean_name, content)
        chunks = chunk_text(text, source=clean_name, config=chunking)
        if not chunks:
            raise ValueError("No text content could be indexed.")
        if self.embedding_provider is not None:
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            chunks = [
                type(chunk)(
                    id=chunk.id,
                    source=chunk.source,
                    text=chunk.text,
                    heading=chunk.heading,
                    position=chunk.position,
                    embedding=embeddings[index] if index < len(embeddings) else None,
                )
                for index, chunk in enumerate(chunks)
            ]

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = self.settings.upload_dir / clean_name
        saved_path.write_bytes(content)
        self.store.replace_source(clean_name, chunks)
        return IngestedDocument(source=clean_name, chunk_count=len(chunks), saved_path=str(saved_path))

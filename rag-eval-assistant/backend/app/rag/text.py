from __future__ import annotations

import re
from pathlib import Path

from app.schemas.models import ChunkingConfig
from app.schemas.models import DocumentChunk

SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return content.decode("utf-8-sig").strip()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF parsing requires installing the pypdf extra.") from exc
        import io

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def tokenize(text: str) -> list[str]:
    return [_normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _normalize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def chunk_text(
    text: str,
    source: str,
    config: ChunkingConfig | None = None,
    max_chars: int | None = None,
) -> list[DocumentChunk]:
    if config is None:
        config = ChunkingConfig(chunk_size=max_chars or 900)
    if config.strategy == "paragraph":
        return _chunk_paragraphs(text, source, config)
    if config.strategy == "fixed":
        return _chunk_fixed(text, source, config)
    return _chunk_heading(text, source, config)


def _make_chunk(source: str, text: str, heading: str | None, position: int) -> DocumentChunk:
    return DocumentChunk(
        id=f"{source}:{position:04d}",
        source=source,
        text=text.strip(),
        heading=heading,
        position=position,
    )


def _chunk_heading(text: str, source: str, config: ChunkingConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        joined = "\n".join(buffer).strip()
        buffer.clear()
        if not joined:
            return
        position = len(chunks) + 1
        chunks.append(_make_chunk(source, joined, heading, position))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading_match:
            flush()
            heading = heading_match.group(1).strip()
            continue
        if not line:
            flush()
            continue
        projected = "\n".join([*buffer, line])
        if buffer and len(projected) > config.chunk_size:
            flush()
        buffer.append(line)
    flush()
    return chunks


def _chunk_paragraphs(text: str, source: str, config: ChunkingConfig) -> list[DocumentChunk]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[DocumentChunk] = []
    for paragraph in paragraphs:
        if len(paragraph) <= config.chunk_size:
            chunks.append(_make_chunk(source, paragraph, None, len(chunks) + 1))
            continue
        step = max(1, config.chunk_size - config.overlap)
        for start in range(0, len(paragraph), step):
            piece = paragraph[start : start + config.chunk_size].strip()
            if piece:
                chunks.append(_make_chunk(source, piece, None, len(chunks) + 1))
            if start + config.chunk_size >= len(paragraph):
                break
    return chunks


def _chunk_fixed(text: str, source: str, config: ChunkingConfig) -> list[DocumentChunk]:
    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[DocumentChunk] = []
    step = max(1, config.chunk_size - config.overlap)
    for start in range(0, len(normalized), step):
        piece = normalized[start : start + config.chunk_size].strip()
        if not piece:
            continue
        chunks.append(_make_chunk(source, piece, None, len(chunks) + 1))
        if start + config.chunk_size >= len(normalized):
            break
    return chunks

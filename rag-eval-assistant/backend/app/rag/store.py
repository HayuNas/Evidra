from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.rag.text import tokenize
from app.schemas.models import DocumentChunk, RetrievalConfig, RetrievedChunk


class LocalDocumentStore:
    def __init__(self, path: Path):
        self.path = path
        self._chunks = self._load()

    def replace_source(self, source: str, chunks: Iterable[DocumentChunk]) -> None:
        remaining = [chunk for chunk in self._chunks if chunk.source != source]
        self._chunks = [*remaining, *chunks]
        self._save()

    def list_chunks(self, source: str | None = None) -> list[DocumentChunk]:
        if source is None:
            return list(self._chunks)
        return [chunk for chunk in self._chunks if chunk.source == source]

    def list_documents(self) -> list[dict[str, int | str]]:
        counts: Counter[str] = Counter(chunk.source for chunk in self._chunks)
        return [
            {"source": source, "chunk_count": counts[source]}
            for source in sorted(counts)
        ]

    def delete_source(self, source: str) -> None:
        self._chunks = [chunk for chunk in self._chunks if chunk.source != source]
        self._save()

    def clear(self) -> None:
        self._chunks = []
        self._save()

    def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        top_k: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        config = config or RetrievalConfig(top_k=top_k or 4)
        query_terms = tokenize(query)
        if not query_terms and not query_embedding:
            return []
        query_counts = Counter(query_terms)
        document_frequencies = self._document_frequencies()
        average_length = self._average_chunk_length()
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            chunk_terms = Counter(tokenize(chunk.text))
            if not chunk_terms and not chunk.embedding:
                continue
            lexical_score = 0.0
            if query_terms and config.mode in {"tfidf", "hybrid"}:
                lexical_score = self._score_tfidf(query_counts, chunk_terms, document_frequencies)
            elif query_terms and config.mode == "bm25":
                lexical_score = self._score_bm25(query_counts, chunk_terms, document_frequencies, average_length)

            embedding_score = 0.0
            if query_embedding and chunk.embedding and config.mode in {"embedding", "hybrid"}:
                embedding_score = self._cosine_similarity(query_embedding, chunk.embedding)

            if config.mode == "embedding":
                score = embedding_score
            elif config.mode == "hybrid":
                score = lexical_score + embedding_score
            else:
                score = lexical_score
            rounded_score = round(score, 4)
            if rounded_score > 0 and rounded_score >= config.score_threshold:
                scored.append(
                    RetrievedChunk(
                        id=chunk.id,
                        source=chunk.source,
                        text=chunk.text,
                        heading=chunk.heading,
                        position=chunk.position,
                        embedding=chunk.embedding,
                        score=rounded_score,
                    )
                )
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[: config.top_k]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        dot = sum(left[index] * right[index] for index in range(size))
        left_norm = math.sqrt(sum(value * value for value in left[:size]))
        right_norm = math.sqrt(sum(value * value for value in right[:size]))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _score_tfidf(
        self,
        query_counts: Counter[str],
        chunk_terms: Counter[str],
        document_frequencies: Counter[str],
    ) -> float:
        score = 0.0
        for term, query_count in query_counts.items():
            if term not in chunk_terms:
                continue
            inverse_document_frequency = math.log((1 + len(self._chunks)) / (1 + document_frequencies[term])) + 1
            score += query_count * chunk_terms[term] * inverse_document_frequency
        return score

    def _score_bm25(
        self,
        query_counts: Counter[str],
        chunk_terms: Counter[str],
        document_frequencies: Counter[str],
        average_length: float,
    ) -> float:
        k1 = 1.5
        b = 0.75
        length = sum(chunk_terms.values()) or 1
        score = 0.0
        for term, query_count in query_counts.items():
            term_frequency = chunk_terms.get(term, 0)
            if term_frequency == 0:
                continue
            idf = math.log((len(self._chunks) - document_frequencies[term] + 0.5) / (document_frequencies[term] + 0.5) + 1)
            denominator = term_frequency + k1 * (1 - b + b * length / max(average_length, 1))
            score += query_count * idf * ((term_frequency * (k1 + 1)) / denominator)
        return score

    def _average_chunk_length(self) -> float:
        lengths = [len(tokenize(chunk.text)) for chunk in self._chunks]
        return sum(lengths) / len(lengths) if lengths else 0.0

    def _document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for chunk in self._chunks:
            frequencies.update(set(tokenize(chunk.text)))
        return frequencies

    def _load(self) -> list[DocumentChunk]:
        if not self.path.exists():
            return []
        raw_chunks = json.loads(self.path.read_text(encoding="utf-8"))
        return [DocumentChunk(**raw_chunk) for raw_chunk in raw_chunks]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [chunk.to_dict() for chunk in self._chunks]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

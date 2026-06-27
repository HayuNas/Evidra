from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "heading"
    chunk_size: int = 900
    overlap: int = 0

    def __post_init__(self) -> None:
        strategy = self.strategy if self.strategy in {"heading", "paragraph", "fixed"} else "heading"
        chunk_size = min(2000, max(300, int(self.chunk_size)))
        overlap = min(chunk_size - 1, max(0, int(self.overlap)))
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "overlap", overlap)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "tfidf"
    top_k: int = 4
    score_threshold: float = 0.0

    def __post_init__(self) -> None:
        mode = self.mode if self.mode in {"tfidf", "bm25", "embedding", "hybrid"} else "tfidf"
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "top_k", min(10, max(1, int(self.top_k))))
        object.__setattr__(self, "score_threshold", max(0.0, float(self.score_threshold)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptConfig:
    version: str = "grounded"

    def __post_init__(self) -> None:
        version = self.version if self.version in {"grounded", "concise", "analyst"} else "grounded"
        object.__setattr__(self, "version", version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagStrategyConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunking": self.chunking.to_dict(),
            "retrieval": self.retrieval.to_dict(),
        }


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    text: str
    heading: str | None = None
    position: int = 0
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk(DocumentChunk):
    score: float = 0.0


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    snippet: str
    score: float
    heading: str | None = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "Citation":
        return cls(
            chunk_id=chunk.id,
            source=chunk.source,
            snippet=chunk.text,
            score=chunk.score,
            heading=chunk.heading,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload


@dataclass(frozen=True)
class IngestedDocument:
    source: str
    chunk_count: int
    saved_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationItem:
    question: str
    expected_terms: list[str] = field(default_factory=list)
    required_source: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    item: EvaluationItem
    answer: AnswerResult
    passed: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.item.question,
            "expected_terms": self.item.expected_terms,
            "required_source": self.item.required_source,
            "answer": self.answer.to_dict(),
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    passed: int
    citation_coverage: float
    average_latency_ms: float
    results: list[EvaluationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.passed / self.total if self.total else 0.0,
            "citation_coverage": self.citation_coverage,
            "average_latency_ms": self.average_latency_ms,
            "results": [result.to_dict() for result in self.results],
        }

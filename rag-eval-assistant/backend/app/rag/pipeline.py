from __future__ import annotations

from app.rag.store import LocalDocumentStore
from app.schemas.models import AnswerResult, Citation, PromptConfig, RetrievalConfig, RetrievedChunk
from app.services.langfuse_trace import LangfuseTraceService
from app.services.model_providers import AnswerProvider, EmbeddingProvider, LocalAnswerProvider, QueryRewriteProvider, RerankProvider

NO_EVIDENCE_ANSWER = "知识库中未找到足够依据。"


class RagPipeline:
    def __init__(
        self,
        store: LocalDocumentStore,
        trace_service: LangfuseTraceService,
        top_k: int = 4,
        answer_provider: AnswerProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        query_rewrite_provider: QueryRewriteProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ):
        self.store = store
        self.trace_service = trace_service
        self.top_k = top_k
        self.answer_provider = answer_provider or LocalAnswerProvider()
        self.embedding_provider = embedding_provider
        self.query_rewrite_provider = query_rewrite_provider
        self.rerank_provider = rerank_provider

    def answer(
        self,
        question: str,
        retrieval: RetrievalConfig | None = None,
        prompt_version: str = "grounded",
        rewrite_query: bool = False,
        multi_route: bool = False,
        rerank: bool = False,
    ) -> AnswerResult:
        retrieval = retrieval or RetrievalConfig(top_k=self.top_k)
        prompt = PromptConfig(prompt_version)
        query = self._query_metadata(question, rewrite_query)
        search_query = query["rewritten"]
        retrieved, retrieval_metadata = self._retrieve(search_query, question, retrieval, multi_route, rerank)
        retrieved, rerank_metadata = self._rerank(search_query, retrieved, retrieval.top_k, rerank)
        base_metadata = {"retrieval": retrieval_metadata, "prompt": prompt.to_dict(), "query": query, "rerank": rerank_metadata}
        if not retrieved:
            trace_id = self.trace_service.record_question(
                question,
                [],
                NO_EVIDENCE_ANSWER,
                metadata=base_metadata,
            )
            return AnswerResult(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                citations=[],
                confidence=0.0,
                trace_id=trace_id,
                metadata=base_metadata,
            )

        citations = [Citation.from_chunk(chunk) for chunk in retrieved]
        generated = self.answer_provider.generate(question, citations, prompt.version)
        answer = generated.text or self._compose_answer(citations)
        confidence = min(1.0, round(sum(chunk.score for chunk in retrieved) / (len(retrieved) * 3), 2))
        trace_id = self.trace_service.record_question(
            question,
            [chunk.to_dict() for chunk in retrieved],
            answer,
            metadata=base_metadata,
        )
        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            confidence=confidence,
            trace_id=trace_id,
            metadata={
                "retrieval": retrieval_metadata,
                "prompt": prompt.to_dict(),
                "query": query,
                "rerank": rerank_metadata,
                "generation": {"provider": generated.provider, "model": generated.model},
            },
        )

    def _retrieve(
        self,
        search_query: str,
        original_question: str,
        retrieval: RetrievalConfig,
        multi_route: bool,
        rerank: bool,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        candidate_k = retrieval.top_k if not rerank and not multi_route else min(10, max(retrieval.top_k * 4, retrieval.top_k))
        if multi_route:
            retrieved = self._multi_route_search(search_query, original_question, retrieval, candidate_k)
        else:
            retrieved = self._single_route_search(search_query, retrieval, candidate_k)
        metadata = retrieval.to_dict()
        metadata["multi_route"] = multi_route
        metadata["candidate_count"] = len(retrieved)
        metadata["candidate_k"] = candidate_k
        return retrieved[:candidate_k], metadata

    def _single_route_search(self, query: str, retrieval: RetrievalConfig, candidate_k: int) -> list[RetrievedChunk]:
        query_embedding = self._embed_query(query) if retrieval.mode in {"embedding", "hybrid"} else None
        config = RetrievalConfig(mode=retrieval.mode, top_k=candidate_k, score_threshold=retrieval.score_threshold)
        return self.store.search(query, config, query_embedding=query_embedding)

    def _multi_route_search(
        self,
        search_query: str,
        original_question: str,
        retrieval: RetrievalConfig,
        candidate_k: int,
    ) -> list[RetrievedChunk]:
        route_modes = ["tfidf", "bm25"]
        if self.embedding_provider is not None:
            route_modes.extend(["embedding", "hybrid"])
        route_queries = [original_question]
        if search_query != original_question:
            route_queries.append(search_query)

        merged: dict[str, RetrievedChunk] = {}
        route_count = 0
        for query in route_queries:
            for mode in route_modes:
                route_count += 1
                query_embedding = self._embed_query(query) if mode in {"embedding", "hybrid"} else None
                config = RetrievalConfig(mode=mode, top_k=candidate_k, score_threshold=retrieval.score_threshold)
                for rank, chunk in enumerate(self.store.search(query, config, query_embedding=query_embedding), start=1):
                    rrf_score = 1 / (60 + rank)
                    score = chunk.score + rrf_score
                    existing = merged.get(chunk.id)
                    if existing is None or score > existing.score:
                        merged[chunk.id] = RetrievedChunk(
                            id=chunk.id,
                            source=chunk.source,
                            text=chunk.text,
                            heading=chunk.heading,
                            position=chunk.position,
                            embedding=chunk.embedding,
                            score=round(score, 4),
                        )
        if route_count == 0:
            return []
        return sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:candidate_k]

    def _embed_query(self, query: str) -> list[float] | None:
        if self.embedding_provider is None:
            return None
        embeddings = self.embedding_provider.embed_texts([query])
        return embeddings[0] if embeddings else None

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
        rerank: bool,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        provider = self.rerank_provider
        if rerank and provider is not None and chunks:
            ranked = provider.rerank(query, chunks, top_k)
            return ranked[:top_k], {
                "enabled": True,
                "provider": provider.provider,
                "model": provider.model,
                "input_count": len(chunks),
            }
        return chunks[:top_k], {"enabled": False, "provider": "none", "model": "none", "input_count": len(chunks)}

    def _query_metadata(self, question: str, rewrite_query: bool) -> dict[str, str | bool]:
        provider = self.query_rewrite_provider
        if rewrite_query and provider is not None:
            rewritten = provider.rewrite(question).strip() or question
            return {
                "original": question,
                "rewritten": rewritten,
                "rewrite_enabled": True,
                "rewrite_provider": provider.provider,
                "rewrite_model": provider.model,
            }
        return {
            "original": question,
            "rewritten": question,
            "rewrite_enabled": False,
            "rewrite_provider": "none",
            "rewrite_model": "none",
        }

    def _compose_answer(self, citations: list[Citation]) -> str:
        lines = []
        for citation in citations:
            lines.append(f"{citation.snippet} [{citation.chunk_id}]")
        return "\n".join(lines)


class RagPipelineFactory:
    def __init__(
        self,
        store: LocalDocumentStore,
        trace_service: LangfuseTraceService,
        top_k: int = 4,
        answer_provider: AnswerProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        query_rewrite_provider: QueryRewriteProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ):
        self.store = store
        self.trace_service = trace_service
        self.top_k = top_k
        self.answer_provider = answer_provider
        self.embedding_provider = embedding_provider
        self.query_rewrite_provider = query_rewrite_provider
        self.rerank_provider = rerank_provider

    def create(self) -> RagPipeline:
        return RagPipeline(
            self.store,
            self.trace_service,
            self.top_k,
            answer_provider=self.answer_provider,
            embedding_provider=self.embedding_provider,
            query_rewrite_provider=self.query_rewrite_provider,
            rerank_provider=self.rerank_provider,
        )

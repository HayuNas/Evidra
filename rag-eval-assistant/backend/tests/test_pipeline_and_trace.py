import json
import tempfile
import unittest
from pathlib import Path

from app.rag.pipeline import RagPipelineFactory
from app.rag.store import LocalDocumentStore
from app.schemas.models import Citation, DocumentChunk, RetrievalConfig
from app.services.langfuse_trace import LangfuseTraceService
from app.services.model_providers import GeneratedAnswer
from app.settings import Settings


class ExplodingClient:
    def trace(self, **kwargs):
        raise RuntimeError("network down")


class HealthyLangfuseClient:
    def __init__(self):
        self.scores = []

    def auth_check(self):
        return True

    def trace(self, **kwargs):
        return type("Trace", (), {"id": "lf-trace-1"})()

    def score(self, **kwargs):
        self.scores.append(kwargs)


class FakeAnswerProvider:
    def __init__(self):
        self.prompt_versions = []

    def generate(self, question: str, citations: list[Citation], prompt_version: str = "grounded") -> GeneratedAnswer:
        self.prompt_versions.append(prompt_version)
        return GeneratedAnswer(text=f"generated: {prompt_version}: {question} / {citations[0].chunk_id}", provider="fake", model="fake-model")


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeQueryRewriteProvider:
    provider = "fake"
    model = "fake-rewriter"

    def __init__(self):
        self.questions = []

    def rewrite(self, question: str) -> str:
        self.questions.append(question)
        return "graduation school education university"


class FakeRerankProvider:
    provider = "fake"
    model = "fake-reranker"

    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, top_n):
        self.calls.append((query, [chunk.id for chunk in chunks], top_n))
        ranked = sorted(chunks, key=lambda chunk: chunk.id, reverse=True)
        return ranked[:top_n]


class TestPipelineAndTrace(unittest.TestCase):
    def test_pipeline_returns_cited_answer_for_retrieved_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.", "Finance", 1)],
            )
            pipeline = RagPipelineFactory(store, LangfuseTraceService(settings)).create()

            result = pipeline.answer("What do expense reports need?")

            self.assertIn("Receipts are required", result.answer)
            self.assertEqual(result.citations[0].chunk_id, "handbook.md:0001")
            self.assertGreater(result.confidence, 0)
            self.assertIsNotNone(result.trace_id)

    def test_answer_uses_retrieval_config_and_returns_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.", "Finance", 1),
                    DocumentChunk("handbook.md:0002", "handbook.md", "Receipts may include hotel invoices.", "Finance", 2),
                ],
            )
            pipeline = RagPipelineFactory(store, LangfuseTraceService(settings)).create()

            result = pipeline.answer("receipts", RetrievalConfig(mode="bm25", top_k=1))

            self.assertEqual(len(result.citations), 1)
            self.assertEqual(result.metadata["retrieval"]["mode"], "bm25")
            self.assertEqual(result.metadata["retrieval"]["top_k"], 1)

    def test_pipeline_can_use_answer_provider_and_embedding_query(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk(
                        "handbook.md:0001",
                        "handbook.md",
                        "Receipts are required for expenses.",
                        embedding=[1.0, 0.0],
                    )
                ],
            )
            provider = FakeAnswerProvider()
            pipeline = RagPipelineFactory(
                store,
                LangfuseTraceService(settings),
                answer_provider=provider,
                embedding_provider=FakeEmbeddingProvider(),
            ).create()

            result = pipeline.answer("What do expense reports need?", RetrievalConfig(mode="embedding"), prompt_version="concise")

            self.assertEqual(result.answer, "generated: concise: What do expense reports need? / handbook.md:0001")
            self.assertEqual(provider.prompt_versions, ["concise"])
            self.assertEqual(result.metadata["prompt"]["version"], "concise")
            self.assertEqual(result.metadata["generation"]["provider"], "fake")
            self.assertEqual(result.metadata["generation"]["model"], "fake-model")

    def test_trace_service_records_feedback_scores_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            trace = LangfuseTraceService(settings)

            feedback_id = trace.record_feedback("trace-123", "helpful", 1.0, "Good citation")

            self.assertTrue(feedback_id.startswith("local-"))
            events = [
                json.loads(line)
                for line in (settings.traces_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            feedback_events = [event for event in events if event.get("type") == "feedback"]
            self.assertEqual(feedback_events[0]["trace_id"], "trace-123")
            self.assertEqual(feedback_events[0]["label"], "helpful")
            self.assertEqual(feedback_events[0]["value"], 1.0)
            self.assertEqual(feedback_events[0]["comment"], "Good citation")

    def test_pipeline_returns_no_evidence_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            pipeline = RagPipelineFactory(store, LangfuseTraceService(settings)).create()

            result = pipeline.answer("Unknown?")

            self.assertEqual(result.answer, "知识库中未找到足够依据。")
            self.assertEqual(result.citations, [])
            self.assertEqual(result.confidence, 0.0)

    def test_pipeline_can_rewrite_colloquial_query_before_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "resume.md",
                [DocumentChunk("resume.md:0001", "resume.md", "Education: Guangzhou University.", "Education", 1)],
            )
            answer_provider = FakeAnswerProvider()
            rewrite_provider = FakeQueryRewriteProvider()
            pipeline = RagPipelineFactory(
                store,
                LangfuseTraceService(settings),
                answer_provider=answer_provider,
                query_rewrite_provider=rewrite_provider,
            ).create()

            result = pipeline.answer("他学校哪的？", RetrievalConfig(mode="tfidf", top_k=1), rewrite_query=True)

            self.assertEqual(rewrite_provider.questions, ["他学校哪的？"])
            self.assertEqual(answer_provider.prompt_versions, ["grounded"])
            self.assertIn("Guangzhou University", result.citations[0].snippet)
            self.assertEqual(result.metadata["query"]["original"], "他学校哪的？")
            self.assertEqual(result.metadata["query"]["rewritten"], "graduation school education university")
            self.assertTrue(result.metadata["query"]["rewrite_enabled"])
            self.assertEqual(result.metadata["query"]["rewrite_provider"], "fake")
            self.assertEqual(result.metadata["query"]["rewrite_model"], "fake-rewriter")

    def test_pipeline_can_use_multi_route_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "resume.md",
                [
                    DocumentChunk("resume.md:0001", "resume.md", "Education: Guangzhou University.", "Education", 1),
                    DocumentChunk("resume.md:0002", "resume.md", "Project: invoice extraction.", "Project", 2),
                ],
            )
            pipeline = RagPipelineFactory(
                store,
                LangfuseTraceService(settings),
                query_rewrite_provider=FakeQueryRewriteProvider(),
            ).create()

            result = pipeline.answer(
                "school?",
                RetrievalConfig(mode="tfidf", top_k=1),
                rewrite_query=True,
                multi_route=True,
            )

            self.assertEqual(result.citations[0].chunk_id, "resume.md:0001")
            self.assertTrue(result.metadata["retrieval"]["multi_route"])
            self.assertGreaterEqual(result.metadata["retrieval"]["candidate_count"], 1)

    def test_pipeline_can_rerank_retrieved_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [
                    DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.", "Finance", 1),
                    DocumentChunk("handbook.md:0002", "handbook.md", "Receipts may include hotel invoices.", "Finance", 2),
                ],
            )
            rerank_provider = FakeRerankProvider()
            pipeline = RagPipelineFactory(
                store,
                LangfuseTraceService(settings),
                rerank_provider=rerank_provider,
            ).create()

            result = pipeline.answer("receipts", RetrievalConfig(mode="tfidf", top_k=2), rerank=True)

            self.assertEqual(result.citations[0].chunk_id, "handbook.md:0002")
            self.assertTrue(result.metadata["rerank"]["enabled"])
            self.assertEqual(result.metadata["rerank"]["provider"], "fake")
            self.assertEqual(rerank_provider.calls[0][0], "receipts")

    def test_trace_service_degrades_to_local_jsonl_when_client_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            trace = LangfuseTraceService(settings, client=ExplodingClient())

            trace_id = trace.record_question("Question?", [{"id": "a"}], "Answer")

            self.assertTrue(trace_id.startswith("local-"))
            events = (settings.traces_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(events[0])["question"], "Question?")

    def test_trace_service_reports_status_and_records_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory), langfuse_public_key="pk", langfuse_secret_key="sk")
            client = HealthyLangfuseClient()
            trace = LangfuseTraceService(settings, client=client)

            self.assertEqual(trace.status()["mode"], "langfuse")
            trace_id = trace.record_question("Question?", [{"id": "a"}], "Answer")
            trace.record_score(trace_id, "pass_rate", 1.0, "evaluation summary")

            self.assertEqual(trace_id, "lf-trace-1")
            self.assertEqual(client.scores[0]["name"], "pass_rate")
            self.assertEqual(client.scores[0]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()

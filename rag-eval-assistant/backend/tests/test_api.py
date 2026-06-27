import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None

from app.api.main import build_app
from app.settings import Settings


class FailingAnswerProvider:
    def generate(self, question, citations, prompt_version="grounded"):
        raise RuntimeError("answer provider exploded")


class FailingEmbeddingProvider:
    def embed_texts(self, texts):
        raise RuntimeError("embedding provider exploded")


class FakeQueryRewriteProvider:
    provider = "fake"
    model = "fake-rewriter"

    def rewrite(self, question):
        return "graduation school education university"


class FakeRerankProvider:
    provider = "fake"
    model = "fake-reranker"

    def rerank(self, query, chunks, top_n):
        return list(reversed(chunks))[:top_n]


@unittest.skipIf(TestClient is None, "FastAPI is not installed in this Python environment.")
class TestApi(unittest.TestCase):
    def test_serves_static_frontend_from_backend_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend_dir = root / "frontend"
            frontend_dir.mkdir()
            (frontend_dir / "index.html").write_text(
                '<!doctype html><html><body><script src="app.js"></script>Evidra</body></html>',
                encoding="utf-8",
            )
            (frontend_dir / "app.js").write_text("window.evidraLoaded = true;", encoding="utf-8")

            app = build_app(Settings(project_root=root))
            client = TestClient(app)

            index = client.get("/")
            script = client.get("/app.js")

            self.assertEqual(index.status_code, 200)
            self.assertIn("Evidra", index.text)
            self.assertEqual(script.status_code, 200)
            self.assertIn("window.evidraLoaded", script.text)

    def test_health_documents_ask_and_eval_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            self.assertEqual(client.get("/health").json()["status"], "ok")

            upload = client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )
            self.assertEqual(upload.status_code, 200)
            self.assertEqual(upload.json()["chunk_count"], 1)

            answer = client.post("/ask", json={"question": "What do expenses need?"})
            self.assertEqual(answer.status_code, 200)
            self.assertTrue(answer.json()["citations"])

    def test_documents_accept_chunking_form_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            upload = client.post(
                "/documents",
                data={"chunk_strategy": "paragraph", "chunk_size": "900", "chunk_overlap": "0"},
                files={"file": ("sample-handbook.md", b"First paragraph.\n\nSecond paragraph.")},
            )

            self.assertEqual(upload.status_code, 200)
            self.assertEqual(upload.json()["chunk_count"], 2)

    def test_document_upload_returns_json_error_when_embedding_provider_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            with patch("app.api.main.build_embedding_provider", return_value=FailingEmbeddingProvider()):
                response = client.post(
                    "/documents",
                    files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
                )

            self.assertEqual(response.status_code, 502)
            self.assertIn("embedding provider exploded", response.json()["detail"])

    def test_ask_returns_json_error_when_answer_provider_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )

            with patch("app.api.main.build_answer_provider", return_value=FailingAnswerProvider()):
                response = client.post("/ask", json={"question": "What do expenses need?"})

            self.assertEqual(response.status_code, 502)
            self.assertIn("answer provider exploded", response.json()["detail"])

    def test_ask_accepts_retrieval_config(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )

            answer = client.post(
                "/ask",
                json={
                    "question": "What do expenses need?",
                    "prompt_version": "concise",
                    "retrieval": {"mode": "bm25", "top_k": 1, "score_threshold": 0},
                },
            )

            self.assertEqual(answer.status_code, 200)
            self.assertEqual(answer.json()["metadata"]["retrieval"]["mode"], "bm25")
            self.assertEqual(answer.json()["metadata"]["prompt"]["version"], "concise")
            self.assertEqual(len(answer.json()["citations"]), 1)

    def test_ask_can_enable_query_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("resume.md", b"# Education\nEducation: Guangzhou University.")},
            )

            with patch("app.api.main.build_query_rewrite_provider", return_value=FakeQueryRewriteProvider()):
                answer = client.post(
                    "/ask",
                    json={
                        "question": "school?",
                        "rewrite_query": True,
                        "retrieval": {"mode": "tfidf", "top_k": 1, "score_threshold": 0},
                    },
                )

            self.assertEqual(answer.status_code, 200)
            query = answer.json()["metadata"]["query"]
            self.assertTrue(query["rewrite_enabled"])
            self.assertEqual(query["original"], "school?")
            self.assertEqual(query["rewritten"], "graduation school education university")
            self.assertEqual(query["rewrite_provider"], "fake")

    def test_ask_can_enable_multi_route_and_rerank(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("resume.md", b"# Education\nEducation: Guangzhou University.")},
            )

            with patch("app.api.main.build_query_rewrite_provider", return_value=FakeQueryRewriteProvider()), patch(
                "app.api.main.build_rerank_provider", return_value=FakeRerankProvider()
            ):
                answer = client.post(
                    "/ask",
                    json={
                        "question": "school?",
                        "rewrite_query": True,
                        "multi_route": True,
                        "rerank": True,
                        "retrieval": {"mode": "tfidf", "top_k": 1, "score_threshold": 0},
                    },
                )

            self.assertEqual(answer.status_code, 200)
            self.assertTrue(answer.json()["metadata"]["retrieval"]["multi_route"])
            self.assertTrue(answer.json()["metadata"]["rerank"]["enabled"])

    def test_compare_evaluations_returns_multiple_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(project_root=root)
            settings.eval_sets_dir.mkdir(parents=True)
            (settings.eval_sets_dir / "demo.json").write_text(
                json.dumps([{"question": "What do expenses need?", "expected_terms": ["Receipts"]}]),
                encoding="utf-8",
            )
            app = build_app(settings)
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )

            response = client.post(
                "/evaluations/compare",
                json={
                    "eval_set": "demo.json",
                    "strategies": [
                        {"name": "tfidf-k2", "retrieval": {"mode": "tfidf", "top_k": 2}},
                        {"name": "bm25-k1", "prompt_version": "analyst", "retrieval": {"mode": "bm25", "top_k": 1}},
                    ],
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual([item["name"] for item in response.json()["comparisons"]], ["tfidf-k2", "bm25-k1"])
            self.assertEqual(response.json()["comparisons"][1]["prompt"]["version"], "analyst")

    def test_feedback_route_records_score_without_exposing_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            app = build_app(settings)
            client = TestClient(app)

            response = client.post(
                "/feedback",
                json={"trace_id": "trace-123", "label": "helpful", "value": 1.0, "comment": "Useful citation"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["recorded"], True)
            self.assertEqual(response.json()["trace_id"], "trace-123")
            events = [
                json.loads(line)
                for line in (settings.traces_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0]["type"], "feedback")
            self.assertEqual(events[0]["label"], "helpful")

    def test_cors_preflight_allows_static_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            response = client.options(
                "/ask",
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], "*")
            self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_runtime_config_can_switch_model_group_and_masks_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            initial = client.get("/config").json()
            self.assertEqual(initial["model_group"], "local")
            self.assertEqual(initial["answer_provider"], "local")

            response = client.put(
                "/config",
                json={
                    "model_group": "openai",
                    "answer_provider": "openai",
                    "embedding_provider": "openai",
                    "answer_api_mode": "chat_completions",
                    "answer_model": "gpt-4.1-mini",
                    "embedding_model": "text-embedding-3-small",
                    "openai_base_url": "https://gateway.example.com/v1",
                    "openai_api_key": "test-openai-secret",
                    "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "embedding_api_key": "test-embedding-secret",
                    "answer_prompt_template": "Answer using context: {question} {context}",
                    "rewrite_prompt_template": "Rewrite query: {question}",
                    "rerank_provider": "dashscope",
                    "rerank_model": "gte-rerank-v2",
                    "rerank_base_url": "https://dashscope.aliyuncs.com",
                    "rerank_api_key": "test-rerank-secret",
                    "langfuse_public_key": "pk-lf",
                    "langfuse_secret_key": "test-lf-secret",
                    "langfuse_host": "https://cloud.langfuse.com",
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["model_group"], "openai")
            self.assertEqual(payload["answer_api_mode"], "chat_completions")
            self.assertEqual(payload["answer_model"], "gpt-4.1-mini")
            self.assertEqual(payload["openai_base_url"], "https://gateway.example.com/v1")
            self.assertEqual(payload["embedding_base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(payload["answer_prompt_template"], "Answer using context: {question} {context}")
            self.assertEqual(payload["rewrite_prompt_template"], "Rewrite query: {question}")
            self.assertEqual(payload["rerank_provider"], "dashscope")
            self.assertEqual(payload["rerank_model"], "gte-rerank-v2")
            self.assertEqual(payload["rerank_base_url"], "https://dashscope.aliyuncs.com")
            self.assertTrue(payload["openai_configured"])
            self.assertTrue(payload["embedding_configured"])
            self.assertTrue(payload["rerank_configured"])
            self.assertTrue(payload["langfuse_configured"])
            self.assertNotIn("test-openai-secret", json.dumps(payload))
            self.assertNotIn("test-embedding-secret", json.dumps(payload))
            self.assertNotIn("test-rerank-secret", json.dumps(payload))

    def test_runtime_config_persists_across_app_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            first_client = TestClient(build_app(settings))

            first_client.put(
                "/config",
                json={
                    "model_group": "openai",
                    "answer_provider": "openai",
                    "embedding_provider": "openai",
                    "answer_api_mode": "chat_completions",
                    "answer_model": "gpt-4.1-mini",
                    "embedding_model": "text-embedding-3-small",
                    "openai_base_url": "https://gateway.example.com/v1",
                    "openai_api_key": "test-openai-secret",
                    "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "embedding_api_key": "test-embedding-secret",
                },
            )

            second_client = TestClient(build_app(settings))
            payload = second_client.get("/config").json()

            self.assertEqual(payload["model_group"], "openai")
            self.assertEqual(payload["answer_provider"], "openai")
            self.assertEqual(payload["answer_api_mode"], "chat_completions")
            self.assertEqual(payload["openai_base_url"], "https://gateway.example.com/v1")
            self.assertEqual(payload["embedding_base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertTrue(payload["openai_configured"])
            self.assertTrue(payload["embedding_configured"])

    def test_observability_status_reports_local_fallback_without_langfuse_client(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            response = client.get("/observability/status")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "local")
            self.assertFalse(response.json()["healthy"])

    def test_document_management_routes_list_chunks_delete_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)
            client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )

            documents = client.get("/documents")
            chunks = client.get("/documents/sample-handbook.md/chunks")
            delete_one = client.delete("/documents/sample-handbook.md")
            after_delete = client.get("/documents")
            client.post(
                "/documents",
                files={"file": ("sample-handbook.md", b"# Finance\nReceipts are required for expenses.")},
            )
            clear = client.delete("/documents")

            self.assertEqual(documents.status_code, 200)
            self.assertEqual(documents.json()[0]["source"], "sample-handbook.md")
            self.assertEqual(chunks.status_code, 200)
            self.assertEqual(chunks.json()[0]["source"], "sample-handbook.md")
            self.assertEqual(delete_one.status_code, 200)
            self.assertEqual(after_delete.json(), [])
            self.assertEqual(clear.status_code, 200)
            self.assertEqual(clear.json()["deleted"], "all")

    def test_evaluation_set_management_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(Settings(project_root=Path(directory)))
            client = TestClient(app)

            create = client.post(
                "/evaluation-sets",
                json={
                    "name": "custom.json",
                    "items": [
                        {
                            "question": "What do expenses need?",
                            "expected_terms": ["receipt"],
                            "required_source": "sample-handbook.md",
                        }
                    ],
                },
            )
            listing = client.get("/evaluation-sets")
            detail = client.get("/evaluation-sets/custom.json")
            delete = client.delete("/evaluation-sets/custom.json")

            self.assertEqual(create.status_code, 200)
            self.assertEqual(listing.json()[0]["name"], "custom.json")
            self.assertEqual(detail.json()[0]["question"], "What do expenses need?")
            self.assertEqual(delete.json()["deleted"], "custom.json")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

from app.schemas.models import Citation, RetrievedChunk
from app.services.model_providers import (
    DashScopeRerankProvider,
    LocalAnswerProvider,
    LocalEmbeddingProvider,
    OpenAIAnswerProvider,
    OpenAIEmbeddingProvider,
    OpenAIQueryRewriteProvider,
    openai_api_url,
    post_openai_json,
)


class TestModelProviders(unittest.TestCase):
    def test_local_answer_provider_composes_cited_snippets(self):
        provider = LocalAnswerProvider()
        answer = provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        self.assertIn("Receipts are required.", answer.text)
        self.assertIn("[c1]", answer.text)
        self.assertEqual(answer.provider, "local")

    def test_local_embedding_provider_returns_stable_vectors(self):
        provider = LocalEmbeddingProvider(dimensions=16)

        first = provider.embed_texts(["Receipts are required."])[0]
        second = provider.embed_texts(["Receipts are required."])[0]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertGreater(sum(abs(value) for value in first), 0)

    def test_openai_answer_provider_uses_responses_payload(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"output_text": "Use receipts with approval."}

        provider = OpenAIAnswerProvider(
            api_key="test-api-key",
            model="gpt-4.1-mini",
            base_url="https://gateway.example.com",
            post_json=fake_post,
        )
        answer = provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        self.assertEqual(answer.text, "Use receipts with approval.")
        self.assertEqual(requests[0][0], "https://gateway.example.com")
        self.assertEqual(requests[0][1], "/v1/responses")
        self.assertEqual(requests[0][2]["model"], "gpt-4.1-mini")
        self.assertIn("Receipts are required.", requests[0][2]["input"])

    def test_openai_answer_provider_can_use_chat_completions_payload(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"choices": [{"message": {"content": "Use receipts with approval."}}]}

        provider = OpenAIAnswerProvider(
            api_key="test-api-key",
            model="gpt-4.1-mini",
            base_url="https://gateway.example.com/v1",
            api_mode="chat_completions",
            post_json=fake_post,
        )
        answer = provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        self.assertEqual(answer.text, "Use receipts with approval.")
        self.assertEqual(requests[0][0], "https://gateway.example.com/v1")
        self.assertEqual(requests[0][1], "/v1/chat/completions")
        self.assertEqual(requests[0][2]["model"], "gpt-4.1-mini")
        self.assertEqual(requests[0][2]["messages"][0]["role"], "user")
        self.assertIn("Receipts are required.", requests[0][2]["messages"][0]["content"])

    def test_openai_answer_provider_auto_falls_back_to_chat_completions(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            if path == "/v1/responses":
                raise RuntimeError("Responses endpoint is not available")
            return {"choices": [{"message": {"content": "Fallback worked."}}]}

        provider = OpenAIAnswerProvider(
            api_key="test-api-key",
            model="gpt-5.5",
            base_url="https://gateway.example.com/v1",
            api_mode="auto",
            post_json=fake_post,
        )
        answer = provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        self.assertEqual(answer.text, "Fallback worked.")
        self.assertEqual([request[1] for request in requests], ["/v1/responses", "/v1/chat/completions"])

    def test_openai_answer_provider_omits_temperature_for_gateway_compatibility(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"output_text": "Use receipts with approval."}

        provider = OpenAIAnswerProvider(
            api_key="test-api-key",
            model="gpt-5.5",
            base_url="https://gateway.example.com/v1",
            api_mode="responses",
            post_json=fake_post,
        )
        provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        self.assertNotIn("temperature", requests[0][2])

    def test_openai_embedding_provider_parses_vectors(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

        provider = OpenAIEmbeddingProvider(
            api_key="test-api-key",
            model="text-embedding-3-small",
            base_url="https://gateway.example.com",
            post_json=fake_post,
        )

        self.assertEqual(provider.embed_texts(["a", "b"]), [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(requests[0][0], "https://gateway.example.com")
        self.assertEqual(requests[0][1], "/v1/embeddings")
        self.assertEqual(requests[0][2]["model"], "text-embedding-3-small")
        self.assertEqual(requests[0][3], "test-api-key")

    def test_openai_query_rewrite_provider_returns_clean_query(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"choices": [{"message": {"content": "毕业院校 教育经历 学校"}}]}

        provider = OpenAIQueryRewriteProvider(
            api_key="test-api-key",
            model="gpt-5.4-mini",
            base_url="https://gateway.example.com/v1",
            api_mode="chat_completions",
            post_json=fake_post,
        )

        rewritten = provider.rewrite("他学校哪的？")

        self.assertEqual(rewritten, "毕业院校 教育经历 学校")
        self.assertEqual(requests[0][1], "/v1/chat/completions")
        self.assertEqual(requests[0][2]["model"], "gpt-5.4-mini")
        self.assertIn("rewrite", requests[0][2]["messages"][0]["content"].lower())

    def test_openai_answer_provider_uses_custom_prompt_template(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"choices": [{"message": {"content": "Custom prompt worked."}}]}

        provider = OpenAIAnswerProvider(
            api_key="test-api-key",
            model="gpt-5.4-mini",
            base_url="https://gateway.example.com/v1",
            api_mode="chat_completions",
            answer_prompt_template="Answer from CONTEXT only.\nQ={question}\nCTX={context}",
            post_json=fake_post,
        )

        provider.generate(
            "What is required?",
            [Citation(chunk_id="c1", source="policy.md", snippet="Receipts are required.", score=1.2)],
        )

        prompt = requests[0][2]["messages"][0]["content"]
        self.assertIn("Answer from CONTEXT only.", prompt)
        self.assertIn("Q=What is required?", prompt)
        self.assertIn("Receipts are required.", prompt)

    def test_openai_query_rewrite_provider_uses_custom_prompt_template(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {"choices": [{"message": {"content": "education school"}}]}

        provider = OpenAIQueryRewriteProvider(
            api_key="test-api-key",
            model="gpt-5.4-mini",
            base_url="https://gateway.example.com/v1",
            api_mode="chat_completions",
            rewrite_prompt_template="Search query only: {question}",
            post_json=fake_post,
        )

        self.assertEqual(provider.rewrite("school?"), "education school")
        self.assertIn("Search query only: school?", requests[0][2]["messages"][0]["content"])

    def test_dashscope_rerank_provider_orders_chunks_by_relevance(self):
        requests = []

        def fake_post(base_url, path, payload, api_key):
            requests.append((base_url, path, payload, api_key))
            return {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.92},
                        {"index": 0, "relevance_score": 0.31},
                    ]
                }
            }

        provider = DashScopeRerankProvider(
            api_key="test-api-key",
            model="gte-rerank-v2",
            base_url="https://dashscope.aliyuncs.com",
            post_json=fake_post,
        )
        chunks = [
            RetrievedChunk(id="a", source="a.md", text="Alpha", score=0.1),
            RetrievedChunk(id="b", source="b.md", text="Beta", score=0.2),
        ]

        ranked = provider.rerank("query", chunks, top_n=2)

        self.assertEqual([item.id for item in ranked], ["b", "a"])
        self.assertEqual(ranked[0].score, 0.92)
        self.assertEqual(requests[0][1], "/api/v1/services/rerank/text-rerank/text-rerank")
        self.assertEqual(requests[0][2]["model"], "gte-rerank-v2")
        self.assertEqual(requests[0][2]["input"]["query"], "query")

    def test_openai_api_url_accepts_base_urls_with_or_without_v1(self):
        self.assertEqual(
            openai_api_url("https://gateway.example.com", "/v1/responses"),
            "https://gateway.example.com/v1/responses",
        )
        self.assertEqual(
            openai_api_url("https://gateway.example.com/v1", "/v1/responses"),
            "https://gateway.example.com/v1/responses",
        )

    def test_openai_api_url_respects_provider_base_urls_that_already_include_versioned_paths(self):
        self.assertEqual(
            openai_api_url("https://qianfan.baidubce.com/v2/coding", "/v1/chat/completions"),
            "https://qianfan.baidubce.com/v2/coding/chat/completions",
        )
        self.assertEqual(
            openai_api_url("https://qianfan.baidubce.com/v2/coding", "/v1/embeddings"),
            "https://qianfan.baidubce.com/v2/coding/embeddings",
        )

    def test_post_openai_json_sends_openai_compatible_headers(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"ok": true}'

        def fake_urlopen(request, timeout):
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["timeout"] = timeout
            return FakeResponse()

        api_key = "test-api-key"
        with patch("urllib.request.urlopen", fake_urlopen):
            response = post_openai_json(
                "https://gateway.example.com/v1",
                "/v1/responses",
                {"model": "gpt-5.4-mini", "input": "hi"},
                api_key,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(captured["headers"]["authorization"], "Bearer " + api_key)
        self.assertEqual(captured["headers"]["accept"], "application/json")
        self.assertIn("openai", captured["headers"]["user-agent"].lower())
        self.assertEqual(captured["timeout"], 60)


if __name__ == "__main__":
    unittest.main()

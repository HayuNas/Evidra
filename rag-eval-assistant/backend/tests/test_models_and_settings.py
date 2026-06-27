import tempfile
import unittest
from pathlib import Path

from app.schemas.models import AnswerResult, Citation, RetrievedChunk
from app.settings import Settings
from app.schemas.models import ChunkingConfig, RagStrategyConfig, RetrievalConfig


class TestModelsAndSettings(unittest.TestCase):
    def test_strategy_configs_have_safe_defaults(self):
        strategy = RagStrategyConfig()

        self.assertEqual(strategy.chunking.strategy, "heading")
        self.assertEqual(strategy.chunking.chunk_size, 900)
        self.assertEqual(strategy.chunking.overlap, 0)
        self.assertEqual(strategy.retrieval.mode, "tfidf")
        self.assertEqual(strategy.retrieval.top_k, 4)
        self.assertEqual(strategy.retrieval.score_threshold, 0.0)

    def test_strategy_configs_normalize_invalid_values(self):
        strategy = RagStrategyConfig(
            chunking=ChunkingConfig(strategy="unknown", chunk_size=10, overlap=9999),
            retrieval=RetrievalConfig(mode="unknown", top_k=99, score_threshold=-4),
        )

        self.assertEqual(strategy.chunking.strategy, "heading")
        self.assertEqual(strategy.chunking.chunk_size, 300)
        self.assertEqual(strategy.chunking.overlap, 299)
        self.assertEqual(strategy.retrieval.mode, "tfidf")
        self.assertEqual(strategy.retrieval.top_k, 10)
        self.assertEqual(strategy.retrieval.score_threshold, 0.0)

    def test_default_settings_keep_data_under_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(project_root=root)

            self.assertEqual(settings.upload_dir, root / "data" / "uploads")
            self.assertEqual(settings.index_dir, root / "data" / "indexes")
            self.assertEqual(settings.eval_sets_dir, root / "data" / "eval_sets")
            self.assertEqual(settings.max_upload_bytes, 5 * 1024 * 1024)

    def test_answer_result_serializes_nested_citations(self):
        chunk = RetrievedChunk(
            id="handbook.md:0001",
            source="handbook.md",
            text="Expense reports must include receipts.",
            score=0.82,
            heading="Finance",
            position=1,
        )
        answer = AnswerResult(
            question="What do expense reports need?",
            answer="Expense reports must include receipts. [handbook.md:0001]",
            citations=[Citation.from_chunk(chunk)],
            confidence=0.82,
            trace_id="trace-123",
        )

        payload = answer.to_dict()

        self.assertEqual(payload["question"], "What do expense reports need?")
        self.assertEqual(payload["citations"][0]["source"], "handbook.md")
        self.assertEqual(payload["citations"][0]["snippet"], "Expense reports must include receipts.")
        self.assertEqual(payload["confidence"], 0.82)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from app.evals.runner import EvaluationRunner
from app.rag.pipeline import RagPipelineFactory
from app.rag.store import LocalDocumentStore
from app.schemas.models import DocumentChunk, RetrievalConfig
from app.services.langfuse_trace import LangfuseTraceService
from app.settings import Settings


class TestEvaluationRunner(unittest.TestCase):
    def test_runner_loads_eval_set_and_computes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.eval_sets_dir.mkdir(parents=True)
            (settings.eval_sets_dir / "demo.json").write_text(
                json.dumps(
                    [
                        {
                            "question": "What do expenses need?",
                            "expected_terms": ["Receipts"],
                            "required_source": "handbook.md",
                        },
                        {"question": "Missing thing?", "expected_terms": ["never"]},
                    ]
                ),
                encoding="utf-8",
            )
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.")],
            )
            pipeline = RagPipelineFactory(store, LangfuseTraceService(settings)).create()
            runner = EvaluationRunner(settings, pipeline)

            summary = runner.run("demo.json")

            self.assertEqual(summary.total, 2)
            self.assertEqual(summary.passed, 1)
            self.assertEqual(summary.citation_coverage, 0.5)
            self.assertGreaterEqual(summary.average_latency_ms, 0)
            self.assertFalse(summary.results[1].passed)

    def test_runner_passes_retrieval_config_to_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.eval_sets_dir.mkdir(parents=True)
            (settings.eval_sets_dir / "demo.json").write_text(
                json.dumps([{"question": "What do expenses need?", "expected_terms": ["Receipts"]}]),
                encoding="utf-8",
            )
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.")],
            )
            runner = EvaluationRunner(
                settings,
                RagPipelineFactory(store, LangfuseTraceService(settings)).create(),
            )

            summary = runner.run("demo.json", RetrievalConfig(mode="bm25", top_k=1), prompt_version="analyst")

            self.assertEqual(summary.results[0].answer.metadata["retrieval"]["mode"], "bm25")
            self.assertEqual(summary.results[0].answer.metadata["retrieval"]["top_k"], 1)
            self.assertEqual(summary.results[0].answer.metadata["prompt"]["version"], "analyst")

    def test_runner_records_evaluation_summary_to_trace_service(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.eval_sets_dir.mkdir(parents=True)
            (settings.eval_sets_dir / "demo.json").write_text(
                json.dumps([{"question": "What do expenses need?", "expected_terms": ["Receipts"]}]),
                encoding="utf-8",
            )
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses.")],
            )
            runner = EvaluationRunner(settings, RagPipelineFactory(store, LangfuseTraceService(settings)).create())

            runner.run("demo.json")

            events = [
                json.loads(line)
                for line in (settings.traces_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            evaluation_events = [event for event in events if event.get("type") == "evaluation"]
            self.assertEqual(evaluation_events[0]["summary"]["passed"], 1)

    def test_runner_compares_named_retrieval_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            settings.eval_sets_dir.mkdir(parents=True)
            (settings.eval_sets_dir / "demo.json").write_text(
                json.dumps([{"question": "How often should keys rotate?", "expected_terms": ["quarterly"]}]),
                encoding="utf-8",
            )
            store = LocalDocumentStore(settings.index_dir / "documents.json")
            store.replace_source(
                "handbook.md",
                [DocumentChunk("handbook.md:0001", "handbook.md", "Access keys should rotate quarterly.")],
            )
            runner = EvaluationRunner(
                settings,
                RagPipelineFactory(store, LangfuseTraceService(settings)).create(),
            )

            comparisons = runner.compare(
                "demo.json",
                {
                    "tfidf-k2": RetrievalConfig(mode="tfidf", top_k=2),
                    "bm25-k1": {"retrieval": RetrievalConfig(mode="bm25", top_k=1), "prompt_version": "concise"},
                },
            )

            self.assertEqual([comparison["name"] for comparison in comparisons], ["tfidf-k2", "bm25-k1"])
            self.assertEqual(comparisons[1]["retrieval"]["mode"], "bm25")
            self.assertEqual(comparisons[1]["prompt"]["version"], "concise")
            self.assertEqual(comparisons[1]["summary"]["total"], 1)

    def test_runner_rejects_eval_set_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            runner = EvaluationRunner(
                settings,
                RagPipelineFactory(LocalDocumentStore(settings.index_dir / "documents.json"), LangfuseTraceService(settings)).create(),
            )

            with self.assertRaisesRegex(ValueError, "Evaluation set name"):
                runner.run("../secret.json")


if __name__ == "__main__":
    unittest.main()

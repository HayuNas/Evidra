from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.rag.pipeline import RagPipeline
from app.schemas.models import EvaluationItem, EvaluationResult, EvaluationSummary, PromptConfig, RetrievalConfig
from app.settings import Settings


class EvaluationRunner:
    def __init__(self, settings: Settings, pipeline: RagPipeline):
        self.settings = settings
        self.pipeline = pipeline

    def run(
        self,
        eval_set_name: str,
        retrieval: RetrievalConfig | None = None,
        prompt_version: str = "grounded",
        rewrite_query: bool = False,
        multi_route: bool = False,
        rerank: bool = False,
    ) -> EvaluationSummary:
        items = self._load_items(eval_set_name)
        prompt = PromptConfig(prompt_version)
        results: list[EvaluationResult] = []
        for item in items:
            started = time.perf_counter()
            error = None
            try:
                answer = self.pipeline.answer(
                    item.question,
                    retrieval,
                    prompt.version,
                    rewrite_query=rewrite_query,
                    multi_route=multi_route,
                    rerank=rerank,
                )
                passed = self._passes(item, answer.answer, [citation.source for citation in answer.citations])
            except Exception as exc:
                error = str(exc)
                answer = self.pipeline.answer("", retrieval, prompt.version, rewrite_query=False)
                passed = False
            latency_ms = (time.perf_counter() - started) * 1000
            results.append(EvaluationResult(item=item, answer=answer, passed=passed, latency_ms=latency_ms, error=error))
        citation_coverage = sum(1 for result in results if result.answer.citations) / len(results) if results else 0.0
        average_latency_ms = sum(result.latency_ms for result in results) / len(results) if results else 0.0
        summary = EvaluationSummary(
            total=len(results),
            passed=sum(1 for result in results if result.passed),
            citation_coverage=round(citation_coverage, 3),
            average_latency_ms=round(average_latency_ms, 2),
            results=results,
        )
        event_id = self.pipeline.trace_service.record_evaluation_summary(eval_set_name, summary.to_dict())
        self.pipeline.trace_service.record_score(event_id, "pass_rate", summary.to_dict()["pass_rate"], "evaluation summary")
        self.pipeline.trace_service.record_score(event_id, "citation_coverage", summary.citation_coverage, "evaluation summary")
        return summary

    def compare(self, eval_set_name: str, strategies: dict[str, Any]) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for name, strategy in strategies.items():
            retrieval, prompt, rewrite_query, multi_route, rerank = self._strategy_parts(strategy)
            summary = self.run(
                eval_set_name,
                retrieval,
                prompt.version,
                rewrite_query=rewrite_query,
                multi_route=multi_route,
                rerank=rerank,
            )
            comparisons.append(
                {
                    "name": name,
                    "retrieval": retrieval.to_dict(),
                    "prompt": prompt.to_dict(),
                    "query": {"rewrite_enabled": rewrite_query, "multi_route": multi_route, "rerank": rerank},
                    "summary": summary.to_dict(),
                }
            )
        return comparisons

    def _strategy_parts(self, strategy: Any) -> tuple[RetrievalConfig, PromptConfig, bool, bool, bool]:
        if isinstance(strategy, RetrievalConfig):
            return strategy, PromptConfig(), False, False, False
        if isinstance(strategy, dict):
            retrieval = strategy.get("retrieval", RetrievalConfig())
            if isinstance(retrieval, dict):
                retrieval = RetrievalConfig(**retrieval)
            prompt = PromptConfig(strategy.get("prompt_version", "grounded"))
            return (
                retrieval,
                prompt,
                bool(strategy.get("rewrite_query", False)),
                bool(strategy.get("multi_route", False)),
                bool(strategy.get("rerank", False)),
            )
        return RetrievalConfig(), PromptConfig(), False, False, False

    def _load_items(self, eval_set_name: str) -> list[EvaluationItem]:
        clean_name = Path(eval_set_name).name
        if clean_name != eval_set_name or not clean_name.endswith(".json"):
            raise ValueError("Evaluation set name must be a .json filename.")
        path = self.settings.eval_sets_dir / eval_set_name
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        return [
            EvaluationItem(
                question=item["question"],
                expected_terms=item.get("expected_terms", []),
                required_source=item.get("required_source"),
            )
            for item in raw_items
        ]

    def _passes(self, item: EvaluationItem, answer: str, citation_sources: list[str]) -> bool:
        lowered_answer = answer.lower()
        terms_present = all(term.lower() in lowered_answer for term in item.expected_terms)
        source_present = item.required_source is None or item.required_source in citation_sources
        return terms_present and source_present

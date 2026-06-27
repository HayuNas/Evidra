from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.settings import Settings


class LangfuseTraceService:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client

    def status(self) -> dict[str, Any]:
        configured = bool(self.settings.langfuse_public_key and self.settings.langfuse_secret_key)
        if self.client is None:
            return {"configured": configured, "healthy": False, "mode": "local", "host": self.settings.langfuse_host}
        try:
            healthy = bool(self.client.auth_check()) if hasattr(self.client, "auth_check") else True
        except Exception:
            healthy = False
        return {
            "configured": configured,
            "healthy": healthy,
            "mode": "langfuse" if healthy else "local",
            "host": self.settings.langfuse_host,
        }

    def record_question(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        if self.client is not None:
            try:
                trace = self.client.trace(name="rag-question", input=question, output=answer, metadata=event)
                return getattr(trace, "id", None) or f"langfuse-{uuid.uuid4().hex[:12]}"
            except Exception:
                pass
        return self._write_local(event)

    def record_score(self, trace_id: str, name: str, value: float, comment: str | None = None) -> None:
        event = {
            "type": "score",
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "comment": comment,
            "timestamp": time.time(),
        }
        if self.client is not None:
            try:
                self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)
                return
            except Exception:
                pass
        self._write_local(event)

    def record_feedback(self, trace_id: str, label: str, value: float, comment: str | None = None) -> str:
        feedback_id = f"local-{uuid.uuid4().hex[:12]}"
        event = {
            "type": "feedback",
            "feedback_id": feedback_id,
            "trace_id": trace_id,
            "label": label,
            "value": value,
            "comment": comment,
            "timestamp": time.time(),
        }
        if self.client is not None:
            try:
                self.client.score(trace_id=trace_id, name=label, value=value, comment=comment)
                event["mode"] = "langfuse"
            except Exception:
                event["mode"] = "local"
        self._write_local(event)
        return feedback_id

    def record_evaluation_summary(self, name: str, summary: dict[str, Any]) -> str:
        event = {
            "type": "evaluation",
            "name": name,
            "summary": summary,
            "timestamp": time.time(),
        }
        return self._write_local(event)

    def _write_local(self, event: dict[str, Any]) -> str:
        trace_id = event.get("trace_id") or f"local-{uuid.uuid4().hex[:12]}"
        event["trace_id"] = trace_id
        path = self.settings.traces_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return trace_id


def build_trace_service(settings: Settings) -> LangfuseTraceService:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return LangfuseTraceService(settings)
    try:
        from langfuse import Langfuse
    except ImportError:
        return LangfuseTraceService(settings)
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return LangfuseTraceService(settings, client=client)

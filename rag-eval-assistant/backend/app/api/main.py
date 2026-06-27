from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.evals.runner import EvaluationRunner
from app.rag.pipeline import RagPipelineFactory
from app.rag.store import LocalDocumentStore
from app.schemas.models import ChunkingConfig, RetrievalConfig
from app.services.ingestion import DocumentIngestionService
from app.services.langfuse_trace import build_trace_service
from app.services.model_providers import build_answer_provider, build_embedding_provider, build_query_rewrite_provider, build_rerank_provider
from app.services.runtime_config import RuntimeConfigService
from app.settings import Settings

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    File = None
    Form = None
    HTTPException = None
    UploadFile = None
    CORSMiddleware = None
    FileResponse = None
    StaticFiles = None
    BaseModel = object
    Field = None


class ChunkingRequest(BaseModel):
    strategy: str = "heading"
    chunk_size: int = 900
    overlap: int = 0

    def to_config(self) -> ChunkingConfig:
        return ChunkingConfig(strategy=self.strategy, chunk_size=self.chunk_size, overlap=self.overlap)


class RetrievalRequest(BaseModel):
    mode: str = "tfidf"
    top_k: int = 4
    score_threshold: float = 0.0

    def to_config(self) -> RetrievalConfig:
        return RetrievalConfig(mode=self.mode, top_k=self.top_k, score_threshold=self.score_threshold)


class AskRequest(BaseModel):
    question: str
    prompt_version: str = "grounded"
    rewrite_query: bool = False
    multi_route: bool = False
    rerank: bool = False
    retrieval: RetrievalRequest = Field(default_factory=RetrievalRequest)


class EvalRequest(BaseModel):
    eval_set: str = "demo.json"
    prompt_version: str = "grounded"
    rewrite_query: bool = False
    multi_route: bool = False
    rerank: bool = False
    retrieval: RetrievalRequest = Field(default_factory=RetrievalRequest)


class NamedStrategyRequest(BaseModel):
    name: str
    prompt_version: str = "grounded"
    rewrite_query: bool = False
    multi_route: bool = False
    rerank: bool = False
    retrieval: RetrievalRequest = Field(default_factory=RetrievalRequest)


class CompareEvalRequest(BaseModel):
    eval_set: str = "demo.json"
    strategies: list[NamedStrategyRequest] = Field(default_factory=list)


class EvaluationSetRequest(BaseModel):
    name: str
    items: list[dict[str, Any]]


class RuntimeConfigRequest(BaseModel):
    model_group: str | None = None
    answer_provider: str | None = None
    embedding_provider: str | None = None
    answer_api_mode: str | None = None
    answer_model: str | None = None
    embedding_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    answer_prompt_template: str | None = None
    rewrite_prompt_template: str | None = None
    rerank_provider: str | None = None
    rerank_model: str | None = None
    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    label: str = "helpful"
    value: float = 1.0
    comment: str | None = None


def build_app(settings: Settings | None = None):
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Run `uv sync` in the backend directory.")

    settings = settings or Settings.from_env()
    store = LocalDocumentStore(settings.index_dir / "documents.json")
    runtime_config = RuntimeConfigService(settings)

    def current_settings() -> Settings:
        config = runtime_config.config
        return replace(
            settings,
            openai_api_key=config.openai_api_key,
            answer_model=config.answer_model,
            embedding_model=config.embedding_model,
            embedding_api_key=config.embedding_api_key,
            embedding_base_url=config.embedding_base_url,
            langfuse_public_key=config.langfuse_public_key,
            langfuse_secret_key=config.langfuse_secret_key,
            langfuse_host=config.langfuse_host,
            openai_base_url=config.openai_base_url,
        )

    def current_trace_service():
        return build_trace_service(current_settings())
    def current_pipeline():
        config = runtime_config.config
        return RagPipelineFactory(
            store,
            current_trace_service(),
            top_k=settings.top_k,
            answer_provider=build_answer_provider(config),
            embedding_provider=build_embedding_provider(config),
            query_rewrite_provider=build_query_rewrite_provider(config),
            rerank_provider=build_rerank_provider(config),
        ).create()

    def current_ingestion():
        return DocumentIngestionService(
            settings,
            store,
            embedding_provider=build_embedding_provider(runtime_config.config),
        )

    def model_provider_error(exc: RuntimeError) -> HTTPException:
        return HTTPException(status_code=502, detail=f"Model provider request failed: {exc}")

    app = FastAPI(title="Evidra", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/config")
    def get_config():
        return runtime_config.config.to_public_dict()

    @app.put("/config")
    def update_config(request: RuntimeConfigRequest):
        return runtime_config.update(request.model_dump(exclude_unset=True)).to_public_dict()

    @app.get("/observability/status")
    def observability_status():
        return current_trace_service().status()

    @app.post("/feedback")
    def record_feedback(request: FeedbackRequest):
        if not request.trace_id:
            raise HTTPException(status_code=400, detail="trace_id is required.")
        value = min(1.0, max(0.0, float(request.value)))
        label = request.label if request.label in {"helpful", "not_helpful", "rating"} else "rating"
        feedback_id = current_trace_service().record_feedback(request.trace_id, label, value, request.comment)
        return {"recorded": True, "feedback_id": feedback_id, "trace_id": request.trace_id}

    @app.post("/documents")
    async def upload_document(
        file: UploadFile = File(...),
        chunk_strategy: str = Form("heading"),
        chunk_size: int = Form(900),
        chunk_overlap: int = Form(0),
    ):
        try:
            content = await file.read()
            chunking = ChunkingConfig(strategy=chunk_strategy, chunk_size=chunk_size, overlap=chunk_overlap)
            result = current_ingestion().ingest(file.filename or "upload.txt", content, chunking)
            return result.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise model_provider_error(exc) from exc

    @app.get("/documents")
    def list_documents():
        return store.list_documents()

    @app.get("/documents/{source}/chunks")
    def list_document_chunks(source: str):
        return [chunk.to_dict() for chunk in store.list_chunks(source)]

    @app.delete("/documents/{source}")
    def delete_document(source: str):
        store.delete_source(source)
        upload_path = settings.upload_dir / source
        if upload_path.exists():
            upload_path.unlink()
        return {"deleted": source}

    @app.delete("/documents")
    def clear_documents():
        store.clear()
        if settings.upload_dir.exists():
            for path in settings.upload_dir.iterdir():
                if path.is_file():
                    path.unlink()
        return {"deleted": "all"}

    def safe_eval_set_path(name: str) -> Path:
        clean_name = Path(name).name
        if clean_name != name or not clean_name.endswith(".json"):
            raise HTTPException(status_code=400, detail="Evaluation set name must be a .json filename.")
        return settings.eval_sets_dir / clean_name

    @app.get("/evaluation-sets")
    def list_evaluation_sets():
        settings.eval_sets_dir.mkdir(parents=True, exist_ok=True)
        sets = []
        for path in sorted(settings.eval_sets_dir.glob("*.json")):
            try:
                count = len(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                count = 0
            sets.append({"name": path.name, "item_count": count})
        return sets

    @app.get("/evaluation-sets/{name}")
    def get_evaluation_set(name: str):
        path = safe_eval_set_path(name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Evaluation set not found.")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/evaluation-sets")
    def save_evaluation_set(request: EvaluationSetRequest):
        path = safe_eval_set_path(request.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        for item in request.items:
            if "question" not in item:
                raise HTTPException(status_code=400, detail="Each evaluation item needs a question.")
        path.write_text(json.dumps(request.items, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"name": path.name, "item_count": len(request.items)}

    @app.delete("/evaluation-sets/{name}")
    def delete_evaluation_set(name: str):
        path = safe_eval_set_path(name)
        if path.exists():
            path.unlink()
        return {"deleted": path.name}

    @app.post("/ask")
    def ask(request: AskRequest):
        try:
            return current_pipeline().answer(
                request.question,
                request.retrieval.to_config(),
                request.prompt_version,
                rewrite_query=request.rewrite_query,
                multi_route=request.multi_route,
                rerank=request.rerank,
            ).to_dict()
        except RuntimeError as exc:
            raise model_provider_error(exc) from exc

    @app.post("/evaluations/run")
    def run_evaluation(request: EvalRequest):
        try:
            return EvaluationRunner(settings, current_pipeline()).run(
                request.eval_set,
                request.retrieval.to_config(),
                request.prompt_version,
                rewrite_query=request.rewrite_query,
                multi_route=request.multi_route,
                rerank=request.rerank,
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Evaluation set not found.") from exc
        except RuntimeError as exc:
            raise model_provider_error(exc) from exc

    @app.post("/evaluations/compare")
    def compare_evaluations(request: CompareEvalRequest):
        strategies = {
            strategy.name: {
                "retrieval": strategy.retrieval.to_config(),
                "prompt_version": strategy.prompt_version,
                "rewrite_query": strategy.rewrite_query,
                "multi_route": strategy.multi_route,
                "rerank": strategy.rerank,
            }
            for strategy in request.strategies
        }
        if not strategies:
            strategies = {
                "tfidf-k2": {"retrieval": RetrievalConfig(mode="tfidf", top_k=2), "prompt_version": "grounded", "rewrite_query": False},
                "bm25-k4": {"retrieval": RetrievalConfig(mode="bm25", top_k=4), "prompt_version": "grounded", "rewrite_query": False},
            }
        try:
            return {
                "eval_set": request.eval_set,
                "comparisons": EvaluationRunner(settings, current_pipeline()).compare(request.eval_set, strategies),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Evaluation set not found.") from exc
        except RuntimeError as exc:
            raise model_provider_error(exc) from exc

    frontend_dir = settings.project_root / "frontend"
    index_path = frontend_dir / "index.html"
    if frontend_dir.exists() and index_path.exists():
        @app.get("/", include_in_schema=False)
        def frontend_index():
            return FileResponse(index_path)

        app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")

    return app


app = build_app() if FastAPI is not None else None

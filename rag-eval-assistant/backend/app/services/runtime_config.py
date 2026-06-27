from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from app.settings import Settings


@dataclass(frozen=True)
class RuntimeConfig:
    model_group: str = "local"
    answer_provider: str = "local"
    embedding_provider: str = "local"
    answer_model: str = "local-citation-composer"
    embedding_model: str = "local-hash-embedding"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    answer_api_mode: str = "auto"
    answer_prompt_template: str | None = None
    rewrite_prompt_template: str | None = None
    rerank_provider: str = "none"
    rerank_model: str = "gte-rerank-v2"
    rerank_api_key: str | None = None
    rerank_base_url: str = "https://dashscope.aliyuncs.com"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "RuntimeConfig":
        has_openai = bool(settings.openai_api_key)
        return cls(
            model_group="openai" if has_openai else "local",
            answer_provider="openai" if has_openai else "local",
            embedding_provider="openai" if has_openai else "local",
            answer_model=settings.answer_model if has_openai else "local-citation-composer",
            embedding_model=settings.embedding_model if has_openai else "local-hash-embedding",
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            answer_api_mode=settings.answer_api_mode,
            rerank_api_key=None,
            langfuse_public_key=settings.langfuse_public_key,
            langfuse_secret_key=settings.langfuse_secret_key,
            langfuse_host=settings.langfuse_host,
        )

    def updated(self, values: dict[str, Any]) -> "RuntimeConfig":
        allowed = {
            "model_group",
            "answer_provider",
            "embedding_provider",
            "answer_model",
            "embedding_model",
            "openai_api_key",
            "openai_base_url",
            "embedding_api_key",
            "embedding_base_url",
            "answer_api_mode",
            "answer_prompt_template",
            "rewrite_prompt_template",
            "rerank_provider",
            "rerank_model",
            "rerank_api_key",
            "rerank_base_url",
            "langfuse_public_key",
            "langfuse_secret_key",
            "langfuse_host",
        }
        clean = {key: value for key, value in values.items() if key in allowed and value not in (None, "")}
        next_config = replace(self, **clean)
        return next_config.normalized()

    def normalized(self) -> "RuntimeConfig":
        model_group = self.model_group if self.model_group in {"local", "openai", "hybrid"} else "local"
        answer_provider = self.answer_provider if self.answer_provider in {"local", "openai"} else "local"
        embedding_provider = self.embedding_provider if self.embedding_provider in {"local", "openai"} else "local"
        answer_api_mode = self.answer_api_mode if self.answer_api_mode in {"auto", "responses", "chat_completions"} else "auto"
        rerank_provider = self.rerank_provider if self.rerank_provider in {"none", "dashscope"} else "none"
        if model_group == "local":
            answer_provider = "local"
            embedding_provider = "local"
        if model_group == "openai":
            answer_provider = "openai"
            embedding_provider = "openai"
        return replace(
            self,
            model_group=model_group,
            answer_provider=answer_provider,
            embedding_provider=embedding_provider,
            answer_api_mode=answer_api_mode,
            answer_model=self.answer_model or "local-citation-composer",
            embedding_model=self.embedding_model or "local-hash-embedding",
            embedding_base_url=self.embedding_base_url or self.openai_base_url,
            answer_prompt_template=self.answer_prompt_template,
            rewrite_prompt_template=self.rewrite_prompt_template,
            rerank_provider=rerank_provider,
            rerank_model=self.rerank_model or "gte-rerank-v2",
            rerank_base_url=self.rerank_base_url or "https://dashscope.aliyuncs.com",
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "model_group": self.model_group,
            "answer_provider": self.answer_provider,
            "embedding_provider": self.embedding_provider,
            "answer_api_mode": self.answer_api_mode,
            "answer_model": self.answer_model,
            "embedding_model": self.embedding_model,
            "openai_base_url": self.openai_base_url,
            "embedding_base_url": self.embedding_base_url or self.openai_base_url,
            "answer_prompt_template": self.answer_prompt_template,
            "rewrite_prompt_template": self.rewrite_prompt_template,
            "rerank_provider": self.rerank_provider,
            "rerank_model": self.rerank_model,
            "rerank_base_url": self.rerank_base_url,
            "openai_configured": bool(self.openai_api_key),
            "embedding_configured": bool(self.embedding_api_key or self.openai_api_key),
            "rerank_configured": bool(self.rerank_api_key),
            "langfuse_configured": bool(self.langfuse_public_key and self.langfuse_secret_key),
            "langfuse_host": self.langfuse_host,
        }


class RuntimeConfigService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._path = settings.data_dir / "runtime_config.json"
        self._config = self._load_config()

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    def update(self, values: dict[str, Any]) -> RuntimeConfig:
        self._config = self._config.updated(values)
        self._save_config()
        return self._config

    def _load_config(self) -> RuntimeConfig:
        default = RuntimeConfig.from_settings(self._settings)
        if not self._path.exists():
            return default
        try:
            values = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        if not isinstance(values, dict):
            return default
        return default.updated(values)

    def _save_config(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._config.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

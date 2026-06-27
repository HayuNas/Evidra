from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = _default_root()
    max_upload_bytes: int = 5 * 1024 * 1024
    top_k: int = 4
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    answer_api_mode: str = "auto"
    answer_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        return cls(
            project_root=project_root or _default_root(),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            embedding_api_key=os.getenv("OPENAI_EMBEDDING_API_KEY"),
            embedding_base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL"),
            answer_api_mode=os.getenv("OPENAI_ANSWER_API_MODE", "auto"),
            answer_model=os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def eval_sets_dir(self) -> Path:
        return self.data_dir / "eval_sets"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

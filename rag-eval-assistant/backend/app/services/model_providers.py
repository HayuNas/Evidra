from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from app.schemas.models import Citation, RetrievedChunk
from app.services.runtime_config import RuntimeConfig


PostJson = Callable[[str, str, dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    provider: str
    model: str


class AnswerProvider(Protocol):
    def generate(self, question: str, citations: list[Citation], prompt_version: str = "grounded") -> GeneratedAnswer:
        ...


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class QueryRewriteProvider(Protocol):
    provider: str
    model: str

    def rewrite(self, question: str) -> str:
        ...


class RerankProvider(Protocol):
    provider: str
    model: str

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        ...


class LocalAnswerProvider:
    provider = "local"
    model = "local-citation-composer"

    def generate(self, question: str, citations: list[Citation], prompt_version: str = "grounded") -> GeneratedAnswer:
        if not citations:
            return GeneratedAnswer(text="", provider=self.provider, model=self.model)
        lines = [f"{citation.snippet} [{citation.chunk_id}]" for citation in citations]
        if prompt_version == "concise":
            lines = lines[:1]
        if prompt_version == "analyst":
            lines = [f"Evidence {index + 1}: {line}" for index, line in enumerate(lines)]
        return GeneratedAnswer(text="\n".join(lines), provider=self.provider, model=self.model)


class LocalEmbeddingProvider:
    provider = "local"
    model = "local-hash-embedding"

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimensions)]
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]


class OpenAIAnswerProvider:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        api_mode: str = "responses",
        answer_prompt_template: str | None = None,
        post_json: PostJson | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode if api_mode in {"auto", "responses", "chat_completions"} else "auto"
        self.answer_prompt_template = answer_prompt_template
        self._post_json = post_json or post_openai_json

    def generate(self, question: str, citations: list[Citation], prompt_version: str = "grounded") -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{citation.chunk_id}] Source: {citation.source}\n{citation.snippet}"
            for citation in citations
        )
        prompt = build_answer_prompt(question, context, prompt_version, self.answer_prompt_template)
        if self.api_mode == "chat_completions":
            return self._generate_chat_completion(prompt)
        if self.api_mode == "auto":
            try:
                return self._generate_response(prompt)
            except RuntimeError:
                return self._generate_chat_completion(prompt)
        return self._generate_response(prompt)

    def _generate_response(self, prompt: str) -> GeneratedAnswer:
        payload = {
            "model": self.model,
            "input": prompt,
        }
        response = self._post_json(self.base_url, "/v1/responses", payload, self.api_key)
        text = response.get("output_text") or self._extract_response_text(response)
        return GeneratedAnswer(text=text.strip(), provider=self.provider, model=self.model)

    def _generate_chat_completion(self, prompt: str) -> GeneratedAnswer:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._post_json(self.base_url, "/v1/chat/completions", payload, self.api_key)
        text = self._extract_chat_completion_text(response)
        return GeneratedAnswer(text=text.strip(), provider=self.provider, model=self.model)

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        parts: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _extract_chat_completion_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
            return "\n".join(parts)
        return str(content)


class OpenAIEmbeddingProvider:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        post_json: PostJson | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post_json = post_json or post_openai_json

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post_json(
            self.base_url,
            "/v1/embeddings",
            {"model": self.model, "input": texts},
            self.api_key,
        )
        return [item["embedding"] for item in response.get("data", [])]


class OpenAIQueryRewriteProvider:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com",
        api_mode: str = "chat_completions",
        rewrite_prompt_template: str | None = None,
        post_json: PostJson | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode if api_mode in {"auto", "responses", "chat_completions"} else "chat_completions"
        self.rewrite_prompt_template = rewrite_prompt_template
        self._post_json = post_json or post_openai_json

    def rewrite(self, question: str) -> str:
        prompt = build_query_rewrite_prompt(question, self.rewrite_prompt_template)
        if self.api_mode == "responses":
            return self._rewrite_response(prompt)
        if self.api_mode == "auto":
            try:
                return self._rewrite_chat_completion(prompt)
            except RuntimeError:
                return self._rewrite_response(prompt)
        return self._rewrite_chat_completion(prompt)

    def _rewrite_chat_completion(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._post_json(self.base_url, "/v1/chat/completions", payload, self.api_key)
        return extract_chat_completion_text(response).strip()

    def _rewrite_response(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "input": prompt,
        }
        response = self._post_json(self.base_url, "/v1/responses", payload, self.api_key)
        text = response.get("output_text") or extract_response_text(response)
        return str(text).strip()


class DashScopeRerankProvider:
    provider = "dashscope"

    def __init__(
        self,
        api_key: str,
        model: str = "gte-rerank-v2",
        base_url: str = "https://dashscope.aliyuncs.com",
        post_json: PostJson | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post_json = post_json or post_dashscope_json

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [chunk.text for chunk in chunks],
            },
            "parameters": {
                "return_documents": False,
                "top_n": min(max(1, top_n), len(chunks)),
            },
        }
        response = self._post_json(
            self.base_url,
            "/api/v1/services/rerank/text-rerank/text-rerank",
            payload,
            self.api_key,
        )
        results = response.get("output", {}).get("results", [])
        ranked: list[RetrievedChunk] = []
        seen: set[int] = set()
        for item in results:
            index = int(item.get("index", -1))
            if index < 0 or index >= len(chunks) or index in seen:
                continue
            seen.add(index)
            score = float(item.get("relevance_score", chunks[index].score))
            chunk = chunks[index]
            ranked.append(
                RetrievedChunk(
                    id=chunk.id,
                    source=chunk.source,
                    text=chunk.text,
                    heading=chunk.heading,
                    position=chunk.position,
                    embedding=chunk.embedding,
                    score=round(score, 4),
                )
            )
        for index, chunk in enumerate(chunks):
            if index not in seen:
                ranked.append(chunk)
        return ranked[:top_n]


def openai_api_url(base_url: str, path: str) -> str:
    clean_base = base_url.rstrip("/")
    clean_path = path if path.startswith("/") else f"/{path}"
    if clean_base.endswith("/v1") and clean_path.startswith("/v1/"):
        return f"{clean_base}{clean_path[3:]}"
    base_path = urllib.request.urlparse(clean_base).path
    if clean_path.startswith("/v1/") and re.search(r"(^|/)v\d+($|/)", base_path):
        return f"{clean_base}{clean_path[3:]}"
    return f"{clean_base}{clean_path}"


def build_answer_prompt(
    question: str,
    context: str,
    prompt_version: str = "grounded",
    custom_template: str | None = None,
) -> str:
    if custom_template:
        return render_prompt_template(custom_template, question=question, context=context)
    version = prompt_version if prompt_version in {"grounded", "concise", "analyst"} else "grounded"
    instructions = {
        "grounded": (
            "Answer the user question using only the cited context. "
            "If the context is insufficient, say the knowledge base does not contain enough evidence. "
            "Keep citations in square brackets."
        ),
        "concise": (
            "Answer briefly using only the cited context. "
            "Prefer one short paragraph and keep the most relevant citation in square brackets. "
            "If evidence is insufficient, say so."
        ),
        "analyst": (
            "Answer using only the cited context. "
            "First give the direct answer, then add a short evidence note explaining which citations support it. "
            "Keep citations in square brackets and do not invent facts."
        ),
    }
    return f"{instructions[version]}\n\nQuestion: {question}\n\nContext:\n{context}"


def build_query_rewrite_prompt(question: str, custom_template: str | None = None) -> str:
    if custom_template:
        return render_prompt_template(custom_template, question=question, context="")
    return (
        "Rewrite the user's colloquial question into a concise search query for enterprise knowledge base retrieval. "
        "Keep names, entities, dates, and domain keywords. Return only the rewritten query, no explanation.\n\n"
        f"Question: {question}"
    )


def render_prompt_template(template: str, question: str, context: str) -> str:
    try:
        rendered = template.format(question=question, context=context)
    except (KeyError, IndexError, ValueError):
        rendered = f"{template}\n\nQuestion: {question}\n\nContext:\n{context}".strip()
    return rendered.strip()


def extract_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def post_openai_json(base_url: str, path: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        openai_api_url(base_url, path),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python 1.0 RAG-Eval-Assistant/0.1",
            "X-Stainless-Lang": "python",
            "X-Stainless-Package-Version": "1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "1010" in message:
            message = (
                f"{message} "
                "Cloudex returned 403/1010, which usually means the gateway protection layer rejected this HTTP client. "
                "Check Base URL/model/key first, then try again after the request headers update."
            )
        raise RuntimeError(f"OpenAI request failed with HTTP {exc.code}: {message}") from exc


def post_dashscope_json(base_url: str, path: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    clean_base = base_url.rstrip("/")
    clean_path = path if path.startswith("/") else f"/{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{clean_base}{clean_path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "RAG-Eval-Assistant/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope rerank request failed with HTTP {exc.code}: {message}") from exc


def build_answer_provider(config: RuntimeConfig) -> AnswerProvider:
    if config.answer_provider == "openai" and config.openai_api_key:
        return OpenAIAnswerProvider(
            config.openai_api_key,
            config.answer_model,
            base_url=config.openai_base_url,
            api_mode=config.answer_api_mode,
            answer_prompt_template=config.answer_prompt_template,
        )
    return LocalAnswerProvider()


def build_embedding_provider(config: RuntimeConfig) -> EmbeddingProvider:
    embedding_api_key = config.embedding_api_key or config.openai_api_key
    if config.embedding_provider == "openai" and embedding_api_key:
        return OpenAIEmbeddingProvider(
            embedding_api_key,
            config.embedding_model,
            base_url=config.embedding_base_url or config.openai_base_url,
        )
    return LocalEmbeddingProvider()


def build_query_rewrite_provider(config: RuntimeConfig) -> QueryRewriteProvider | None:
    if config.answer_provider == "openai" and config.openai_api_key:
        return OpenAIQueryRewriteProvider(
            config.openai_api_key,
            config.answer_model,
            base_url=config.openai_base_url,
            api_mode=config.answer_api_mode,
            rewrite_prompt_template=config.rewrite_prompt_template,
        )
    return None


def build_rerank_provider(config: RuntimeConfig) -> RerankProvider | None:
    if config.rerank_provider == "dashscope" and config.rerank_api_key:
        return DashScopeRerankProvider(
            config.rerank_api_key,
            config.rerank_model,
            base_url=config.rerank_base_url,
        )
    return None

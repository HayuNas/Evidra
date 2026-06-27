# RAG Eval Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable interview-demo MVP for an enterprise knowledge-base RAG evaluation assistant.

**Architecture:** Create a new `rag-eval-assistant` app rather than modifying `upstream/haystack` or `upstream/langfuse`. The backend keeps the core RAG behavior dependency-light and testable, with optional FastAPI and Langfuse integration around it. The frontend is a static demo console that talks to the backend endpoints.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, unittest, optional Langfuse SDK, static HTML/CSS/JS frontend.

---

## Tasks

- [x] Backend scaffold and domain models.
- [x] Text parsing and heading-aware chunking.
- [x] JSON-backed local document store and lexical retrieval.
- [x] Document ingestion service.
- [x] RAG answer pipeline and safe Langfuse tracing wrapper.
- [x] Evaluation runner with summary metrics.
- [x] FastAPI routes for health, documents, ask, and evaluations.
- [x] Static frontend demo console.
- [x] Final verification.

## Verification Notes

- 2026-06-26: `python -m unittest discover -s tests` with Codex bundled Python ran 19 tests: 17 passed, 2 skipped because FastAPI is not installed in that environment.
- 2026-06-26: `node --check frontend/app.js` passed.
- 2026-06-26: Added FastAPI CORS middleware and a skipped-until-FastAPI API preflight test so the static frontend can call the local backend from a browser.
- 2026-06-26: Added a frontend security regression test and replaced dynamic `innerHTML` rendering with DOM `textContent` rendering.
- 2026-06-26: Added `httpx2` to backend dependencies after `uv run python -m unittest discover -s tests` reported Starlette `TestClient` requires it.
- 2026-06-26: Using `C:\Users\11571\AppData\Local\Programs\Python\Python312\Scripts\uv.exe`, `uv sync` installed `httpx2`, `httpcore2`, and `truststore`.
- 2026-06-26: `uv run python -m unittest discover -s tests` ran 19 tests, all passed.
- 2026-06-26: Started FastAPI with `uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000` and smoke-tested `/health`, `/documents`, `/ask`, and `/evaluations/run`; evaluation passed 2/2 with pass rate 1.0 after uploading `docs/sample-handbook.md`.
- Remaining limitation: Browser screenshot-level frontend verification was not completed because the in-app browser policy blocked local file/localhost navigation in this session.

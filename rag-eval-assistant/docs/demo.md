# Evidra Demo

## Backend

From `rag-eval-assistant/backend`, install and run with `uv`:

```powershell
uv sync
uv run python -m unittest discover -s tests
uv run uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8010
```

For PDF and Langfuse SDK support:

```powershell
uv sync --extra pdf --extra langfuse
```

## Frontend

Open the backend-served frontend:

```text
http://127.0.0.1:8010/
```

## Suggested Demo Document

Upload `rag-eval-assistant/docs/sample-handbook.md`:

```markdown
# Finance
Receipts are required for expenses.

# Security
Access keys should rotate quarterly.
```

Upload it, ask "What does an expense report need?", then run the evaluation set.

## Demo Flow

1. Open `工作台`, upload `docs/sample-handbook.md`.
2. Ask `What does an expense report need?`.
3. Open `参数设置`, switch retrieval mode and Top K.
4. Enable `查询改写 Query rewrite`, ask a colloquial version of the same question.
5. Enable `多路召回 Multi-route` and optional `Rerank`.
6. Open `评测`, run `demo.json`, then compare strategies.
7. Open `日志` to show request and error history.

## Interview Talking Points

- The answer is returned with explicit citations instead of unsupported prose.
- Retrieval, answer generation, and evaluation all use the same local pipeline.
- Query rewrite and multi-route retrieval make colloquial questions more robust.
- Rerank can improve evidence ordering before generation.
- Langfuse integration is isolated behind a wrapper, so observability failures degrade to local JSONL traces.
- Evaluation metrics make prompt or retrieval changes measurable.

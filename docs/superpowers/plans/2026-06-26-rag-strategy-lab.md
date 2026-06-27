# RAG Strategy Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current RAG Eval Assistant MVP into a strategy lab where users can switch chunking and retrieval settings from the frontend and compare evaluation results across strategies.

**Architecture:** Keep the current dependency-light FastAPI + static frontend architecture. Add typed dataclass strategy configuration objects in the backend, route those configs through ingestion, question answering, and evaluation, then expose compact controls in the frontend. Implement only local lexical and BM25-like retrieval in this phase; leave embeddings, rerankers, and LLM synthesis as future additions.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic request models, dataclasses, unittest, static HTML/CSS/JS.

---

## File Structure

- Modify `rag-eval-assistant/backend/app/schemas/models.py`
  - Add `ChunkingConfig`, `RetrievalConfig`, `RagStrategyConfig`, `StrategyEvaluationSummary`.
  - Add `strategy` metadata to answers and evaluation summaries through existing `metadata` fields where possible.
- Modify `rag-eval-assistant/backend/app/rag/text.py`
  - Keep `extract_text` and `tokenize`.
  - Replace the single fixed `chunk_text` behavior with configurable `chunk_text(text, source, config=None)`.
  - Support `heading`, `paragraph`, and `fixed` chunk strategies.
  - Support `chunk_size` and `overlap`.
- Modify `rag-eval-assistant/backend/app/rag/store.py`
  - Keep JSON persistence.
  - Add `RetrievalConfig`-aware `search(query, config=None)`.
  - Support `mode="tfidf"` and `mode="bm25"`.
  - Support `top_k` and `score_threshold`.
- Modify `rag-eval-assistant/backend/app/services/ingestion.py`
  - Accept optional `ChunkingConfig` during ingest.
- Modify `rag-eval-assistant/backend/app/rag/pipeline.py`
  - Accept optional `RetrievalConfig` during `answer`.
  - Include strategy metadata in `AnswerResult.metadata`.
- Modify `rag-eval-assistant/backend/app/evals/runner.py`
  - Accept optional `RetrievalConfig`.
  - Add strategy comparison runner for named strategy configs.
- Modify `rag-eval-assistant/backend/app/api/main.py`
  - Add request models for chunking, retrieval, ask, evaluation, and comparison.
  - Update `/documents`, `/ask`, `/evaluations/run`.
  - Add `/evaluations/compare`.
- Modify `rag-eval-assistant/frontend/index.html`
  - Add a compact `Retrieval Settings` panel in the left sidebar.
  - Add a `Compare strategies` button and strategy comparison table.
- Modify `rag-eval-assistant/frontend/app.js`
  - Read settings from controls.
  - Send settings with upload, ask, eval, and compare requests.
  - Render strategy metadata and comparison results.
- Modify `rag-eval-assistant/frontend/styles.css`
  - Style settings controls and comparison table without changing the current visual language.
- Add/modify tests in `rag-eval-assistant/backend/tests/`
  - `test_models_and_settings.py`
  - `test_text_chunking.py`
  - `test_store_retrieval.py`
  - `test_ingestion_service.py`
  - `test_pipeline_and_trace.py`
  - `test_evaluation_runner.py`
  - `test_api.py`
  - `test_frontend_security.py`

---

## Task 1: Strategy Config Domain Models

**Files:**
- Modify: `rag-eval-assistant/backend/app/schemas/models.py`
- Test: `rag-eval-assistant/backend/tests/test_models_and_settings.py`

- [ ] **Step 1: Write failing tests for config defaults and clamping**

Add tests that instantiate configs and verify normalized values:

```python
from app.schemas.models import ChunkingConfig, RetrievalConfig, RagStrategyConfig


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
cd "C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\backend"
uv run python -m unittest tests.test_models_and_settings
```

Expected: import failure for the new config classes.

- [ ] **Step 3: Implement config dataclasses**

Add these dataclasses to `models.py`:

```python
@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "heading"
    chunk_size: int = 900
    overlap: int = 0

    def __post_init__(self) -> None:
        strategy = self.strategy if self.strategy in {"heading", "paragraph", "fixed"} else "heading"
        chunk_size = min(2000, max(300, int(self.chunk_size)))
        overlap = min(chunk_size - 1, max(0, int(self.overlap)))
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "overlap", overlap)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "tfidf"
    top_k: int = 4
    score_threshold: float = 0.0

    def __post_init__(self) -> None:
        mode = self.mode if self.mode in {"tfidf", "bm25"} else "tfidf"
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "top_k", min(10, max(1, int(self.top_k))))
        object.__setattr__(self, "score_threshold", max(0.0, float(self.score_threshold)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagStrategyConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunking": self.chunking.to_dict(),
            "retrieval": self.retrieval.to_dict(),
        }
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
uv run python -m unittest tests.test_models_and_settings
```

Expected: all tests in that file pass.

---

## Task 2: Configurable Chunking

**Files:**
- Modify: `rag-eval-assistant/backend/app/rag/text.py`
- Modify: `rag-eval-assistant/backend/app/services/ingestion.py`
- Test: `rag-eval-assistant/backend/tests/test_text_chunking.py`
- Test: `rag-eval-assistant/backend/tests/test_ingestion_service.py`

- [ ] **Step 1: Write failing chunking strategy tests**

Add tests:

```python
from app.schemas.models import ChunkingConfig


def test_paragraph_chunking_splits_on_blank_lines(self):
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text(text, "doc.md", ChunkingConfig(strategy="paragraph", chunk_size=900))

    self.assertEqual([chunk.text for chunk in chunks], ["First paragraph.", "Second paragraph.", "Third paragraph."])


def test_fixed_chunking_uses_size_and_overlap(self):
    text = "abcdefghij"

    chunks = chunk_text(text, "doc.txt", ChunkingConfig(strategy="fixed", chunk_size=4, overlap=1))

    self.assertEqual([chunk.text for chunk in chunks], ["abcd", "defg", "ghij"])
```

Because `ChunkingConfig` clamps `chunk_size` to 300 by default, this test should construct a test-only config with small limits by adding a private helper or by testing fixed chunking with a 300+ character string. Prefer a 310 character string to avoid bypassing production validation.

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run python -m unittest tests.test_text_chunking
```

Expected: failures because `chunk_text` does not accept a config object or fixed strategy.

- [ ] **Step 3: Implement chunk strategy dispatcher**

Keep backward compatibility:

```python
def chunk_text(
    text: str,
    source: str,
    config: ChunkingConfig | None = None,
    max_chars: int | None = None,
) -> list[DocumentChunk]:
    if config is None:
        config = ChunkingConfig(chunk_size=max_chars or 900)
    if config.strategy == "paragraph":
        return _chunk_paragraphs(text, source, config)
    if config.strategy == "fixed":
        return _chunk_fixed(text, source, config)
    return _chunk_heading(text, source, config)
```

Implement helpers:

```python
def _make_chunk(source: str, text: str, heading: str | None, position: int) -> DocumentChunk:
    return DocumentChunk(
        id=f"{source}:{position:04d}",
        source=source,
        text=text.strip(),
        heading=heading,
        position=position,
    )
```

Heading strategy should preserve current behavior and use `config.chunk_size`.

Paragraph strategy should group paragraphs until adding another paragraph would exceed `chunk_size`.

Fixed strategy should create sliding windows:

```python
step = max(1, config.chunk_size - config.overlap)
for start in range(0, len(text), step):
    piece = text[start : start + config.chunk_size].strip()
```

- [ ] **Step 4: Update ingestion to pass config**

Change `DocumentIngestionService.ingest` signature to:

```python
def ingest(self, filename: str, content: bytes, chunking: ChunkingConfig | None = None) -> IngestedDocument:
```

Call:

```python
chunks = chunk_text(text, filename, chunking)
```

- [ ] **Step 5: Run chunking and ingestion tests**

Run:

```powershell
uv run python -m unittest tests.test_text_chunking tests.test_ingestion_service
```

Expected: tests pass.

---

## Task 3: Retrieval Modes and Recall Controls

**Files:**
- Modify: `rag-eval-assistant/backend/app/rag/store.py`
- Test: `rag-eval-assistant/backend/tests/test_store_retrieval.py`

- [ ] **Step 1: Write failing tests for retrieval config**

Add tests:

```python
from app.schemas.models import RetrievalConfig


def test_search_respects_top_k_and_score_threshold(self):
    with tempfile.TemporaryDirectory() as directory:
        store = LocalDocumentStore(Path(directory) / "documents.json")
        store.replace_source(
            "handbook.md",
            [
                DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses."),
                DocumentChunk("handbook.md:0002", "handbook.md", "Receipts may include hotel invoices."),
            ],
        )

        results = store.search("receipts", RetrievalConfig(top_k=1, score_threshold=0.1))

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].score, 0.1)


def test_bm25_mode_retrieves_matching_chunk(self):
    with tempfile.TemporaryDirectory() as directory:
        store = LocalDocumentStore(Path(directory) / "documents.json")
        store.replace_source(
            "handbook.md",
            [
                DocumentChunk("handbook.md:0001", "handbook.md", "Receipts are required for expenses."),
                DocumentChunk("handbook.md:0002", "handbook.md", "Rotate access keys quarterly."),
            ],
        )

        results = store.search("quarterly key rotation", RetrievalConfig(mode="bm25", top_k=1))

        self.assertEqual(results[0].id, "handbook.md:0002")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run python -m unittest tests.test_store_retrieval
```

Expected: type or behavior failure because `search` currently accepts only `top_k`.

- [ ] **Step 3: Update `search` signature while preserving old calls**

Implement:

```python
def search(
    self,
    query: str,
    config: RetrievalConfig | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    config = config or RetrievalConfig(top_k=top_k or 4)
```

If callers pass `top_k=1`, keep that behavior.

- [ ] **Step 4: Implement TF-IDF and BM25-like scoring helpers**

Use existing TF-IDF as `_score_tfidf`.

Add BM25-like scoring:

```python
def _score_bm25(
    self,
    query_counts: Counter[str],
    chunk_terms: Counter[str],
    document_frequencies: Counter[str],
    average_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    length = sum(chunk_terms.values()) or 1
    score = 0.0
    for term, query_count in query_counts.items():
        term_frequency = chunk_terms.get(term, 0)
        if term_frequency == 0:
            continue
        idf = math.log((len(self._chunks) - document_frequencies[term] + 0.5) / (document_frequencies[term] + 0.5) + 1)
        denominator = term_frequency + k1 * (1 - b + b * length / max(average_length, 1))
        score += query_count * idf * ((term_frequency * (k1 + 1)) / denominator)
    return score
```

- [ ] **Step 5: Apply top_k and threshold**

After scoring:

```python
filtered = [chunk for chunk in scored if chunk.score >= config.score_threshold]
return sorted(filtered, key=lambda chunk: chunk.score, reverse=True)[: config.top_k]
```

- [ ] **Step 6: Run retrieval tests**

Run:

```powershell
uv run python -m unittest tests.test_store_retrieval
```

Expected: tests pass.

---

## Task 4: Pipeline and Evaluation Config Propagation

**Files:**
- Modify: `rag-eval-assistant/backend/app/rag/pipeline.py`
- Modify: `rag-eval-assistant/backend/app/evals/runner.py`
- Test: `rag-eval-assistant/backend/tests/test_pipeline_and_trace.py`
- Test: `rag-eval-assistant/backend/tests/test_evaluation_runner.py`

- [ ] **Step 1: Write failing pipeline metadata test**

Add test:

```python
from app.schemas.models import RetrievalConfig


def test_answer_uses_retrieval_config_and_returns_metadata(self):
    # Existing store setup with two matching chunks.
    result = pipeline.answer("receipts", RetrievalConfig(top_k=1, mode="bm25"))

    self.assertEqual(len(result.citations), 1)
    self.assertEqual(result.metadata["retrieval"]["mode"], "bm25")
    self.assertEqual(result.metadata["retrieval"]["top_k"], 1)
```

- [ ] **Step 2: Run pipeline tests and verify failure**

Run:

```powershell
uv run python -m unittest tests.test_pipeline_and_trace
```

Expected: `answer` does not accept retrieval config yet.

- [ ] **Step 3: Update pipeline**

Change:

```python
def answer(self, question: str, retrieval: RetrievalConfig | None = None) -> AnswerResult:
    retrieval = retrieval or RetrievalConfig(top_k=self.top_k)
    retrieved = self.store.search(question, retrieval)
```

Return metadata:

```python
metadata={"retrieval": retrieval.to_dict()}
```

- [ ] **Step 4: Write failing evaluation config test**

Add to `test_evaluation_runner.py`:

```python
summary = runner.run("demo.json", RetrievalConfig(mode="bm25", top_k=1))

self.assertEqual(summary.results[0].answer.metadata["retrieval"]["mode"], "bm25")
```

- [ ] **Step 5: Update evaluation runner**

Change:

```python
def run(self, eval_set_name: str, retrieval: RetrievalConfig | None = None) -> EvaluationSummary:
    answer = self.pipeline.answer(item.question, retrieval)
```

- [ ] **Step 6: Add strategy comparison runner**

Add method:

```python
def compare(self, eval_set_name: str, strategies: dict[str, RetrievalConfig]) -> list[dict[str, Any]]:
    comparisons = []
    for name, retrieval in strategies.items():
        summary = self.run(eval_set_name, retrieval)
        comparisons.append({
            "name": name,
            "retrieval": retrieval.to_dict(),
            "summary": summary.to_dict(),
        })
    return comparisons
```

- [ ] **Step 7: Run pipeline and evaluation tests**

Run:

```powershell
uv run python -m unittest tests.test_pipeline_and_trace tests.test_evaluation_runner
```

Expected: tests pass.

---

## Task 5: FastAPI Request Models and Routes

**Files:**
- Modify: `rag-eval-assistant/backend/app/api/main.py`
- Test: `rag-eval-assistant/backend/tests/test_api.py`

- [ ] **Step 1: Write failing API tests for strategy settings**

Add tests that call `/documents`, `/ask`, `/evaluations/run`, and `/evaluations/compare`:

```python
def test_ask_accepts_retrieval_config(self):
    answer = client.post(
        "/ask",
        json={
            "question": "What do expenses need?",
            "retrieval": {"mode": "bm25", "top_k": 1, "score_threshold": 0},
        },
    )

    self.assertEqual(answer.status_code, 200)
    self.assertEqual(answer.json()["metadata"]["retrieval"]["mode"], "bm25")
    self.assertEqual(len(answer.json()["citations"]), 1)


def test_compare_evaluations_returns_multiple_strategies(self):
    response = client.post(
        "/evaluations/compare",
        json={
            "eval_set": "demo.json",
            "strategies": [
                {"name": "tfidf-k2", "retrieval": {"mode": "tfidf", "top_k": 2}},
                {"name": "bm25-k1", "retrieval": {"mode": "bm25", "top_k": 1}},
            ],
        },
    )

    self.assertEqual(response.status_code, 200)
    self.assertEqual([item["name"] for item in response.json()["comparisons"]], ["tfidf-k2", "bm25-k1"])
```

- [ ] **Step 2: Run API tests and verify failure**

Run:

```powershell
uv run python -m unittest tests.test_api
```

Expected: route or schema failures.

- [ ] **Step 3: Add Pydantic request models**

Add:

```python
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
```

Update `AskRequest`:

```python
class AskRequest(BaseModel):
    question: str
    retrieval: RetrievalRequest = RetrievalRequest()
```

Update `EvalRequest` similarly.

- [ ] **Step 4: Let upload accept chunking config through form fields**

Change upload route signature:

```python
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("heading"),
    chunk_size: int = Form(900),
    chunk_overlap: int = Form(0),
):
```

Import `Form`.

Call:

```python
chunking = ChunkingConfig(strategy=chunk_strategy, chunk_size=chunk_size, overlap=chunk_overlap)
result = ingestion.ingest(file.filename or "upload.txt", content, chunking)
```

- [ ] **Step 5: Add compare endpoint**

Add:

```python
@app.post("/evaluations/compare")
def compare_evaluations(request: CompareEvalRequest):
    strategies = {item.name: item.retrieval.to_config() for item in request.strategies}
    return {
        "eval_set": request.eval_set,
        "comparisons": EvaluationRunner(settings, pipeline).compare(request.eval_set, strategies),
    }
```

- [ ] **Step 6: Run API tests**

Run:

```powershell
uv run python -m unittest tests.test_api
```

Expected: tests pass.

---

## Task 6: Frontend Strategy Controls

**Files:**
- Modify: `rag-eval-assistant/frontend/index.html`
- Modify: `rag-eval-assistant/frontend/app.js`
- Modify: `rag-eval-assistant/frontend/styles.css`
- Test: `rag-eval-assistant/backend/tests/test_frontend_security.py`

- [ ] **Step 1: Add static test for required frontend controls**

Extend `test_frontend_security.py`:

```python
def test_frontend_contains_strategy_controls(self):
    index_html = frontend_index_html().read_text(encoding="utf-8")

    self.assertIn('id="chunkStrategy"', index_html)
    self.assertIn('id="retrievalMode"', index_html)
    self.assertIn('id="topK"', index_html)
    self.assertIn('id="compareStrategiesButton"', index_html)
```

Add helper `frontend_index_html()` that mirrors the existing `frontend_app_js()` lookup.

- [ ] **Step 2: Run frontend static tests and verify failure**

Run:

```powershell
uv run python -m unittest tests.test_frontend_security
```

Expected: missing control IDs.

- [ ] **Step 3: Add `Retrieval Settings` panel**

In `index.html`, add a sidebar section between upload and ask:

```html
<section class="panel settings-panel">
  <h2>2. Retrieval settings</h2>
  <div class="settings-grid">
    <label><span>Chunk strategy</span><select id="chunkStrategy">...</select></label>
    <label><span>Chunk size</span><select id="chunkSize">...</select></label>
    <label><span>Overlap</span><select id="chunkOverlap">...</select></label>
    <label><span>Retrieval mode</span><select id="retrievalMode">...</select></label>
    <label><span>Top K</span><select id="topK">...</select></label>
    <label><span>Score threshold</span><input id="scoreThreshold" type="number" min="0" step="0.1" value="0" /></label>
  </div>
</section>
```

Renumber Ask to `3. Ask a question`.

- [ ] **Step 4: Add JS helpers for config payloads**

In `app.js`, add:

```javascript
function chunkingConfig() {
  return {
    strategy: document.querySelector("#chunkStrategy").value,
    chunk_size: Number(document.querySelector("#chunkSize").value),
    overlap: Number(document.querySelector("#chunkOverlap").value),
  };
}

function retrievalConfig() {
  return {
    mode: document.querySelector("#retrievalMode").value,
    top_k: Number(document.querySelector("#topK").value),
    score_threshold: Number(document.querySelector("#scoreThreshold").value),
  };
}
```

Update upload `FormData`:

```javascript
const chunking = chunkingConfig();
data.append("chunk_strategy", chunking.strategy);
data.append("chunk_size", String(chunking.chunk_size));
data.append("chunk_overlap", String(chunking.overlap));
```

Update ask and eval JSON bodies to include `retrieval: retrievalConfig()`.

- [ ] **Step 5: Render strategy metadata**

Change the Retrieval meta strip strong text from static `lexical local` to:

```html
<strong id="retrievalLabel">tfidf k=4</strong>
```

In `renderAnswer`:

```javascript
const retrieval = result.metadata?.retrieval;
retrievalLabel.textContent = retrieval ? `${retrieval.mode} k=${retrieval.top_k}` : "local";
```

- [ ] **Step 6: Add CSS**

Add compact styling:

```css
.settings-grid {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.settings-grid label {
  display: grid;
  gap: 6px;
  color: var(--muted);
  font-size: 12px;
}

.settings-grid select,
.settings-grid input {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  color: var(--ink);
  padding: 0 10px;
}
```

- [ ] **Step 7: Run frontend static and JS checks**

Run:

```powershell
uv run python -m unittest tests.test_frontend_security
C:\Users\11571\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --check app.js
```

Expected: tests and JS syntax pass.

---

## Task 7: Strategy Comparison UI

**Files:**
- Modify: `rag-eval-assistant/frontend/index.html`
- Modify: `rag-eval-assistant/frontend/app.js`
- Modify: `rag-eval-assistant/frontend/styles.css`
- Test: `rag-eval-assistant/backend/tests/test_frontend_security.py`

- [ ] **Step 1: Add static test for comparison table**

Add:

```python
def test_frontend_contains_comparison_table(self):
    index_html = frontend_index_html().read_text(encoding="utf-8")

    self.assertIn('id="compareStrategiesButton"', index_html)
    self.assertIn('id="comparisonRows"', index_html)
```

- [ ] **Step 2: Add button and table markup**

Add button near `Run evaluation`:

```html
<button type="button" id="compareStrategiesButton" class="secondary">Compare strategies</button>
```

Add comparison section below evaluation table:

```html
<section class="evaluation comparison">
  <div class="panel-heading">
    <h2>Strategy comparison</h2>
    <span id="comparisonSummary">No comparison yet</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Mode</th>
          <th>Top K</th>
          <th>Pass rate</th>
          <th>Coverage</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="comparisonRows">
        <tr><td colspan="6">Compare strategies to populate this table.</td></tr>
      </tbody>
    </table>
  </div>
</section>
```

- [ ] **Step 3: Add comparison request**

In `app.js`:

```javascript
compareStrategiesButton.addEventListener("click", async () => {
  const result = await request("/evaluations/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      eval_set: "demo.json",
      strategies: [
        { name: "tfidf-k2", retrieval: { mode: "tfidf", top_k: 2, score_threshold: 0 } },
        { name: "tfidf-k4", retrieval: { mode: "tfidf", top_k: 4, score_threshold: 0 } },
        { name: "bm25-k4", retrieval: { mode: "bm25", top_k: 4, score_threshold: 0 } },
      ],
    }),
  });
  renderComparison(result);
});
```

- [ ] **Step 4: Render comparison results safely**

Use DOM creation and `textContent`, not `innerHTML`:

```javascript
function renderComparison(result) {
  comparisonSummary.textContent = `${result.comparisons.length} strategies compared`;
  clear(comparisonRows);
  result.comparisons.forEach((comparison) => {
    const row = document.createElement("tr");
    const summary = comparison.summary;
    const retrieval = comparison.retrieval;
    [comparison.name, retrieval.mode, retrieval.top_k, `${Math.round(summary.pass_rate * 100)}%`, `${Math.round(summary.citation_coverage * 100)}%`, `${summary.average_latency_ms} ms`]
      .forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.appendChild(cell);
      });
    comparisonRows.appendChild(row);
  });
}
```

- [ ] **Step 5: Run frontend tests and syntax check**

Run:

```powershell
uv run python -m unittest tests.test_frontend_security
C:\Users\11571\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --check app.js
```

Expected: tests and JS syntax pass.

---

## Task 8: End-to-End Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-06-26-rag-strategy-lab.md`
- No production code changes in this task.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
cd "C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\backend"
uv run python -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 2: Run frontend syntax check**

Run:

```powershell
cd "C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\frontend"
C:\Users\11571\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe --check app.js
```

Expected: command exits 0 with no syntax error.

- [ ] **Step 3: Run API smoke test**

Start backend:

```powershell
cd "C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\backend"
uv run uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8010
```

Open frontend:

```text
C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\frontend\index.html
```

Set backend URL:

```text
http://127.0.0.1:8010
```

Upload:

```text
C:\Users\11571\Desktop\RAG+langfuse\.worktrees\codex-rag-eval-assistant\rag-eval-assistant\docs\sample-handbook.md
```

Expected:

- Upload shows `sample-handbook.md` with `2 chunks` for heading strategy.
- Ask returns at least one citation from `sample-handbook.md`.
- `Run evaluation` returns `2 / 2 tests passed`.
- `Compare strategies` shows three strategy rows.

- [ ] **Step 4: Update verification notes**

Append actual test results to this plan under a `Verification Notes` section.

---

## Self-Review

- Spec coverage: The plan covers configurable chunking, retrieval modes, recall controls, frontend settings, current-strategy evaluation, and strategy comparison.
- Scope control: The plan intentionally excludes embeddings, external rerankers, LLM generation, authentication, database migration, and full document management. Those remain future phases.
- Type consistency: Config objects use dataclass names `ChunkingConfig`, `RetrievalConfig`, and `RagStrategyConfig`; Pydantic request models convert into those dataclasses.
- Testing: Every backend behavior change has a corresponding unittest task. Frontend behavior has static markup/security checks plus JS syntax verification and manual browser smoke testing.

## Verification Notes

- 2026-06-26: `uv run python -m unittest discover -s tests` ran 34 tests, all passed.
- 2026-06-26: `node --check frontend/app.js` passed.
- 2026-06-26: Started a temporary FastAPI server on `127.0.0.1:8020` and smoke-tested `/health`, `/documents`, `/ask`, `/evaluations/run`, and `/evaluations/compare`.
- 2026-06-26: Smoke test uploaded `docs/sample-handbook.md`, asked with `retrieval.mode=bm25` and `top_k=1`, ran evaluation with 2/2 passed, and compared `tfidf-k2`, `tfidf-k4`, and `bm25-k4`.

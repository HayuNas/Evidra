# Evidra

Evidra 是一个面向面试展示的企业知识库 RAG 评测与调优工作台。它不只是“上传文档后聊天”，而是把检索、重排、Prompt、模型配置、评测和观测放到同一个可复现实验环境里，展示一个 RAG 应用从文档接入到质量回归的完整工程闭环。

实际项目目录在：

```text
rag-eval-assistant/
```

## 功能亮点

- 文档上传与本地索引：支持 Markdown、TXT，安装 `pdf` extra 后支持 PDF。
- 多种切分策略：标题感知、段落、固定长度。
- 多种检索策略：TF-IDF、BM25-like、Embedding、Hybrid。
- 查询改写 Query rewrite：用大模型把口语问题改写成更适合检索的 query。
- 多路召回 Multi-route：原始问题与改写 query 分别走多种检索路线，合并去重。
- Rerank：支持百炼 DashScope rerank 模型对候选 chunk 重排。
- Prompt 可配置：回答 Prompt 和查询改写 Prompt 都能在前端调整。
- 评测集与策略对比：用固定问题集比较不同检索和 Prompt 策略。
- 可观测性：Langfuse 可选；未配置时自动落到本地 JSONL trace。
- 安全展示：API Key 保存到本地运行配置，前端只显示“已配置”占位符，不回显密钥。

## 项目结构

```text
rag-eval-assistant/
  backend/              FastAPI 后端、RAG pipeline、评测 runner
  frontend/             静态 HTML/CSS/JS 前端工作台
  data/eval_sets/       示例评测集
  docs/                 演示文档与面试说明
```

## 快速启动

建议使用 `uv`。

```powershell
cd "rag-eval-assistant\backend"
uv sync
uv run python -m unittest discover -s tests
uv run uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8010
```

浏览器打开：

```text
http://127.0.0.1:8010/
```

如果需要 PDF 解析或 Langfuse SDK：

```powershell
uv sync --extra pdf --extra langfuse
```

## 模型配置

前端进入 `参数设置`：

- Answer model：OpenAI-compatible 回答模型。
- Embedding model：OpenAI-compatible embedding 模型，例如百炼 `text-embedding-v4`。
- Rerank provider：选择 `百炼 DashScope` 后填写 rerank key 和模型名。
- Langfuse：填写 public key、secret key、host 后启用远端观测。

## 演示流程

1. 启动后端，打开前端。
2. 上传 `rag-eval-assistant/docs/sample-handbook.md` 或自己的简历 PDF。
3. 在工作台提问，例如 `What does an expense report need?`。
4. 到参数设置切换检索模式、Top K、查询改写、多路召回、Rerank。
5. 回到工作台再次提问，对比引用证据和检索链路 metadata。
6. 到评测页运行 `demo.json`，再运行策略对比。
7. 打开日志页，说明请求、错误和本地/远端 trace 的作用。

## 面试讲解重点

- 不是普通聊天框，而是可调参、可评测、可观测的 RAG 工作台。
- 每次回答都有 citations，可以解释答案来自哪些文档片段。
- 检索策略、切分策略、Prompt、query rewrite、multi-route、rerank 都能从 UI 切换。
- 评测集让优化不靠感觉，而是用固定问题集比较通过率、引用覆盖率和延迟。
- Langfuse 失败时会降级到本地 trace，保证演示稳定。

## 测试

```powershell
cd "rag-eval-assistant\backend"
uv run python -m unittest discover -s tests
```

前端语法检查：

```powershell
node --check "rag-eval-assistant\frontend\app.js"
```
## 界面
<img width="1415" height="1011" alt="image" src="https://github.com/user-attachments/assets/95cfed4e-c491-490d-a1a7-e8312d4e1b89" />



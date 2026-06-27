# RAG + Langfuse 面试作品设计

## 背景

目标是做一个适合 AI 应用开发岗位展示的作品：企业知识库 RAG 评测助手。它不追求训练模型或复刻大型平台，而是展示真实 AI 应用落地中更关键的能力：文档接入、检索增强生成、引用溯源、可观测性、批量评测和质量回归。

本项目使用 `upstream/haystack` 和 `upstream/langfuse` 作为参考源码。实际作品新建在 `rag-eval-assistant`，避免直接改大型开源项目核心代码。

## 产品范围

MVP 支持用户上传文档，系统解析并切分为 chunks，建立本地索引。用户提问后，系统用 Haystack 检索相关片段，构造 prompt，调用 LLM 生成回答，并返回答案、引用片段、来源文件和置信信息。

每次问答都会记录到 Langfuse Cloud，包括用户问题、检索结果、prompt、模型回答、耗时、模型参数和自定义 metadata。系统还提供一组测试问题的批量评测入口，用来展示 prompt 或检索策略变更前后的质量变化。

第一版不做复杂权限系统、多租户、在线协同编辑或完整 Langfuse 自托管。Langfuse 采用 Cloud 版，降低部署复杂度。

## 推荐架构

```text
rag-eval-assistant/
  backend/
    app/
      api/
      rag/
      services/
      evals/
      schemas/
    tests/
    pyproject.toml
  frontend/
    src/
  data/
    uploads/
    indexes/
    eval_sets/
  docs/
```

后端使用 FastAPI，负责文件上传、索引、问答和评测任务。RAG 核心用 Haystack pipeline 组织，第一版优先使用轻量本地存储，后续可替换为 Qdrant 或 Chroma。

前端使用一个简单的 Web UI，包含文档管理、问答页、引用查看和评测结果页。页面只服务作品展示，不做复杂后台管理。

前端进入实现前必须使用 `frontend-design` skill。设计方向要围绕“可追溯的企业知识库问答”和“RAG 质量评测”建立视觉语言，而不是套用通用 SaaS 模板。前端计划阶段需要先给出颜色、字体、布局和一个可解释的 signature element，再进入代码实现。

Langfuse Cloud 通过 Python SDK 或 Haystack 集成接入。每次问答创建 trace，检索、prompt 构建和 LLM 调用作为 observation/span 记录。

## 数据流

文档索引流程：

1. 用户上传 PDF、Markdown 或 TXT。
2. 后端保存原文件到 `data/uploads`。
3. 解析器抽取文本和基础 metadata。
4. 分块器按长度和标题边界生成 chunks。
5. embedding 组件为 chunks 生成向量。
6. DocumentStore 写入 chunk、embedding、来源文件、页码或段落位置。

问答流程：

1. 用户输入问题。
2. 系统创建 Langfuse trace。
3. Haystack pipeline 执行检索。
4. 系统将 top-k chunks 写入 Langfuse observation。
5. PromptBuilder 构造带引用约束的 prompt。
6. LLM 生成答案。
7. AnswerBuilder 输出答案和引用。
8. 系统写入 Langfuse 分数或 metadata，并返回给前端。

评测流程：

1. 从 `data/eval_sets` 读取测试集。
2. 批量调用同一套 RAG pipeline。
3. 记录每条样本的答案、引用、耗时和错误。
4. 将运行结果发送到 Langfuse dataset run 或 trace metadata。
5. 在本地 UI 展示通过率、平均耗时、失败样本和引用覆盖情况。

## 核心模块

`DocumentIngestionService` 负责上传文件、解析文本、切分和写入索引。

`RagPipelineFactory` 负责构造 Haystack pipeline，包括 retriever、prompt builder、generator 和 answer builder。

`LangfuseTraceService` 负责封装 Langfuse Cloud 接入，避免业务代码到处散落 trace 调用。

`EvaluationRunner` 负责批量读取测试集、执行问答、计算基础指标并写入 Langfuse。

`Settings` 负责读取模型配置、Langfuse key、索引路径和环境变量。

## 错误处理

上传失败时保留明确错误：不支持的文件类型、解析失败、文件过大。

索引失败时不写入半成品索引；需要记录失败文件和失败原因。

问答时如果没有检索结果，系统返回“知识库中未找到足够依据”，并记录 Langfuse trace。

LLM 或 Langfuse 调用失败不能导致整个服务崩溃。LLM 失败返回可读错误；Langfuse 失败降级为本地日志。

## 测试策略

后端优先写单元测试覆盖文本切分、引用映射、评测指标和 Langfuse 封装。

RAG pipeline 用小型 fixture 文档做集成测试，验证问题能检索到预期 chunk。

前端做轻量手动验收即可，重点是面试 demo 稳定，不追求复杂端到端测试。

## 面试展示重点

这个作品要讲清楚三件事：

1. RAG 不只是把文档塞给模型，而是有解析、切分、检索、prompt、引用和错误降级。
2. AI 应用上线后需要观测和评测，Langfuse 用来记录每次运行，并支持回看失败样本。
3. 评测集能让 prompt 和检索策略有可量化改进，不靠主观感觉调参。

## 后续增强

第二阶段可以加入混合检索 BM25 + embedding、reranker、Qdrant/Chroma 持久化、用户反馈评分、引用高亮、批量对比不同 prompt 版本。

第三阶段可以做权限过滤、部门知识库隔离、成本统计、自动失败样本聚类和 CI 评测门禁。

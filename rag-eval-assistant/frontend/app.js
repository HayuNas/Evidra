const backendUrl = document.querySelector("#backendUrl");
const languageSelect = document.querySelector("#languageSelect");
const statusEl = document.querySelector("#status");
const navButtons = document.querySelectorAll("[data-view-target]");
const viewPanels = document.querySelectorAll("[data-view]");
const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const documentList = document.querySelector("#documentList");
const refreshDocumentsButton = document.querySelector("#refreshDocumentsButton");
const clearDocumentsButton = document.querySelector("#clearDocumentsButton");
const chunkPreview = document.querySelector("#chunkPreview");
const chunkStrategy = document.querySelector("#chunkStrategy");
const chunkSize = document.querySelector("#chunkSize");
const chunkOverlap = document.querySelector("#chunkOverlap");
const retrievalMode = document.querySelector("#retrievalMode");
const topK = document.querySelector("#topK");
const scoreThreshold = document.querySelector("#scoreThreshold");
const askForm = document.querySelector("#askForm");
const questionInput = document.querySelector("#questionInput");
const promptVersion = document.querySelector("#promptVersion");
const queryRewrite = document.querySelector("#queryRewrite");
const multiRoute = document.querySelector("#multiRoute");
const rerankToggle = document.querySelector("#rerankToggle");
const runEvalButton = document.querySelector("#runEvalButton");
const compareStrategiesButton = document.querySelector("#compareStrategiesButton");
const configForm = document.querySelector("#configForm");
const modelGroup = document.querySelector("#modelGroup");
const answerModel = document.querySelector("#answerModel");
const answerApiMode = document.querySelector("#answerApiMode");
const embeddingModel = document.querySelector("#embeddingModel");
const openaiApiKey = document.querySelector("#openaiApiKey");
const openaiBaseUrl = document.querySelector("#openaiBaseUrl");
const embeddingApiKey = document.querySelector("#embeddingApiKey");
const embeddingBaseUrl = document.querySelector("#embeddingBaseUrl");
const rerankProvider = document.querySelector("#rerankProvider");
const rerankModel = document.querySelector("#rerankModel");
const rerankApiKey = document.querySelector("#rerankApiKey");
const rerankBaseUrl = document.querySelector("#rerankBaseUrl");
const answerPromptTemplate = document.querySelector("#answerPromptTemplate");
const rewritePromptTemplate = document.querySelector("#rewritePromptTemplate");
const langfusePublicKey = document.querySelector("#langfusePublicKey");
const langfuseSecretKey = document.querySelector("#langfuseSecretKey");
const langfuseHost = document.querySelector("#langfuseHost");
const configStatus = document.querySelector("#configStatus");
const evalSetSelect = document.querySelector("#evalSetSelect");
const evalSetFileInput = document.querySelector("#evalSetFileInput");
const evalSetNameInput = document.querySelector("#evalSetNameInput");
const uploadEvalSetButton = document.querySelector("#uploadEvalSetButton");
const deleteEvalSetButton = document.querySelector("#deleteEvalSetButton");
const answerText = document.querySelector("#answerText");
const rewrittenQuery = document.querySelector("#rewrittenQuery");
const citationList = document.querySelector("#citationList");
const traceId = document.querySelector("#traceId");
const retrievalLabel = document.querySelector("#retrievalLabel");
const generationProvider = document.querySelector("#generationProvider");
const confidence = document.querySelector("#confidence");
const chunkStat = document.querySelector("#chunkStat");
const citationCount = document.querySelector("#citationCount");
const evalRows = document.querySelector("#evalRows");
const evalSummary = document.querySelector("#evalSummary");
const passRate = document.querySelector("#passRate");
const coverage = document.querySelector("#coverage");
const latency = document.querySelector("#latency");
const comparisonRows = document.querySelector("#comparisonRows");
const comparisonSummary = document.querySelector("#comparisonSummary");
const strategyBars = document.querySelector(".strategy-bars");
const activityLog = document.querySelector("#activityLog");
const feedbackHelpfulButton = document.querySelector("#feedbackHelpfulButton");
const feedbackNotHelpfulButton = document.querySelector("#feedbackNotHelpfulButton");
const feedbackStatus = document.querySelector("#feedbackStatus");

if (backendUrl && window.location.protocol.startsWith("http")) {
  backendUrl.value = window.location.origin;
}

const translations = {
  zh: {
    appTitle: "Evidra",
    languageLabel: "语言",
    uploadTitle: "1. 上传知识文档",
    uploadButton: "上传文档",
    uploadHint: "支持 PDF、Markdown 和 TXT。文件会在本地索引。",
    settingsTitle: "2. 检索设置",
    askTitle: "3. 提出问题",
    askButton: "提问",
    runEvalButton: "运行评测",
    compareButton: "对比策略",
    notRun: "尚未运行",
    confidenceLabel: "置信度",
    answerTitle: "回答",
    answerEmpty: "上传文档并提问后，回答会带着来源引用显示在这里。",
    evidenceTitle: "证据",
    topMatches: "Top matches",
    evalTitle: "评测结果",
    noRunYet: "尚未运行",
    passRate: "通过率",
    coverage: "引用覆盖率",
    latency: "平均延迟",
    questionColumn: "问题",
    expectedTermsColumn: "期望关键词",
    statusColumn: "状态",
    latencyColumn: "延迟",
    evalEmpty: "运行评测后会填充这张表。",
    comparisonTitle: "策略对比",
    noComparisonYet: "尚未对比",
    strategyColumn: "策略",
    passRateColumn: "通过率",
    coverageColumn: "覆盖率",
    comparisonEmpty: "对比策略后会填充这张表。",
    flowUpload: "Upload",
    flowRetrieve: "Retrieve",
    flowCite: "Cite",
    flowEvaluate: "Evaluate",
    chunkStatLabel: "Chunks",
    idle: "空闲",
    working: "处理中",
    ready: "就绪",
    citations: "条引用",
    chunks: "chunks",
    testsPassed: "条测试通过",
    anyCitedAnswer: "任意有引用的回答",
    strategiesCompared: "组策略已对比",
    refreshDocumentsButton: "刷新文档",
    clearDocumentsButton: "清空索引",
    modelTitle: "模型与观测",
    saveConfigButton: "保存配置",
    evalSetTitle: "评测集",
    evalSetLabel: "当前评测集",
    evalSetUploadLabel: "上传 JSON",
    evalSetNameLabel: "保存为",
    uploadEvalSetButton: "上传评测集",
    deleteEvalSetButton: "删除评测集",
    logTitle: "运行日志",
    logHint: "请求和错误说明",
    logReadyTitle: "前端已就绪",
    logReadyDetail: "等待上传、问答或评测请求。",
    logRequestTitle: "正在请求",
    logSuccessTitle: "请求成功",
    logErrorTitle: "请求失败",
    backendOffline: "后端未连接",
    requestFailedShort: "请求失败",
    networkErrorHelp:
      "无法连接后端。请确认 uvicorn 正在运行，Backend URL 端口正确，例如 http://127.0.0.1:8010，并检查 /health。",
    apiErrorHelp: "后端返回了错误响应。请查看接口路径、请求参数和服务端终端日志。",
    genericErrorHelp: "请求未完成。请检查 Backend URL、浏览器控制台和后端终端输出。",
  },
  en: {
    appTitle: "Evidra",
    languageLabel: "Language",
    uploadTitle: "1. Upload knowledge documents",
    uploadButton: "Upload document",
    uploadHint: "PDF, Markdown, and TXT. Files are indexed locally.",
    settingsTitle: "2. Retrieval settings",
    askTitle: "3. Ask a question",
    askButton: "Ask",
    runEvalButton: "Run evaluation",
    compareButton: "Compare strategies",
    notRun: "Not run",
    confidenceLabel: "Confidence",
    answerTitle: "Answer",
    answerEmpty: "Upload a document, ask a question, and the answer will appear here with source-backed citations.",
    evidenceTitle: "Evidence",
    topMatches: "Top matches",
    evalTitle: "Evaluation results",
    noRunYet: "No run yet",
    passRate: "Pass rate",
    coverage: "Citation coverage",
    latency: "Avg latency",
    questionColumn: "Question",
    expectedTermsColumn: "Expected terms",
    statusColumn: "Status",
    latencyColumn: "Latency",
    evalEmpty: "Run an evaluation to populate this table.",
    comparisonTitle: "Strategy comparison",
    noComparisonYet: "No comparison yet",
    strategyColumn: "Strategy",
    passRateColumn: "Pass rate",
    coverageColumn: "Coverage",
    comparisonEmpty: "Compare strategies to populate this table.",
    flowUpload: "Upload",
    flowRetrieve: "Retrieve",
    flowCite: "Cite",
    flowEvaluate: "Evaluate",
    chunkStatLabel: "Chunks",
    idle: "Idle",
    working: "Working",
    ready: "Ready",
    citations: "citations",
    chunks: "chunks",
    testsPassed: "tests passed",
    anyCitedAnswer: "Any cited answer",
    strategiesCompared: "strategies compared",
    refreshDocumentsButton: "Refresh docs",
    clearDocumentsButton: "Clear index",
    modelTitle: "Model and observability",
    saveConfigButton: "Save config",
    evalSetTitle: "Evaluation set",
    evalSetLabel: "Current set",
    evalSetUploadLabel: "Upload JSON",
    evalSetNameLabel: "Save as",
    uploadEvalSetButton: "Upload eval set",
    deleteEvalSetButton: "Delete eval set",
    logTitle: "Activity log",
    logHint: "Requests and error notes",
    logReadyTitle: "Frontend ready",
    logReadyDetail: "Waiting for upload, question, or evaluation request.",
    logRequestTitle: "Requesting",
    logSuccessTitle: "Request succeeded",
    logErrorTitle: "Request failed",
    backendOffline: "Backend offline",
    requestFailedShort: "Request failed",
    networkErrorHelp:
      "The frontend cannot reach the backend. Confirm uvicorn is running, the Backend URL port is correct, for example http://127.0.0.1:8010, and check /health.",
    apiErrorHelp: "The backend returned an error response. Check the API path, request parameters, and server terminal logs.",
    genericErrorHelp: "The request did not complete. Check the Backend URL, browser console, and backend terminal output.",
  },
};

let currentLanguage = localStorage.getItem("evidraLanguage") || localStorage.getItem("ragEvalLanguage") || "zh";
let latestTraceId = "";

[openaiApiKey, embeddingApiKey, rerankApiKey, langfusePublicKey, langfuseSecretKey]
  .filter(Boolean)
  .forEach((input) => {
    input.dataset.emptyPlaceholder = input.placeholder;
  });

function t(key) {
  return translations[currentLanguage][key] || translations.en[key] || key;
}

function clear(element) {
  element.replaceChildren();
}

function setFlowStep(step) {
  document.querySelectorAll("[data-flow-step]").forEach((element) => {
    const steps = ["upload", "retrieve", "cite", "evaluate"];
    const currentIndex = steps.indexOf(step);
    const elementIndex = steps.indexOf(element.dataset.flowStep);
    element.classList.toggle("active", elementIndex <= currentIndex);
  });
}

function switchView(viewName) {
  viewPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.view === viewName);
  });
  navButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.viewTarget === viewName);
  });
}

function endpoint(path) {
  return `${backendUrl.value.replace(/\/$/, "")}${path}`;
}

function chunkingConfig() {
  return {
    strategy: chunkStrategy.value,
    chunk_size: Number(chunkSize.value),
    overlap: Number(chunkOverlap.value),
  };
}

function retrievalConfig() {
  return {
    mode: retrievalMode.value,
    top_k: Number(topK.value),
    score_threshold: Number(scoreThreshold.value),
  };
}

function currentPromptVersion() {
  return promptVersion?.value || "grounded";
}

function rewriteQueryEnabled() {
  return queryRewrite?.value === "llm";
}

function multiRouteEnabled() {
  return multiRoute?.value === "on";
}

function rerankEnabled() {
  return rerankToggle?.value === "on";
}

function setStatus(text, state = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${state}`;
}

function setStatusKey(key, state = "") {
  statusEl.dataset.statusKey = key;
  setStatus(t(key), state);
}

function addLog(kind, title, detail) {
  if (!activityLog) return;
  const item = document.createElement("li");
  item.className = `log-entry ${kind}`;

  const meta = document.createElement("div");
  meta.className = "log-meta";
  const label = document.createElement("strong");
  label.textContent = title;
  const time = document.createElement("time");
  time.dateTime = new Date().toISOString();
  time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  meta.append(label, time);

  const body = document.createElement("p");
  body.textContent = detail;
  item.append(meta, body);
  activityLog.prepend(item);

  while (activityLog.children.length > 8) {
    activityLog.lastElementChild.remove();
  }
}

function isNetworkError(error) {
  const message = String(error?.message || error || "");
  return message.includes("Failed to fetch") || message.includes("NetworkError") || error instanceof TypeError;
}

function explainError(error) {
  const message = String(error?.message || error || "");
  if (isNetworkError(error)) {
    return t("networkErrorHelp");
  }
  if (error?.apiStatus) {
    return `${t("apiErrorHelp")} HTTP ${error.apiStatus}: ${message}`;
  }
  return `${t("genericErrorHelp")} ${message}`.trim();
}

function applyLanguage(language) {
  currentLanguage = translations[language] ? language : "zh";
  localStorage.setItem("evidraLanguage", currentLanguage);
  document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
  languageSelect.value = currentLanguage;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  if (statusEl.dataset.statusKey) {
    const state = statusEl.classList.contains("error") ? "error" : statusEl.classList.contains("busy") ? "busy" : "";
    setStatusKey(statusEl.dataset.statusKey, state);
  }
}

async function request(path, options) {
  setStatusKey("working", "busy");
  const method = options?.method || "GET";
  addLog("info", t("logRequestTitle"), `${method} ${path} -> ${endpoint(path)}`);
  try {
    const response = await fetch(endpoint(path), options);
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text };
      }
    }
    if (!response.ok) {
      const error = new Error(payload.detail || "Request failed");
      error.apiStatus = response.status;
      throw error;
    }
    setStatusKey("ready");
    addLog("success", t("logSuccessTitle"), `${method} ${path}`);
    return payload;
  } catch (error) {
    const detail = explainError(error);
    setStatus(isNetworkError(error) ? t("backendOffline") : t("requestFailedShort"), "error");
    addLog("error", t("logErrorTitle"), detail);
    throw error;
  }
}

function selectedEvalSet() {
  return evalSetSelect.value || "demo.json";
}

function renderConfigStatus(config, observability) {
  const provider = `${config.model_group} / answer=${config.answer_provider} / embedding=${config.embedding_provider}`;
  const answerMode = `mode=${config.answer_api_mode || "auto"}`;
  const openai = config.openai_configured ? "OpenAI configured" : "OpenAI local fallback";
  const baseUrl = config.openai_base_url || "https://api.openai.com";
  const embedding = config.embedding_configured ? "Embedding configured" : "Embedding fallback";
  const embeddingUrl = config.embedding_base_url || baseUrl;
  const rerank = config.rerank_configured ? `Rerank ${config.rerank_provider} configured` : "Rerank off/fallback";
  const langfuse = observability
    ? `Langfuse ${observability.mode}${observability.healthy ? " healthy" : " fallback"}`
    : "Langfuse unknown";
  configStatus.textContent = `${provider}; ${answerMode}; ${openai}; answer=${baseUrl}; ${embedding}; embedding=${embeddingUrl}; ${rerank}; ${langfuse}`;
}

function applySecretPlaceholder(input, configured) {
  if (!input) return;
  input.value = "";
  input.dataset.configured = configured ? "true" : "false";
  input.placeholder = configured ? "•••••••• 已配置" : input.dataset.emptyPlaceholder || input.placeholder;
}

async function loadConfig() {
  const config = await request("/config");
  modelGroup.value = config.model_group;
  answerApiMode.value = config.answer_api_mode || "auto";
  answerModel.value = config.answer_model;
  embeddingModel.value = config.embedding_model;
  openaiBaseUrl.value = config.openai_base_url || "https://api.openai.com";
  embeddingBaseUrl.value = config.embedding_base_url || config.openai_base_url || "https://dashscope.aliyuncs.com/compatible-mode/v1";
  rerankProvider.value = config.rerank_provider || "none";
  rerankModel.value = config.rerank_model || "gte-rerank-v2";
  rerankBaseUrl.value = config.rerank_base_url || "https://dashscope.aliyuncs.com";
  answerPromptTemplate.value = config.answer_prompt_template || "";
  rewritePromptTemplate.value = config.rewrite_prompt_template || "";
  langfuseHost.value = config.langfuse_host || "https://cloud.langfuse.com";
  applySecretPlaceholder(openaiApiKey, config.openai_configured);
  applySecretPlaceholder(embeddingApiKey, config.embedding_configured);
  applySecretPlaceholder(rerankApiKey, config.rerank_configured);
  applySecretPlaceholder(langfuseSecretKey, config.langfuse_configured);
  applySecretPlaceholder(langfusePublicKey, config.langfuse_configured);
  const observability = await request("/observability/status");
  renderConfigStatus(config, observability);
}

async function loadDocuments() {
  const documents = await request("/documents");
  renderDocuments(documents);
}

function renderDocuments(documents) {
  clear(documentList);
  if (!documents.length) {
    const item = document.createElement("li");
    const empty = document.createElement("span");
    empty.textContent = "No documents indexed";
    item.appendChild(empty);
    documentList.appendChild(item);
    return;
  }
  documents.forEach((documentInfo) => {
    const item = document.createElement("li");
    const label = document.createElement("button");
    label.type = "button";
    label.className = "secondary";
    label.textContent = documentInfo.source;
    label.addEventListener("click", () => loadDocumentChunks(documentInfo.source));
    const chunks = document.createElement("strong");
    chunks.textContent = `${documentInfo.chunk_count} ${t("chunks")}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      await request(`/documents/${encodeURIComponent(documentInfo.source)}`, { method: "DELETE" });
      clear(chunkPreview);
      await loadDocuments();
    });
    item.append(label, chunks, remove);
    documentList.appendChild(item);
  });
}

async function loadDocumentChunks(source) {
  const chunks = await request(`/documents/${encodeURIComponent(source)}/chunks`);
  clear(chunkPreview);
  chunks.slice(0, 5).forEach((chunk) => {
    const card = document.createElement("div");
    card.className = "chunk-card";
    const title = document.createElement("strong");
    title.textContent = chunk.id;
    const text = document.createElement("p");
    text.textContent = chunk.text;
    card.append(title, text);
    chunkPreview.appendChild(card);
  });
}

async function loadEvaluationSets() {
  const sets = await request("/evaluation-sets");
  clear(evalSetSelect);
  sets.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${item.name} (${item.item_count})`;
    evalSetSelect.appendChild(option);
  });
  if (!sets.length) {
    const option = document.createElement("option");
    option.value = "demo.json";
    option.textContent = "demo.json";
    evalSetSelect.appendChild(option);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;
  const data = new FormData();
  data.append("file", fileInput.files[0]);
  const chunking = chunkingConfig();
  data.append("chunk_strategy", chunking.strategy);
  data.append("chunk_size", String(chunking.chunk_size));
  data.append("chunk_overlap", String(chunking.overlap));
  const result = await request("/documents", { method: "POST", body: data });
  const item = document.createElement("li");
  const source = document.createElement("span");
  source.textContent = result.source;
  const chunks = document.createElement("strong");
  chunks.textContent = `${result.chunk_count} ${t("chunks")}`;
  item.append(source, chunks);
  documentList.prepend(item);
  chunkStat.textContent = String(result.chunk_count);
  setFlowStep("upload");
  await loadDocuments();
});

refreshDocumentsButton.addEventListener("click", loadDocuments);

clearDocumentsButton.addEventListener("click", async () => {
  await request("/documents", { method: "DELETE" });
  clear(chunkPreview);
  await loadDocuments();
});

configForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    model_group: modelGroup.value,
    answer_provider: modelGroup.value === "local" ? "local" : "openai",
    embedding_provider: modelGroup.value === "local" ? "local" : "openai",
    answer_api_mode: answerApiMode.value,
    answer_model: answerModel.value,
    embedding_model: embeddingModel.value,
    openai_base_url: openaiBaseUrl.value,
    embedding_base_url: embeddingBaseUrl.value,
    answer_prompt_template: answerPromptTemplate.value,
    rewrite_prompt_template: rewritePromptTemplate.value,
    rerank_provider: rerankProvider.value,
    rerank_model: rerankModel.value,
    rerank_base_url: rerankBaseUrl.value,
    langfuse_host: langfuseHost.value,
  };
  if (openaiApiKey.value) payload.openai_api_key = openaiApiKey.value;
  if (embeddingApiKey.value) payload.embedding_api_key = embeddingApiKey.value;
  if (rerankApiKey.value) payload.rerank_api_key = rerankApiKey.value;
  if (langfusePublicKey.value) payload.langfuse_public_key = langfusePublicKey.value;
  if (langfuseSecretKey.value) payload.langfuse_secret_key = langfuseSecretKey.value;
  const config = await request("/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  openaiApiKey.value = "";
  embeddingApiKey.value = "";
  rerankApiKey.value = "";
  langfuseSecretKey.value = "";
  langfusePublicKey.value = "";
  applySecretPlaceholder(openaiApiKey, config.openai_configured);
  applySecretPlaceholder(embeddingApiKey, config.embedding_configured);
  applySecretPlaceholder(rerankApiKey, config.rerank_configured);
  applySecretPlaceholder(langfuseSecretKey, config.langfuse_configured);
  applySecretPlaceholder(langfusePublicKey, config.langfuse_configured);
  const observability = await request("/observability/status");
  renderConfigStatus(config, observability);
});

uploadEvalSetButton.addEventListener("click", async () => {
  if (!evalSetFileInput.files.length) return;
  const raw = await evalSetFileInput.files[0].text();
  const items = JSON.parse(raw);
  await request("/evaluation-sets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: evalSetNameInput.value || evalSetFileInput.files[0].name, items }),
  });
  await loadEvaluationSets();
});

deleteEvalSetButton.addEventListener("click", async () => {
  await request(`/evaluation-sets/${encodeURIComponent(selectedEvalSet())}`, { method: "DELETE" });
  await loadEvaluationSets();
});

languageSelect.addEventListener("change", () => {
  applyLanguage(languageSelect.value);
});

navButtons.forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.viewTarget));
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = await request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: questionInput.value,
      prompt_version: currentPromptVersion(),
      rewrite_query: rewriteQueryEnabled(),
      multi_route: multiRouteEnabled(),
      rerank: rerankEnabled(),
      retrieval: retrievalConfig(),
    }),
  });
  renderAnswer(result);
  setFlowStep("cite");
});

runEvalButton.addEventListener("click", async () => {
  const result = await request("/evaluations/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      eval_set: selectedEvalSet(),
      prompt_version: currentPromptVersion(),
      rewrite_query: rewriteQueryEnabled(),
      multi_route: multiRouteEnabled(),
      rerank: rerankEnabled(),
      retrieval: retrievalConfig(),
    }),
  });
  renderEvaluation(result);
  setFlowStep("evaluate");
});

compareStrategiesButton.addEventListener("click", async () => {
  const result = await request("/evaluations/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      eval_set: selectedEvalSet(),
      strategies: [
        { name: "tfidf-k2", rewrite_query: rewriteQueryEnabled(), multi_route: false, rerank: false, retrieval: { mode: "tfidf", top_k: 2, score_threshold: 0 } },
        { name: "tfidf-k4", rewrite_query: rewriteQueryEnabled(), multi_route: false, rerank: rerankEnabled(), retrieval: { mode: "tfidf", top_k: 4, score_threshold: 0 } },
        { name: "bm25-k4", rewrite_query: rewriteQueryEnabled(), multi_route: false, rerank: rerankEnabled(), retrieval: { mode: "bm25", top_k: 4, score_threshold: 0 } },
        { name: "hybrid-k4", prompt_version: currentPromptVersion(), rewrite_query: rewriteQueryEnabled(), multi_route: false, rerank: rerankEnabled(), retrieval: { mode: "hybrid", top_k: 4, score_threshold: 0 } },
        { name: "multi-route", prompt_version: currentPromptVersion(), rewrite_query: rewriteQueryEnabled(), multi_route: true, rerank: rerankEnabled(), retrieval: { mode: "hybrid", top_k: 4, score_threshold: 0 } },
        { name: "multi-route-rerank", prompt_version: "analyst", rewrite_query: true, multi_route: true, rerank: true, retrieval: { mode: "hybrid", top_k: 4, score_threshold: 0 } },
      ],
    }),
  });
  renderComparison(result);
  setFlowStep("evaluate");
});

function renderAnswer(result) {
  latestTraceId = result.trace_id || "";
  renderAnswerWithCitationTokens(result.answer, result.citations);
  traceId.textContent = latestTraceId || "local";
  const retrieval = result.metadata?.retrieval;
  const generation = result.metadata?.generation;
  const prompt = result.metadata?.prompt;
  const query = result.metadata?.query;
  const routeLabel = `${query?.rewrite_enabled ? " + rewrite" : ""}${retrieval?.multi_route ? " + multi" : ""}${result.metadata?.rerank?.enabled ? " + rerank" : ""}`;
  retrievalLabel.textContent = retrieval ? `${retrieval.mode} k=${retrieval.top_k}${routeLabel}` : "local";
  generationProvider.textContent = generation ? `${generation.provider} / ${generation.model} / ${prompt?.version || "grounded"}` : `local / ${prompt?.version || "grounded"}`;
  renderRewrittenQuery(query);
  confidence.textContent = Number(result.confidence).toFixed(2);
  citationCount.textContent = `${result.citations.length} ${t("citations")}`;
  clear(citationList);
  result.citations.forEach((citation, index) => {
    const item = document.createElement("li");
    item.dataset.index = String(index + 1);
    item.dataset.chunkId = citation.chunk_id;
    const citationSource = document.createElement("div");
    citationSource.className = "citation-source";
    const sourceName = document.createElement("span");
    sourceName.textContent = citation.source;
    const score = document.createElement("span");
    score.textContent = Number(citation.score).toFixed(2);
    citationSource.append(sourceName, score);

    const snippet = document.createElement("p");
    snippet.className = "snippet";
    snippet.textContent = citation.snippet;

    item.append(citationSource, snippet);
    item.addEventListener("click", () => highlightCitation(citation.chunk_id));
    citationList.appendChild(item);
  });
  if (feedbackStatus) feedbackStatus.textContent = result.trace_id ? "可提交反馈" : "等待 trace";
}

function renderRewrittenQuery(query) {
  if (!rewrittenQuery) return;
  if (!query?.rewrite_enabled) {
    rewrittenQuery.hidden = true;
    rewrittenQuery.textContent = "";
    return;
  }
  rewrittenQuery.hidden = false;
  rewrittenQuery.textContent = `查询改写 Query rewrite: ${query.original} -> ${query.rewritten} (${query.rewrite_provider} / ${query.rewrite_model})`;
}

function renderAnswerWithCitationTokens(answer, citations) {
  clear(answerText);
  const citationIds = new Set(citations.map((citation) => citation.chunk_id));
  const pattern = /\[([^\]]+)\]/g;
  let lastIndex = 0;
  let match = pattern.exec(answer);
  while (match) {
    if (match.index > lastIndex) {
      answerText.appendChild(document.createTextNode(answer.slice(lastIndex, match.index)));
    }
    const tokenText = match[1];
    if (citationIds.has(tokenText)) {
      const token = document.createElement("button");
      token.type = "button";
      token.className = "citation-token";
      token.textContent = `[${tokenText}]`;
      token.addEventListener("click", () => highlightCitation(tokenText));
      answerText.appendChild(token);
    } else {
      answerText.appendChild(document.createTextNode(match[0]));
    }
    lastIndex = pattern.lastIndex;
    match = pattern.exec(answer);
  }
  if (lastIndex < answer.length) {
    answerText.appendChild(document.createTextNode(answer.slice(lastIndex)));
  }
}

function highlightCitation(chunkId) {
  citationList.querySelectorAll("li").forEach((item) => {
    item.classList.toggle("active", item.dataset.chunkId === chunkId);
  });
  answerText.querySelectorAll(".citation-token").forEach((token) => {
    token.classList.toggle("active", token.textContent === `[${chunkId}]`);
  });
}

async function sendFeedback(label, value) {
  if (!latestTraceId) {
    if (feedbackStatus) feedbackStatus.textContent = "请先完成一次问答";
    return;
  }
  const result = await request("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: latestTraceId, label, value }),
  });
  if (feedbackStatus) feedbackStatus.textContent = `已记录 ${result.feedback_id}`;
}

function renderEvaluation(summary) {
  evalSummary.textContent = `${summary.passed} / ${summary.total} ${t("testsPassed")}`;
  passRate.textContent = `${Math.round(summary.pass_rate * 100)}%`;
  coverage.textContent = `${Math.round(summary.citation_coverage * 100)}%`;
  latency.textContent = `${summary.average_latency_ms} ms`;
  clear(evalRows);
  summary.results.forEach((result) => {
    const row = document.createElement("tr");
    const question = document.createElement("td");
    question.textContent = result.question;
    const expectedTerms = document.createElement("td");
    expectedTerms.textContent = result.expected_terms.join(", ") || t("anyCitedAnswer");
    const status = document.createElement("td");
    status.className = result.passed ? "pass" : "fail";
    status.textContent = result.passed ? "PASS" : "FAIL";
    const rowLatency = document.createElement("td");
    rowLatency.textContent = `${Number(result.latency_ms).toFixed(1)} ms`;
    row.append(question, expectedTerms, status, rowLatency);
    evalRows.appendChild(row);
  });
}

function renderComparison(result) {
  comparisonSummary.textContent = `${result.comparisons.length} ${t("strategiesCompared")}`;
  clear(comparisonRows);
  clear(strategyBars);
  result.comparisons.forEach((comparison) => {
    const row = document.createElement("tr");
    const summary = comparison.summary;
    const retrieval = comparison.retrieval;
    [
      comparison.name,
      retrieval.mode,
      retrieval.top_k,
      comparison.prompt?.version || "grounded",
      `${Math.round(summary.pass_rate * 100)}%`,
      `${Math.round(summary.citation_coverage * 100)}%`,
      `${summary.average_latency_ms} ms`,
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    });
    comparisonRows.appendChild(row);

    const bar = document.createElement("div");
    bar.className = "strategy-bar";
    const label = document.createElement("span");
    label.textContent = comparison.name;
    const track = document.createElement("div");
    track.className = "strategy-track";
    const fill = document.createElement("div");
    fill.className = "strategy-fill";
    fill.style.width = `${Math.round(summary.pass_rate * 100)}%`;
    const value = document.createElement("strong");
    value.textContent = `${Math.round(summary.pass_rate * 100)}%`;
    track.appendChild(fill);
    bar.append(label, track, value);
    strategyBars.appendChild(bar);
  });
}

feedbackHelpfulButton?.addEventListener("click", () => sendFeedback("helpful", 1.0));
feedbackNotHelpfulButton?.addEventListener("click", () => sendFeedback("not_helpful", 0.0));

applyLanguage(currentLanguage);
addLog("info", t("logReadyTitle"), t("logReadyDetail"));
loadConfig().catch(() => {});
loadDocuments().catch(() => {});
loadEvaluationSets().catch(() => {});

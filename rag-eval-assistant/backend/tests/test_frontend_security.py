import unittest
from pathlib import Path


def frontend_app_js() -> Path:
    for base in (Path.cwd(), Path.cwd().parent):
        candidate = base / "frontend" / "app.js"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate frontend/app.js from the test working directory.")


def frontend_index_html() -> Path:
    for base in (Path.cwd(), Path.cwd().parent):
        candidate = base / "frontend" / "index.html"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate frontend/index.html from the test working directory.")


class TestFrontendSecurity(unittest.TestCase):
    def test_app_does_not_render_api_content_with_inner_html(self):
        source = frontend_app_js().read_text(encoding="utf-8")

        self.assertNotIn("innerHTML = `", source)

    def test_frontend_contains_strategy_controls(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")

        self.assertIn('id="chunkStrategy"', index_html)
        self.assertIn('id="retrievalMode"', index_html)
        self.assertIn('id="topK"', index_html)
        self.assertIn('id="compareStrategiesButton"', index_html)
        self.assertIn("分块策略 Chunk strategy", index_html)
        self.assertIn("标题感知 Heading-aware", index_html)
        self.assertIn("检索模式 Retrieval mode", index_html)
        self.assertIn("混合检索 Hybrid", index_html)
        self.assertIn('<option value="hybrid" selected>', index_html)

    def test_frontend_contains_comparison_table(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")

        self.assertIn('id="compareStrategiesButton"', index_html)
        self.assertIn('id="comparisonRows"', index_html)

    def test_frontend_contains_language_switcher(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")

        self.assertIn('id="languageSelect"', index_html)
        self.assertIn('data-i18n="appTitle"', index_html)
        self.assertIn("const translations", app_js)
        self.assertIn("localStorage", app_js)
        self.assertIn("上传知识文档", app_js)

    def test_frontend_contains_visual_refresh_regions(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="workflow-strip"', index_html)
        self.assertIn('data-flow-step="retrieve"', index_html)
        self.assertIn('id="chunkStat"', index_html)
        self.assertIn('class="strategy-bars"', index_html)
        self.assertIn("@keyframes", styles)

    def test_frontend_uses_second_visual_refresh_without_retrieval_map(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="workflow-strip"', index_html)
        self.assertIn('class="strategy-bars"', index_html)
        self.assertNotIn('id="retrievalMap"', index_html)
        self.assertNotIn("function animateRetrievalMap", app_js)
        self.assertNotIn("@keyframes mapPulse", styles)

    def test_frontend_contains_activity_log(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")

        self.assertIn('id="activityLog"', index_html)
        self.assertIn('data-i18n="logTitle"', index_html)
        self.assertIn("function addLog", app_js)
        self.assertIn("explainError", app_js)

    def test_frontend_upload_button_is_constrained_and_fonts_are_modern(self):
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".upload-box button", styles)
        self.assertIn("max-width: 100%", styles)
        self.assertIn('"Segoe UI Variable Display"', styles)
        self.assertNotIn("Georgia", styles)

    def test_frontend_contains_model_evalset_and_document_management_controls(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")

        self.assertIn('id="configForm"', index_html)
        self.assertIn('id="modelGroup"', index_html)
        self.assertIn('id="openaiApiKey"', index_html)
        self.assertIn('id="openaiBaseUrl"', index_html)
        self.assertIn('id="embeddingApiKey"', index_html)
        self.assertIn('id="embeddingBaseUrl"', index_html)
        self.assertIn('id="langfuseSecretKey"', index_html)
        self.assertIn('id="evalSetSelect"', index_html)
        self.assertIn('id="generationProvider"', index_html)
        self.assertIn('id="refreshDocumentsButton"', index_html)
        self.assertIn("function loadConfig", app_js)
        self.assertIn("generation.provider", app_js)
        self.assertIn("openai_base_url", app_js)
        self.assertIn("embedding_base_url", app_js)
        self.assertIn("embedding_api_key", app_js)
        self.assertIn("function loadEvaluationSets", app_js)
        self.assertIn("function loadDocuments", app_js)

    def test_frontend_contains_second_priority_prompt_feedback_and_citation_highlight(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn('id="promptVersion"', index_html)
        self.assertIn('id="feedbackHelpfulButton"', index_html)
        self.assertIn('id="feedbackNotHelpfulButton"', index_html)
        self.assertIn("prompt_version", app_js)
        self.assertIn("function renderAnswerWithCitationTokens", app_js)
        self.assertIn("function highlightCitation", app_js)
        self.assertIn("function sendFeedback", app_js)
        self.assertIn(".citation-token", styles)
        self.assertIn(".citations li.active", styles)

    def test_frontend_contains_query_rewrite_controls(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn('id="queryRewrite"', index_html)
        self.assertIn('id="rewrittenQuery"', index_html)
        self.assertIn("查询改写 Query rewrite", index_html)
        self.assertIn("rewrite_query", app_js)
        self.assertIn("function rewriteQueryEnabled", app_js)
        self.assertIn("function renderRewrittenQuery", app_js)
        self.assertIn(".rewritten-query", styles)

    def test_frontend_contains_top_workspace_navigation_and_settings_page(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")
        styles = (frontend_index_html().parent / "styles.css").read_text(encoding="utf-8")

        self.assertIn('class="top-nav"', index_html)
        self.assertIn('data-view-target="workbench"', index_html)
        self.assertIn('data-view-target="settings"', index_html)
        self.assertIn('data-view-target="evaluation"', index_html)
        self.assertIn('data-view-target="logs"', index_html)
        self.assertIn('data-view="settings"', index_html)
        self.assertIn("function switchView", app_js)
        self.assertIn(".view-panel", styles)

    def test_frontend_contains_prompt_rerank_and_configured_secret_placeholders(self):
        index_html = frontend_index_html().read_text(encoding="utf-8")
        app_js = frontend_app_js().read_text(encoding="utf-8")

        self.assertIn('id="answerPromptTemplate"', index_html)
        self.assertIn('id="rewritePromptTemplate"', index_html)
        self.assertIn('id="rerankProvider"', index_html)
        self.assertIn('id="rerankModel"', index_html)
        self.assertIn('id="rerankApiKey"', index_html)
        self.assertIn('id="rerankBaseUrl"', index_html)
        self.assertIn("answer_prompt_template", app_js)
        self.assertIn("rewrite_prompt_template", app_js)
        self.assertIn("rerank_provider", app_js)
        self.assertIn("applySecretPlaceholder", app_js)
        self.assertIn("•••••••• 已配置", app_js)

if __name__ == "__main__":
    unittest.main()

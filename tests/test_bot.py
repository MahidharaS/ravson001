from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_vision_bot.config import Settings
from rag_vision_bot.knowledge import KnowledgeBase, make_snippet
from rag_vision_bot.models import RetrievedChunk
from rag_vision_bot.providers import KeywordEmbedder, OllamaVisionProvider, extract_tags_from_text
from rag_vision_bot.services import BotService
from rag_vision_bot.storage import Storage
from rag_vision_bot.telegram_bot import (
    TelegramHttpClient,
    extract_command,
    has_non_image_document,
    scope_from_menu_text,
    source_prefixes_for_scope,
)


def make_settings(root: Path) -> Settings:
    return Settings(
        project_root=root,
        data_dir=root / "data",
        db_path=root / "data" / "bot.db",
        knowledge_dir=root / "knowledge_base",
        transport_mode="cli",
        telegram_bot_token=None,
        telegram_api_base="https://api.telegram.org",
        telegram_poll_timeout=1,
        max_image_bytes=1024 * 1024,
        auto_caption_photos=True,
        llm_provider="none",
        llm_model="llama3.2:3b",
        embedding_provider="keyword",
        embedding_model="nomic-embed-text",
        vision_provider="none",
        vision_model="llava:7b",
        sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2",
        transformers_vision_model="Salesforce/blip-image-captioning-base",
        ollama_base_url="http://localhost:11434",
        request_timeout_seconds=5,
        top_k=3,
        chunk_target_chars=500,
        chunk_overlap_chars=80,
        max_context_chars=1800,
        max_history_turns=3,
        cache_ttl_seconds=3600,
        enable_history=True,
        enable_cache=True,
        enable_vision=True,
        enable_summarize=True,
    )


def seed_docs(path: Path) -> None:
    (path / "knowledge_base").mkdir(parents=True, exist_ok=True)
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "knowledge_base" / "leave.md").write_text(
        "# Leave Policy\n\nUnused leave can be carried forward up to 10 days into the next year.\n",
        encoding="utf-8",
    )
    (path / "knowledge_base" / "expense.md").write_text(
        "# Expense Policy\n\nAny single expense above 10,000 INR requires manager approval before purchase.\n",
        encoding="utf-8",
    )


class KnowledgeBaseTests(unittest.TestCase):
    def test_keyword_retrieval_finds_relevant_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            storage = Storage(settings.db_path)
            knowledge = KnowledgeBase(settings, storage, KeywordEmbedder())
            knowledge.ensure_index()

            results = knowledge.search("carry forward leave", top_k=2)

            self.assertTrue(results)
            self.assertEqual(results[0].source_title, "Leave Policy")

    def test_nested_subfolders_are_indexed_with_relative_source_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            nested_dir = root / "knowledge_base" / "company" / "candidate_profiles"
            nested_dir.mkdir(parents=True, exist_ok=True)
            resume_path = nested_dir / "resume.md"
            resume_path.write_text(
                "# Candidate Resume\n\nBuilt analytics dashboards and production Python automation.\n",
                encoding="utf-8",
            )
            settings = make_settings(root)
            storage = Storage(settings.db_path)
            knowledge = KnowledgeBase(settings, storage, KeywordEmbedder())

            manifest = knowledge.ensure_index()
            results = knowledge.search("analytics dashboards python", top_k=3)

            self.assertIn("company/candidate_profiles/resume.md", manifest["documents"])
            self.assertTrue(results)
            self.assertEqual(results[0].source_key, "company/candidate_profiles/resume.md")
            self.assertEqual(results[0].source_title, "Candidate Resume")

    def test_search_can_be_limited_to_a_scope_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            (root / "knowledge_base" / "company").mkdir(parents=True, exist_ok=True)
            (root / "knowledge_base" / "My_Profile").mkdir(parents=True, exist_ok=True)
            (root / "knowledge_base" / "company" / "financials.md").write_text(
                "# Company Financials\n\nThe company is focused on profitability and capital efficiency.\n",
                encoding="utf-8",
            )
            (root / "knowledge_base" / "My_Profile" / "profile.md").write_text(
                "# My Profile\n\nBuilt Python automation and analytics workflows.\n",
                encoding="utf-8",
            )
            settings = make_settings(root)
            storage = Storage(settings.db_path)
            knowledge = KnowledgeBase(settings, storage, KeywordEmbedder())
            knowledge.ensure_index()

            company_results = knowledge.search(
                "profitability capital efficiency",
                top_k=3,
                source_prefixes=["company/"],
            )
            profile_results = knowledge.search(
                "python automation analytics",
                top_k=3,
                source_prefixes=["my_profile/"],
            )

            self.assertTrue(company_results)
            self.assertTrue(profile_results)
            self.assertTrue(all(result.source_key.lower().startswith("company/") for result in company_results))
            self.assertTrue(all(result.source_key.lower().startswith("my_profile/") for result in profile_results))


class BotServiceTests(unittest.TestCase):
    def test_ask_includes_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)
            service.prepare()

            response = service.ask("user-1", "chat-1", "What is the carry forward policy?")

            self.assertIn("References:", response.answer)
            self.assertIn("Leave Policy", response.answer)

    def test_empty_query_returns_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)

            response = service.ask("user-1", "chat-1", "   ")

            self.assertEqual(response.answer, "Usage: /ask <question>")
            self.assertEqual(response.sources, [])
            self.assertEqual(response.provider, "none")

    def test_summarize_returns_latest_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)
            service.prepare()
            service.ask("user-1", "chat-1", "What is the carry forward policy?")

            summary = service.summarize("user-1", "chat-1")

            self.assertIn("Latest mode", summary.summary)

    def test_image_flow_degrades_cleanly_when_provider_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)
            service.prepare()

            response = service.describe_image(
                user_id="user-1",
                chat_id="chat-1",
                image_bytes=b"fake-image-bytes",
                mime_type="image/jpeg",
            )

            self.assertIn("Vision backend is unavailable", response.caption)

    def test_waiting_for_image_session_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)

            service.mark_waiting_for_image("user-1", "chat-1")
            self.assertTrue(service.is_waiting_for_image("user-1", "chat-1"))

            service.clear_waiting_for_image("user-1", "chat-1")
            self.assertFalse(service.is_waiting_for_image("user-1", "chat-1"))

    def test_query_scope_persists_when_image_wait_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)

            service.set_query_scope("user-1", "chat-1", "company")
            service.mark_waiting_for_image("user-1", "chat-1")
            self.assertEqual(service.get_query_scope("user-1", "chat-1"), "company")

            service.clear_waiting_for_image("user-1", "chat-1")
            self.assertEqual(service.get_query_scope("user-1", "chat-1"), "company")

    def test_formatted_sources_use_clean_references_without_snippet_dump(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_docs(root)
            settings = make_settings(root)
            service = BotService(settings)
            sources = [
                RetrievedChunk(
                    source_key="My_Profile/Mahidhara_S_Profile.md",
                    source_title="👤 Mahidhara S — Complete Professional Profile",
                    chunk_index=0,
                    content="profile content",
                    score=0.9,
                    snippet="ting Strategist Program and fragmented markdown",
                ),
                RetrievedChunk(
                    source_key="company/01_AVIVO_Company_Overview.md",
                    source_title="AVIVO — Company Overview & What They Do",
                    chunk_index=1,
                    content="company content",
                    score=0.8,
                    snippet="deliver continuous strategic value to organisations",
                ),
            ]

            formatted = service._format_sources(sources)

            self.assertIn("References:", formatted)
            self.assertIn("Mahidhara S — Complete Professional Profile (My_Profile/Mahidhara_S_Profile.md)", formatted)
            self.assertIn("AVIVO — Company Overview & What They Do (company/01_AVIVO_Company_Overview.md)", formatted)
            self.assertNotIn("fragmented markdown", formatted)
            self.assertNotIn("👤", formatted)

    def test_make_snippet_avoids_cutting_the_first_word_midway(self) -> None:
        content = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
        )

        snippet = make_snippet(content, "theta", max_chars=28)

        self.assertFalse(snippet.startswith("eta"))
        self.assertNotIn("\n", snippet)


class VisionProviderTests(unittest.TestCase):
    def test_ollama_vision_provider_parses_fenced_json_cleanly(self) -> None:
        class FakeChatProvider:
            def generate(self, system_prompt: str, user_prompt: str, history=None, image_bytes=None) -> str:
                return """```json
{
  "caption": "A promotional image showing a bowl of Asian noodles with toppings and menu pricing.",
  "tags": ["advertisement", "food", "noodles"]
}
```"""

        provider = OllamaVisionProvider(FakeChatProvider())

        response = provider.describe(b"fake-image")

        self.assertEqual(
            response.caption,
            "A promotional image showing a bowl of Asian noodles with toppings and menu pricing.",
        )
        self.assertEqual(response.tags, ["advertisement", "food", "noodles"])
        self.assertEqual(response.warnings, [])

    def test_fallback_tag_extraction_ignores_json_words(self) -> None:
        tags = extract_tags_from_text("```json caption this bowl of noodles and vegetables```")

        self.assertEqual(tags, ["bowl", "noodles", "vegetables"])


class TelegramParsingTests(unittest.TestCase):
    def test_extract_command_supports_bot_suffix(self) -> None:
        command, args = extract_command(
            "/ask@projectragbot What is the policy?",
            [{"type": "bot_command", "offset": 0, "length": 18}],
        )

        self.assertEqual(command, "/ask")
        self.assertEqual(args, "What is the policy?")

    def test_placeholder_token_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = make_settings(root)
            settings.transport_mode = "telegram"
            settings.telegram_bot_token = "PASTE_YOUR_BOTFATHER_TOKEN_HERE"

            with self.assertRaisesRegex(ValueError, "real token from BotFather"):
                TelegramHttpClient(settings)

    def test_has_non_image_document_detects_pdf_but_not_images(self) -> None:
        self.assertTrue(has_non_image_document({"document": {"mime_type": "application/pdf"}}))
        self.assertFalse(has_non_image_document({"document": {"mime_type": "image/png"}}))
        self.assertFalse(has_non_image_document({"photo": [{"file_id": "abc"}]}))

    def test_scope_helpers_map_menu_text_and_prefixes(self) -> None:
        self.assertEqual(scope_from_menu_text("Company"), "company")
        self.assertEqual(scope_from_menu_text("Profile"), "profile")
        self.assertIsNone(scope_from_menu_text("Unknown"))
        self.assertIn("company/", source_prefixes_for_scope("company"))
        self.assertIn("my_profile/", source_prefixes_for_scope("profile"))


if __name__ == "__main__":
    unittest.main()

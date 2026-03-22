from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from .config import Settings
from .knowledge import KnowledgeBase
from .models import AskResponse, ImageResponse, ProviderStatus, RetrievedChunk, SummaryResponse
from .prompts import RAG_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT
from .providers import (
    KeywordEmbedder,
    ProviderResponseError,
    ProviderUnavailableError,
    build_chat_provider,
    build_embedder,
    build_vision_provider,
)
from .storage import Storage


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recent_history_as_messages(turns: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.user_text})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    return messages


class BotService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = Storage(settings.db_path)
        self.embedder = build_embedder(settings)
        self.chat_provider = build_chat_provider(settings)
        self.vision_provider = build_vision_provider(settings)
        self.knowledge = KnowledgeBase(settings, self.storage, self.embedder)
        self.runtime_warnings: list[str] = []
        self._manifest: dict[str, Any] = {}

    def prepare(self) -> None:
        try:
            self._manifest = self.knowledge.ensure_index()
        except ProviderUnavailableError as exc:
            if self.settings.embedding_provider.lower() != "keyword":
                self.runtime_warnings.append(
                    f"Embedding provider '{self.settings.embedding_provider}' is unavailable; "
                    "falling back to keyword retrieval."
                )
                self.embedder = KeywordEmbedder()
                self.knowledge = KnowledgeBase(self.settings, self.storage, self.embedder)
                self._manifest = self.knowledge.ensure_index()
            else:
                raise exc

    def help_text(self) -> str:
        lines = [
            "Hybrid RAG + vision bot",
            "",
            "Tap Company or Profile to choose a focus area, or use the commands below.",
            "",
            "Commands:",
            "/ask <question> - ask a question about the local knowledge base",
            "/image - send this command, then upload a photo for captioning",
            "/help - show usage",
        ]
        if self.settings.enable_summarize:
            lines.append("/summarize - summarize your latest bot interaction")
        lines.extend(
            [
                "",
                "Examples:",
                "/ask What is the leave carry-forward policy?",
                "/ask Which purchases need manager approval?",
                "/image",
                "",
                (
                    "Features: "
                    f"vision={'on' if self.settings.enable_vision else 'off'}, "
                    f"summarize={'on' if self.settings.enable_summarize else 'off'}, "
                    f"history={'on' if self.settings.enable_history else 'off'}, "
                    f"cache={'on' if self.settings.enable_cache else 'off'}"
                ),
                (
                    "Active modes: "
                    f"embedder={self.embedder.name}, "
                    f"llm={self.chat_provider.name}, "
                    f"vision={self.vision_provider.name}"
                ),
            ]
        )
        if self.runtime_warnings:
            lines.extend(["", "Runtime notes:"])
            lines.extend(f"- {warning}" for warning in self.runtime_warnings)
        return "\n".join(lines)

    def session_key(self, user_id: str, chat_id: str) -> str:
        return f"{chat_id}:{user_id}"

    def _session_state(self, user_id: str, chat_id: str) -> dict[str, Any]:
        return self.storage.get_session_state(self.session_key(user_id, chat_id))

    def _update_session_state(self, user_id: str, chat_id: str, **updates: Any) -> None:
        state = self._session_state(user_id, chat_id)
        state.update(updates)
        self.storage.set_session_state(self.session_key(user_id, chat_id), state)

    def mark_waiting_for_image(self, user_id: str, chat_id: str) -> None:
        self._update_session_state(user_id, chat_id, awaiting_image=True)

    def is_waiting_for_image(self, user_id: str, chat_id: str) -> bool:
        return bool(self._session_state(user_id, chat_id).get("awaiting_image"))

    def clear_waiting_for_image(self, user_id: str, chat_id: str) -> None:
        self._update_session_state(user_id, chat_id, awaiting_image=False)

    def set_query_scope(self, user_id: str, chat_id: str, scope: str | None) -> None:
        normalized = scope.strip().lower() if scope else None
        self._update_session_state(user_id, chat_id, query_scope=normalized)

    def get_query_scope(self, user_id: str, chat_id: str) -> str | None:
        value = self._session_state(user_id, chat_id).get("query_scope")
        if not value:
            return None
        return str(value).strip().lower()

    def ask(
        self,
        user_id: str,
        chat_id: str,
        query: str,
        source_prefixes: list[str] | None = None,
    ) -> AskResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            return AskResponse(
                answer="Usage: /ask <question>",
                sources=[],
                provider="none",
            )

        if not self._manifest:
            self.prepare()

        cache_payload = {
            "query": cleaned_query,
            "top_k": self.settings.top_k,
            "manifest": self._manifest.get("hash"),
            "provider": self.chat_provider.name,
            "source_prefixes": source_prefixes or [],
        }
        cache_key = _hash_payload(cache_payload)

        if self.settings.enable_cache:
            cached = self.storage.get_query_cache(cache_key)
            if cached is not None:
                sources = [
                    RetrievedChunk(**source_payload)
                    for source_payload in cached.get("sources", [])
                ]
                response = AskResponse(
                    answer=cached["answer"],
                    sources=sources,
                    provider=cached.get("provider", self.chat_provider.name),
                    from_cache=True,
                )
                self.storage.add_turn(
                    user_id=user_id,
                    chat_id=chat_id,
                    mode="rag",
                    user_text=cleaned_query,
                    assistant_text=response.answer,
                    artifact={
                        "sources": [asdict(source) for source in response.sources],
                        "provider": response.provider,
                        "from_cache": True,
                    },
                )
                return response

        sources = self.knowledge.search(
            cleaned_query,
            self.settings.top_k,
            source_prefixes=source_prefixes,
        )
        answer_text, provider_name = self._render_rag_answer(
            user_id=user_id,
            chat_id=chat_id,
            query=cleaned_query,
            sources=sources,
        )

        response = AskResponse(answer=answer_text, sources=sources, provider=provider_name)
        self.storage.add_turn(
            user_id=user_id,
            chat_id=chat_id,
            mode="rag",
            user_text=cleaned_query,
            assistant_text=response.answer,
            artifact={
                "sources": [asdict(source) for source in response.sources],
                "provider": response.provider,
                "from_cache": False,
            },
        )

        if self.settings.enable_cache:
            self.storage.put_query_cache(
                cache_key=cache_key,
                query_hash=_hash_payload({"query": cleaned_query}),
                query_text=cleaned_query,
                top_k=self.settings.top_k,
                model_name=provider_name,
                result={
                    "answer": response.answer,
                    "sources": [asdict(source) for source in response.sources],
                    "provider": provider_name,
                },
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
        return response

    def _render_rag_answer(
        self,
        user_id: str,
        chat_id: str,
        query: str,
        sources: list[RetrievedChunk],
    ) -> tuple[str, str]:
        history = (
            self.storage.get_recent_turns(
                user_id=user_id,
                chat_id=chat_id,
                limit=self.settings.max_history_turns,
            )
            if self.settings.enable_history
            else []
        )
        if not sources:
            return (
                "I couldn't find a reliable answer in the provided documents. "
                "Try a narrower question or add more source documents.",
                "extractive",
            )

        context_blocks: list[str] = []
        current_size = 0
        for index, source in enumerate(sources, start=1):
            block = (
                f"[{index}] {source.source_title} ({source.source_key})\n"
                f"{source.content.strip()}"
            )
            if current_size + len(block) > self.settings.max_context_chars and context_blocks:
                break
            context_blocks.append(block)
            current_size += len(block)

        prompt = (
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{'\n\n'.join(context_blocks)}\n\n"
            "Answer the question using only the context above."
        )
        try:
            generated = self.chat_provider.generate(
                system_prompt=RAG_SYSTEM_PROMPT,
                user_prompt=prompt,
                history=_recent_history_as_messages(history),
            ).strip()
            if generated:
                answer = self._sanitize_answer(generated)
                provider_name = self.chat_provider.name
                if not answer:
                    answer = self._fallback_answer(query, sources)
                    provider_name = "extractive"
            else:
                answer = self._fallback_answer(query, sources)
                provider_name = "extractive"
        except (ProviderUnavailableError, ProviderResponseError):
            answer = self._fallback_answer(query, sources)
            provider_name = "extractive"

        source_block = self._format_sources(sources)
        mode_block = ""
        if provider_name == "extractive":
            mode_block = "\n\nMode: retrieval-only fallback (no live LLM response was used)."
        return f"{answer}{source_block}{mode_block}", provider_name

    def _fallback_answer(self, query: str, sources: list[RetrievedChunk]) -> str:
        primary = sources[0]
        if len(sources) == 1:
            return f"Most relevant finding: {self._clean_fragment(primary.snippet)}"
        secondary = sources[1]
        return "\n".join(
            [
                "Most relevant findings:",
                f"- {self._clean_fragment(primary.snippet)}",
                f"- {self._clean_fragment(secondary.snippet)}",
            ]
        )

    def _format_sources(self, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return ""
        lines = ["", "", "References:"]
        seen: set[str] = set()
        for source in sources:
            if source.source_key in seen:
                continue
            seen.add(source.source_key)
            title = self._clean_source_title(source.source_title)
            lines.append(f"- {title} ({source.source_key})")
            if len(seen) == 2:
                break
        return "\n".join(lines)

    def _sanitize_answer(self, text: str) -> str:
        lines: list[str] = []
        for raw_line in text.splitlines():
            normalized = raw_line.strip().lower()
            if normalized.startswith("sources:") or normalized.startswith("references:"):
                break
            lines.append(raw_line.rstrip())
        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _clean_fragment(self, text: str) -> str:
        cleaned = re.sub(r"[`*_>#|]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        return cleaned

    def _clean_source_title(self, title: str) -> str:
        cleaned = re.sub(r"^[^\w]+", "", title).strip()
        return cleaned or title

    def describe_image(
        self,
        user_id: str,
        chat_id: str,
        image_bytes: bytes,
        mime_type: str | None = None,
    ) -> ImageResponse:
        self.clear_waiting_for_image(user_id, chat_id)
        if not self.settings.enable_vision:
            return ImageResponse(
                caption="Image support is disabled in the current configuration.",
                tags=["disabled", "vision", "off"],
                provider="none",
            )
        try:
            response = self.vision_provider.describe(image_bytes, mime_type)
        except (ProviderUnavailableError, ProviderResponseError):
            response = ImageResponse(
                caption=(
                    "Vision backend is unavailable right now. "
                    "Configure Ollama or the optional BLIP provider to enable image captioning."
                ),
                tags=["vision", "unavailable", "setup"],
                provider="none",
            )
        except Exception:
            response = ImageResponse(
                caption="I couldn't process that image. Try another file or switch the vision backend.",
                tags=["image", "error", "retry"],
                provider="none",
            )
        self.storage.add_turn(
            user_id=user_id,
            chat_id=chat_id,
            mode="vision",
            user_text="image upload",
            assistant_text=self.format_image_response(response),
            artifact={
                "caption": response.caption,
                "tags": response.tags,
                "provider": response.provider,
                "warnings": response.warnings,
            },
        )
        return response

    def format_image_response(self, response: ImageResponse) -> str:
        lines = [
            f"Caption: {response.caption}",
            f"Tags: {', '.join(response.tags)}",
        ]
        if response.warnings:
            lines.append(f"Note: {' '.join(response.warnings)}")
        return "\n".join(lines)

    def summarize(self, user_id: str, chat_id: str) -> SummaryResponse:
        if not self.settings.enable_summarize:
            return SummaryResponse("Summaries are disabled.", "none")
        turns = self.storage.get_recent_turns(
            user_id=user_id,
            chat_id=chat_id,
            limit=self.settings.max_history_turns,
        )
        if not turns:
            return SummaryResponse("No interactions to summarize yet.", "none")

        prompt_lines = []
        for turn in turns:
            prompt_lines.append(f"User ({turn.mode}): {turn.user_text}")
            prompt_lines.append(f"Assistant: {turn.assistant_text}")
        prompt = "\n".join(prompt_lines)
        try:
            summary = self.chat_provider.generate(
                system_prompt=SUMMARIZE_SYSTEM_PROMPT,
                user_prompt=prompt,
            ).strip()
            if summary:
                result = SummaryResponse(summary=summary, provider=self.chat_provider.name)
            else:
                result = SummaryResponse(
                    summary=self._fallback_summary(turns),
                    provider="extractive",
                )
        except (ProviderUnavailableError, ProviderResponseError):
            result = SummaryResponse(
                summary=self._fallback_summary(turns),
                provider="extractive",
            )

        self.storage.add_turn(
            user_id=user_id,
            chat_id=chat_id,
            mode="summary",
            user_text="/summarize",
            assistant_text=result.summary,
            artifact={"provider": result.provider},
        )
        return result

    def _fallback_summary(self, turns: list[Any]) -> str:
        latest = turns[-1]
        lines = [
            f"Latest mode: {latest.mode}",
            f"Latest user input: {latest.user_text}",
            f"Latest bot result: {latest.assistant_text[:220]}",
        ]
        if latest.mode == "vision":
            tags = latest.artifact.get("tags", [])
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
        return "\n".join(lines)

    def health_report(self) -> dict[str, Any]:
        provider_statuses: list[ProviderStatus] = [
            self.embedder.status(),
            self.chat_provider.status(),
            self.vision_provider.status(),
        ]
        self.prepare()
        doc_state = self.storage.get_state("knowledge_manifest") or {}
        return {
            "transport_mode": self.settings.transport_mode,
            "knowledge_manifest": doc_state,
            "active_modes": {
                "embedder": self.embedder.name,
                "llm": self.chat_provider.name,
                "vision": self.vision_provider.name,
            },
            "providers": [asdict(status) for status in provider_statuses],
            "runtime_warnings": self.runtime_warnings,
        }

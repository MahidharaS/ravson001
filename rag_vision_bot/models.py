from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    source_key: str
    title: str
    source_type: str
    content: str
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    source_key: str
    source_title: str
    chunk_index: int
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    source_key: str
    source_title: str
    chunk_index: int
    content: str
    score: float
    snippet: str


@dataclass(slots=True)
class ConversationTurn:
    user_id: str
    chat_id: str
    mode: str
    user_text: str
    assistant_text: str
    artifact: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class AskResponse:
    answer: str
    sources: list[RetrievedChunk]
    provider: str
    from_cache: bool = False


@dataclass(slots=True)
class ImageResponse:
    caption: str
    tags: list[str]
    provider: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SummaryResponse:
    summary: str
    provider: str


@dataclass(slots=True)
class ProviderStatus:
    name: str
    available: bool
    details: str

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .models import ChunkRecord, DocumentRecord, RetrievedChunk
from .providers import Embedder
from .storage import Storage


def compute_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_source_key(source_key: str) -> str:
    return source_key.replace("\\", "/").strip().lower()


def load_source_documents(knowledge_dir: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=2)
        title = extract_title(path, content)
        relative_path = path.relative_to(knowledge_dir).as_posix()
        documents.append(
            DocumentRecord(
                source_key=relative_path,
                title=title,
                source_type=path.suffix.lower().lstrip("."),
                content=content,
                checksum=compute_checksum(content),
                metadata={"path": str(path), "relative_path": relative_path},
            )
        )
    return documents


def extract_title(path: Path, content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").title()


def _split_large_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for index in range(0, len(part), max_chars):
                chunks.append(part[index : index + max_chars].strip())
            continue
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip()
            continue
        if current:
            chunks.append(current)
        current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]
    expanded: list[str] = []
    for paragraph in paragraphs:
        expanded.extend(_split_large_paragraph(paragraph, target_chars))

    chunks: list[str] = []
    current = ""
    for paragraph in expanded:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if overlap_chars and chunks:
            overlap = current[-overlap_chars:].strip()
            current = f"{overlap}\n\n{paragraph}".strip()
            if len(current) > target_chars:
                chunks.append(current[:target_chars].strip())
                current = current[target_chars - overlap_chars :].strip()
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return [chunk[:target_chars].strip() for chunk in chunks if chunk.strip()]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = sum(left[i] * right[i] for i in range(size))
    left_magnitude = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_magnitude = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return numerator / (left_magnitude * right_magnitude)


def make_snippet(content: str, query: str, max_chars: int = 200) -> str:
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
    lowered = content.lower()
    best_index = 0
    for term in terms:
        index = lowered.find(term)
        if index >= 0:
            best_index = index
            break
    start = max(best_index - max_chars // 3, 0)
    if start > 0:
        while start < len(content) and not content[start - 1].isspace():
            start += 1
    end = min(start + max_chars, len(content))
    if end < len(content):
        while end > start and not content[end - 1].isspace():
            end -= 1
    snippet = re.sub(r"\s+", " ", content[start:end]).strip()
    return snippet if end == len(content) else f"{snippet}..."


class KnowledgeBase:
    def __init__(self, settings: Settings, storage: Storage, embedder: Embedder) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder

    @property
    def manifest_key(self) -> str:
        return "knowledge_manifest"

    def _build_manifest(self, documents: list[DocumentRecord]) -> dict[str, Any]:
        manifest_input = {
            "documents": {doc.source_key: doc.checksum for doc in documents},
            "embedding_provider": self.embedder.name,
            "embedding_model": getattr(self.embedder, "model_name", self.embedder.name),
            "chunk_target_chars": self.settings.chunk_target_chars,
            "chunk_overlap_chars": self.settings.chunk_overlap_chars,
        }
        encoded = json.dumps(manifest_input, sort_keys=True)
        return {"hash": compute_checksum(encoded), **manifest_input}

    def ensure_index(self) -> dict[str, Any]:
        documents = load_source_documents(self.settings.knowledge_dir)
        manifest = self._build_manifest(documents)
        existing = self.storage.get_state(self.manifest_key)
        if existing and existing.get("hash") == manifest["hash"]:
            return manifest
        chunk_map: dict[str, list[ChunkRecord]] = {}
        for document in documents:
            contents = chunk_text(
                document.content,
                target_chars=self.settings.chunk_target_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            embeddings = self.embedder.embed_texts(contents) if contents else []
            chunk_map[document.source_key] = [
                ChunkRecord(
                    source_key=document.source_key,
                    source_title=document.title,
                    chunk_index=index,
                    content=chunk_content,
                    embedding=embedding,
                    metadata={
                        "content_hash": compute_checksum(chunk_content),
                        "source_type": document.source_type,
                        "path": document.metadata["path"],
                    },
                )
                for index, (chunk_content, embedding) in enumerate(zip(contents, embeddings, strict=False))
            ]
        self.storage.replace_knowledge(documents, chunk_map)
        self.storage.set_state(self.manifest_key, manifest)
        return manifest

    def search(
        self,
        query: str,
        top_k: int,
        source_prefixes: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.ensure_index()
        query_vector = self.embedder.embed_query(query)
        rows = self.storage.load_chunks()
        normalized_prefixes = [
            prefix.rstrip("/").lower() + "/"
            for prefix in (source_prefixes or [])
            if prefix.strip()
        ]
        ranked = sorted(
            (
                (
                    cosine_similarity(query_vector, row["embedding"]),
                    row,
                )
                for row in rows
                if not normalized_prefixes
                or any(
                    normalize_source_key(row["source_key"]).startswith(prefix)
                    for prefix in normalized_prefixes
                )
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        per_source: dict[str, int] = {}
        for score, row in ranked:
            source_hits = per_source.get(row["source_key"], 0)
            if source_hits >= 2:
                continue
            results.append(
                RetrievedChunk(
                    source_key=row["source_key"],
                    source_title=row["source_title"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    score=score,
                    snippet=make_snippet(row["content"], query),
                )
            )
            per_source[row["source_key"]] = source_hits + 1
            if len(results) == top_k:
                break
        return results

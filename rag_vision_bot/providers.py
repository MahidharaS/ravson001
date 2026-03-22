from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib import error, request

from .config import Settings
from .models import ImageResponse, ProviderStatus
from .prompts import IMAGE_CAPTION_PROMPT


class ProviderUnavailableError(RuntimeError):
    """Raised when a configured provider cannot be reached or is not installed."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider responds with an invalid payload."""


def _json_request(
    url: str,
    payload: dict[str, Any] | None,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:  # pragma: no cover - network environment varies
        raise ProviderUnavailableError(str(exc)) from exc

    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"Invalid JSON response from {url}") from exc


def _normalize(values: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / magnitude for value in values]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "json",
    "caption",
    "captions",
    "tag",
    "tags",
    "code",
    "block",
    "output",
    "return",
    "strict",
}


def extract_tags_from_text(text: str, limit: int = 3) -> list[str]:
    seen: list[str] = []
    for token in _tokenize(text):
        if token in STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) == limit:
            break
    while len(seen) < limit:
        seen.append(f"tag-{len(seen) + 1}")
    return seen


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    for candidate in [text.strip(), _strip_code_fence(text)]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def _clean_caption_text(text: str) -> str:
    cleaned = _strip_code_fence(text)
    cleaned = re.sub(r"^\s*caption\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip('"')
    return cleaned


def _normalize_tags(tags: Any, caption: str, limit: int = 3) -> list[str]:
    if isinstance(tags, str):
        raw_tags = [part.strip() for part in tags.split(",")]
    elif isinstance(tags, list):
        raw_tags = [str(tag).strip() for tag in tags]
    else:
        raw_tags = []

    normalized: list[str] = []
    for tag in raw_tags:
        candidate = re.sub(r"\s+", " ", tag.lower()).strip(" '\"`")
        if not candidate or candidate in STOPWORDS or candidate in normalized:
            continue
        normalized.append(candidate)
        if len(normalized) == limit:
            return normalized

    for derived in extract_tags_from_text(caption, limit=10):
        if derived in normalized:
            continue
        normalized.append(derived)
        if len(normalized) == limit:
            return normalized
    return normalized[:limit]


class Embedder:
    name = "embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "available")


class KeywordEmbedder(Embedder):
    name = "keyword"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % self.dimensions
                vector[bucket] += 1.0
            vectors.append(_normalize(vector))
        return vectors


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts}
        response = _json_request(f"{self.base_url}/api/embed", payload, self.timeout)
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings
        embedding = response.get("embedding")
        if isinstance(embedding, list):
            return [embedding]
        raise ProviderResponseError("Ollama embed response did not include embeddings")

    def status(self) -> ProviderStatus:
        try:
            _json_request(f"{self.base_url}/api/tags", None, min(self.timeout, 5))
        except ProviderUnavailableError as exc:
            return ProviderStatus(self.name, False, str(exc))
        return ProviderStatus(self.name, True, f"endpoint reachable, model={self.model_name}")


class SentenceTransformersEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def status(self) -> ProviderStatus:
        try:
            self._load_model()
        except Exception as exc:  # pragma: no cover - optional dependency
            return ProviderStatus(self.name, False, str(exc))
        return ProviderStatus(self.name, True, f"model={self.model_name}")


class ChatProvider:
    name = "chat"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        image_bytes: bytes | None = None,
    ) -> str:
        raise NotImplementedError

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "available")


class NullChatProvider(ChatProvider):
    name = "none"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        image_bytes: bytes | None = None,
    ) -> str:
        raise ProviderUnavailableError("No chat provider is configured")

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, False, "disabled")


class OllamaChatProvider(ChatProvider):
    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        image_bytes: bytes | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image_bytes:
            user_message["images"] = [base64.b64encode(image_bytes).decode("ascii")]
        messages.append(user_message)

        payload = {"model": self.model_name, "messages": messages, "stream": False}
        response = _json_request(f"{self.base_url}/api/chat", payload, self.timeout)
        message = response.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"]).strip()
        if response.get("response"):
            return str(response["response"]).strip()
        raise ProviderResponseError("Ollama chat response did not include message content")

    def status(self) -> ProviderStatus:
        try:
            _json_request(f"{self.base_url}/api/tags", None, min(self.timeout, 5))
        except ProviderUnavailableError as exc:
            return ProviderStatus(self.name, False, str(exc))
        return ProviderStatus(self.name, True, f"endpoint reachable, model={self.model_name}")


class VisionProvider:
    name = "vision"

    def describe(self, image_bytes: bytes, mime_type: str | None = None) -> ImageResponse:
        raise NotImplementedError

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "available")


class NullVisionProvider(VisionProvider):
    name = "none"

    def describe(self, image_bytes: bytes, mime_type: str | None = None) -> ImageResponse:
        raise ProviderUnavailableError("No vision provider is configured")

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, False, "disabled")


class OllamaVisionProvider(VisionProvider):
    name = "ollama"

    def __init__(self, chat_provider: OllamaChatProvider) -> None:
        self.chat_provider = chat_provider

    def describe(self, image_bytes: bytes, mime_type: str | None = None) -> ImageResponse:
        prompt = (
            f"{IMAGE_CAPTION_PROMPT}\n\n"
            "Return strict JSON with keys 'caption' and 'tags'. "
            "The 'tags' value must be an array of exactly 3 lowercase tags."
        )
        raw = self.chat_provider.generate(
            system_prompt="You are a precise image captioning assistant.",
            user_prompt=prompt,
            image_bytes=image_bytes,
        )
        parsed = _extract_json_blob(raw)
        if parsed:
            caption = _clean_caption_text(str(parsed.get("caption", "")))
            tags = _normalize_tags(parsed.get("tags") or [], caption)
            return ImageResponse(
                caption=caption or "The image was processed, but the caption response was empty.",
                tags=tags[:3],
                provider=self.name,
            )

        caption = _clean_caption_text(raw)
        return ImageResponse(
            caption=caption,
            tags=extract_tags_from_text(caption),
            provider=self.name,
            warnings=["Vision provider returned non-JSON output; tags were derived from the caption."],
        )

    def status(self) -> ProviderStatus:
        return self.chat_provider.status()


class TransformersBlipVisionProvider(VisionProvider):
    name = "transformers_blip"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from PIL import Image  # type: ignore # noqa: F401
            from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        return self._processor, self._model

    def describe(self, image_bytes: bytes, mime_type: str | None = None) -> ImageResponse:
        from PIL import Image  # type: ignore

        processor, model = self._load()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        return ImageResponse(
            caption=caption,
            tags=extract_tags_from_text(caption),
            provider=self.name,
        )

    def status(self) -> ProviderStatus:
        try:
            self._load()
        except Exception as exc:  # pragma: no cover - optional dependency
            return ProviderStatus(self.name, False, str(exc))
        return ProviderStatus(self.name, True, f"model={self.model_name}")


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model,
            timeout=settings.request_timeout_seconds,
        )
    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(settings.sentence_transformer_model)
    return KeywordEmbedder()


def build_chat_provider(settings: Settings) -> ChatProvider:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaChatProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.llm_model,
            timeout=settings.request_timeout_seconds,
        )
    return NullChatProvider()


def build_vision_provider(settings: Settings) -> VisionProvider:
    provider = settings.vision_provider.lower()
    if provider == "ollama":
        chat_provider = OllamaChatProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.vision_model,
            timeout=settings.request_timeout_seconds,
        )
        return OllamaVisionProvider(chat_provider)
    if provider == "transformers_blip":
        return TransformersBlipVisionProvider(settings.transformers_vision_model)
    return NullVisionProvider()

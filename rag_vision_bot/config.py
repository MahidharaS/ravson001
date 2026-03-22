from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value.strip())


@dataclass(slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    db_path: Path
    knowledge_dir: Path
    transport_mode: str
    telegram_bot_token: str | None
    telegram_api_base: str
    telegram_poll_timeout: int
    max_image_bytes: int
    auto_caption_photos: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    vision_provider: str
    vision_model: str
    sentence_transformer_model: str
    transformers_vision_model: str
    ollama_base_url: str
    request_timeout_seconds: int
    top_k: int
    chunk_target_chars: int
    chunk_overlap_chars: int
    max_context_chars: int
    max_history_turns: int
    cache_ttl_seconds: int
    enable_history: bool
    enable_cache: bool
    enable_vision: bool
    enable_summarize: bool

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path.cwd()
    _load_dotenv(root / ".env")
    data_dir = Path(os.getenv("BOT_DATA_DIR", root / "data"))
    knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", root / "knowledge_base"))
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    transport_mode = os.getenv("BOT_TRANSPORT")
    if not transport_mode:
        transport_mode = "telegram" if telegram_bot_token else "cli"

    settings = Settings(
        project_root=root,
        data_dir=data_dir,
        db_path=Path(os.getenv("BOT_DB_PATH", data_dir / "bot.db")),
        knowledge_dir=knowledge_dir,
        transport_mode=transport_mode,
        telegram_bot_token=telegram_bot_token,
        telegram_api_base=os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org"),
        telegram_poll_timeout=_as_int(os.getenv("TELEGRAM_POLL_TIMEOUT"), 30),
        max_image_bytes=_as_int(os.getenv("MAX_IMAGE_BYTES"), 20 * 1024 * 1024),
        auto_caption_photos=_as_bool(os.getenv("AUTO_CAPTION_PHOTOS"), True),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "keyword"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        vision_provider=os.getenv("VISION_PROVIDER", "ollama"),
        vision_model=os.getenv("VISION_MODEL", "llava:7b"),
        sentence_transformer_model=os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        transformers_vision_model=os.getenv(
            "TRANSFORMERS_VISION_MODEL", "Salesforce/blip-image-captioning-base"
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout_seconds=_as_int(os.getenv("REQUEST_TIMEOUT_SECONDS"), 60),
        top_k=_as_int(os.getenv("TOP_K"), 4),
        chunk_target_chars=_as_int(os.getenv("CHUNK_TARGET_CHARS"), 700),
        chunk_overlap_chars=_as_int(os.getenv("CHUNK_OVERLAP_CHARS"), 120),
        max_context_chars=_as_int(os.getenv("MAX_CONTEXT_CHARS"), 2800),
        max_history_turns=_as_int(os.getenv("MAX_HISTORY_TURNS"), 3),
        cache_ttl_seconds=_as_int(os.getenv("CACHE_TTL_SECONDS"), 7 * 24 * 60 * 60),
        enable_history=_as_bool(os.getenv("ENABLE_HISTORY"), True),
        enable_cache=_as_bool(os.getenv("ENABLE_CACHE"), True),
        enable_vision=_as_bool(os.getenv("ENABLE_VISION"), True),
        enable_summarize=_as_bool(os.getenv("ENABLE_SUMMARIZE"), True),
    )
    settings.ensure_directories()
    return settings

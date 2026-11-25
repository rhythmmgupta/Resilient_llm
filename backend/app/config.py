from dataclasses import dataclass
import os
from typing import List
from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass(frozen=True)
class KeyEntry:
    """Represents a configured API key slot in the failover pool."""

    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    key_pool: List[KeyEntry]
    database_url: str
    allowed_origins: List[str]
    enable_failure_simulation: bool
    cache_type: str
    cache_dir: str
    default_system_prompt: str
    default_model: str
    request_timeout_seconds: float = 30.0


def sqlite_path_from_url(database_url: str) -> str:
    """Convert a SQLite DATABASE_URL into a filesystem path for sqlite3."""
    if database_url == "sqlite:///:memory:":
        return ":memory:"
    if database_url.startswith("sqlite:////"):
        return "/" + database_url[len("sqlite:////"):]
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///"):]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://"):]
    return database_url


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_settings() -> Settings:
    """Read configuration from environment variables."""
    raw_keys = _parse_csv(os.getenv("LLM_API_KEYS", ""))
    default_model = (
        os.getenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite").strip()
        or "gemini/gemini-3.5-flash-lite"
    )
    per_key_models = _parse_csv(os.getenv("LLM_MODELS", ""))

    key_pool: List[KeyEntry] = []
    for idx, api_key in enumerate(raw_keys, start=1):
        model = per_key_models[idx - 1] if idx - 1 < len(per_key_models) else default_model
        key_pool.append(KeyEntry(key_id=f"key_{idx}", api_key=api_key, model=model))

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip()
    allowed_origins = _parse_csv(
        os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )
    enable_failure_simulation = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower() == "true"
    cache_type = os.getenv("CACHE_TYPE", "local").strip().lower()
    if cache_type not in {"local", "disk"}:
        cache_type = "local"
    cache_dir = os.getenv("CACHE_DIR", "/data/litellm_cache").strip() or "/data/litellm_cache"
    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    ).strip()

    try:
        request_timeout_seconds = max(1.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "30")))
    except ValueError:
        request_timeout_seconds = 30.0

    return Settings(
        key_pool=key_pool,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        cache_type=cache_type,
        cache_dir=cache_dir,
        default_system_prompt=default_system_prompt,
        default_model=default_model,
        request_timeout_seconds=request_timeout_seconds,
    )

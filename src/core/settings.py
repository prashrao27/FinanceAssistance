"""Typed application settings loaded from config.yaml and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    llm_model: str
    router_model: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    mcp_server_url: str
    log_level: str
    data_directory: Path
    conversation_db_path: Path
    conversation_history_turns: int
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    openai_api_key: str | None
    alpha_vantage_api_key: str | None
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str | None
    dense_embedding_model: str
    sparse_embedding_model: str
    search_cache_ttl_seconds: int
    search_cache_max_entries: int
    search_max_results: int
    finance_api_timeout_seconds: int
    finance_api_max_attempts: int
    finance_api_max_retry_delay_seconds: int


def _read_config() -> dict[str, Any]:
    try:
        with CONFIG_FILE.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise RuntimeError(f"Could not read configuration file {CONFIG_FILE}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in configuration file {CONFIG_FILE}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping at its root")
    return config


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml section {name!r} must be a mapping")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once; environment values override config.yaml defaults."""
    config = _read_config()
    app = _section(config, "app")
    llm = _section(config, "llm")
    router = _section(config, "router")
    mcp = _section(config, "mcp")
    logging_config = _section(config, "logging")
    langsmith = _section(config, "langsmith")
    storage = _section(config, "storage")
    data = _section(config, "data")
    qdrant = _section(config, "qdrant")
    rag = _section(config, "rag")
    search = _section(config, "search")
    finance_api = _section(config, "finance_api")

    def env_string(name: str, default: Any = None) -> str | None:
        value = os.environ.get(name)
        return value.strip() if value is not None else default

    def env_integer(name: str, default: Any) -> int:
        value = os.environ.get(name)
        try:
            return int(value) if value is not None else int(default)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc

    def env_boolean(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean")

    langsmith_tracing_default = langsmith.get("tracing", False)
    if not isinstance(langsmith_tracing_default, bool):
        raise ValueError("config.yaml langsmith.tracing must be a boolean")

    data_directory = Path(
        env_string("DATA_DIRECTORY", data.get("directory", "src/data"))
    )
    if not data_directory.is_absolute():
        data_directory = PROJECT_ROOT / data_directory
    conversation_db_path = Path(
        env_string(
            "CONVERSATION_DB_PATH",
            storage.get("sqlite_path", "data/finance_assistant.sqlite3"),
        )
    )
    if not conversation_db_path.is_absolute():
        conversation_db_path = PROJECT_ROOT / conversation_db_path

    return Settings(
        app_name=str(app.get("name", "Finance Assistant")),
        llm_model=str(env_string("OPENAI_MODEL", llm.get("model", "gpt-4o-mini"))),
        router_model=str(
            env_string(
                "OPENROUTER_MODEL",
                router.get("model", "~typesafe/jev-latest"),
            )
        ),
        openrouter_api_key=env_string("OPENROUTER_API_KEY"),
        openrouter_base_url=str(
            env_string(
                "OPENROUTER_BASE_URL",
                router.get("base_url", "https://openrouter.ai"),
            )
        ),
        mcp_server_url=str(
            env_string(
                "MCP_SERVER_URL",
                mcp.get("server_url", "http://127.0.0.1:8000/mcp"),
            )
        ),
        log_level=str(env_string("LOG_LEVEL", logging_config.get("level", "INFO"))),
        data_directory=data_directory,
        conversation_db_path=conversation_db_path,
        conversation_history_turns=env_integer(
            "CONVERSATION_HISTORY_TURNS", storage.get("history_turns", 12)
        ),
        qdrant_url=env_string("QDRANT_URL"),
        qdrant_api_key=env_string("QDRANT_API_KEY"),
        qdrant_collection=str(
            env_string("QDRANT_COLLECTION", qdrant.get("collection", "FinAi"))
        ),
        openai_api_key=env_string("OPENAI_API_KEY"),
        alpha_vantage_api_key=env_string("ALPHA_VANTAGE_API_KEY"),
        langsmith_tracing=env_boolean("LANGSMITH_TRACING", langsmith_tracing_default),
        langsmith_api_key=env_string("LANGSMITH_API_KEY"),
        langsmith_project=str(
            env_string(
                "LANGSMITH_PROJECT",
                langsmith.get("project", "finance-assistant"),
            )
        ),
        langsmith_endpoint=env_string(
            "LANGSMITH_ENDPOINT", langsmith.get("endpoint")
        ),
        dense_embedding_model=str(
            env_string(
                "DENSE_EMBEDDING_MODEL",
                rag.get("dense_embedding_model", "BAAI/bge-large-en-v1.5"),
            )
        ),
        sparse_embedding_model=str(
            env_string(
                "SPARSE_EMBEDDING_MODEL",
                rag.get("sparse_embedding_model", "prithvida/Splade_PP_en_v1"),
            )
        ),
        search_cache_ttl_seconds=env_integer(
            "SEARCH_CACHE_TTL_SECONDS", search.get("cache_ttl_seconds", 60)
        ),
        search_cache_max_entries=env_integer(
            "SEARCH_CACHE_MAX_ENTRIES", search.get("cache_max_entries", 256)
        ),
        search_max_results=env_integer(
            "SEARCH_MAX_RESULTS", search.get("max_results", 50)
        ),
        finance_api_timeout_seconds=int(
            env_integer(
                "FINANCE_API_TIMEOUT_SECONDS",
                finance_api.get("request_timeout_seconds", 15),
            )
        ),
        finance_api_max_attempts=env_integer(
            "FINANCE_API_MAX_ATTEMPTS", finance_api.get("max_attempts", 3)
        ),
        finance_api_max_retry_delay_seconds=env_integer(
            "FINANCE_API_MAX_RETRY_DELAY_SECONDS",
            finance_api.get("max_retry_delay_seconds", 8),
        ),
    )

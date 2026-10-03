"""Deployment settings (environment, prefix GH_). Changing them requires a restart."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GH_", env_file=".env", extra="ignore")

    db_path: str = "graphhop.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    log_level: str = "INFO"
    admin_api_key: str | None = None  # protects ingestion and deletion; unset disables them

    # ingestion
    chunk_chars: int = 800
    chunk_overlap_chars: int = 100
    title_as_entity: bool = True
    extractor: str = "proper_noun"  # "proper_noun" | "llm"

    # retrieval defaults (each can be overridden per request)
    default_k: int = 5
    max_k: int = 50
    graph_strategy: str = "expand"  # "expand" (graph proposes, dense ranks) | "ppr"
    expand_hops: int = 1
    title_only: bool = True
    ppr_alpha: float = 0.85
    max_hops: int = 3
    seed_from_dense_top_n: int = 1
    dense_seed_weight: float = 0.5
    w_dense: float = 1.0
    w_graph: float = 1.0
    max_context_chars: int = 8000

    # generation (optional: retrieval works without it)
    llm_provider: str | None = None  # "anthropic" | "openai_compat"
    llm_model_id: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None  # name of the env variable holding the key
    llm_max_tokens: int = 500
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    min_answer_chars: int = 12  # letters required besides citation markers
    abstain_token: str = "INSUFFICIENT_CONTEXT"  # noqa: S105 (sentinel string, not a secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""
Centralized, typed configuration for the whole platform.

Everything that could plausibly change between a laptop dev run, CI, and
the production GPU host is read from environment variables (or a .env
file) here -- nothing is hardcoded in business logic.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendMode(str, Enum):
    MOCK = "mock"
    VLLM = "vllm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Backend ----
    llm_backend_mode: BackendMode = BackendMode.MOCK
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    vllm_api_key: str = "not-needed"

    # ---- KV cache / LMCache ----
    kv_cache_budget_gb: float = 8.0
    lmcache_cpu_offload_gb: float = 40.0
    lmcache_disk_path: str = "/mnt/lmcache/store"
    lmcache_chunk_size: int = 256

    # ---- Concurrency / sessions ----
    max_concurrent_sessions: int = 100
    session_ttl_seconds: int = 1800
    max_history_turns: int = 8

    # ---- Retrieval ----
    chroma_persist_dir: str = "./data/chroma"
    kb_top_k: int = 4

    # ---- Quality gate ----
    quality_score_threshold: float = 0.75
    max_quality_retries: int = 2

    # ---- Logging ----
    log_level: str = "INFO"

    # ---- Model geometry (used only for the KV-cache sizing calculator,
    # defaults match Llama-3-8B; override for a different model) ----
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    kv_dtype_bytes: int = 2  # fp16/bf16


@lru_cache
def get_settings() -> Settings:
    return Settings()

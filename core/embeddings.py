"""Embedding provider for semantic search.

Supports multiple backends via OpenAI-compatible /v1/embeddings API:
  - OpenAI (text-embedding-3-small)
  - Local: lm-studio, llama.cpp, mlx-lm, Ollama, vLLM
  - Fallback: hash-based (no server needed)

Config in config.yaml:
  embedding:
    provider: local          # local | openai | off
    base_url: http://localhost:1234/v1   # lm-studio default
    model: nomic-embed-text-v1.5         # or any GGUF/MLX embedding model
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import httpx

from core.config import EmbeddingConfig

logger = logging.getLogger(__name__)

_config = EmbeddingConfig()
_client: httpx.AsyncClient | None = None


def configure(cfg: EmbeddingConfig) -> None:
    """Set embedding configuration. Called at startup from AppConfig."""
    global _config, _client
    _config = EmbeddingConfig(
        provider=cfg.provider, base_url=cfg.base_url.rstrip("/"), model=cfg.model, api_key=cfg.api_key
    )
    # Reset client so it picks up new config
    _client = None
    if cfg.provider != "off":
        logger.info("Embedding: provider=%s, base_url=%s, model=%s", cfg.provider, cfg.base_url, cfg.model or "(auto)")


async def embed_text(text: str, provider: Any = None) -> list[float]:
    """Generate embedding vector for text.

    Priority: configured provider > passed provider > local fallback.
    """
    if _config.provider == "local":
        try:
            return await _embed_local_server(text)
        except Exception:
            logger.debug("Local embedding server failed, using hash fallback", exc_info=True)

    if _config.provider == "openai" and provider:
        try:
            return await _embed_openai_sdk(text, provider)
        except Exception:
            logger.debug("OpenAI embedding failed, using hash fallback", exc_info=True)

    return _local_embed(text)


async def _embed_local_server(text: str) -> list[float]:
    """Call any OpenAI-compatible /v1/embeddings endpoint.

    Works with: lm-studio, llama.cpp (--embedding), mlx-lm, Ollama, vLLM, LocalAI.
    """
    url = f"{_config.base_url}/embeddings"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _config.api_key:
        headers["Authorization"] = f"Bearer {_config.api_key}"

    body: dict[str, Any] = {"input": text[:8000]}
    if _config.model:
        body["model"] = _config.model

    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30)
    resp = await _client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    embedding = data["data"][0]["embedding"]
    logger.debug("Local embedding: dim=%d, model=%s", len(embedding), data.get("model", "?"))
    return embedding


async def _embed_openai_sdk(text: str, provider: Any) -> list[float]:
    """Use OpenAI SDK's embedding API."""
    if hasattr(provider, "_client") and hasattr(provider._client, "embeddings"):
        model = _config.model or "text-embedding-3-small"
        resp = await provider._client.embeddings.create(model=model, input=text[:8000])
        return resp.data[0].embedding
    raise NotImplementedError("Provider does not support embeddings")


def _local_embed(text: str, dim: int = 256) -> list[float]:
    """Hash-based embedding — no server needed.

    Not semantically meaningful, but provides consistent vectors for
    exact/near-exact match. Used as fallback when no embedding server is running.
    """
    words = text.lower().split()
    vec = [0.0] * dim

    for i, word in enumerate(words):
        h = hash(word) % dim
        weight = 1.0 / (1.0 + math.log1p(i))
        vec[h] += weight

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Handle different dimensions (e.g. switching between providers)
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(min_len))
    norm_a = math.sqrt(sum(x * x for x in a[:min_len])) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b[:min_len])) or 1.0
    return dot / (norm_a * norm_b)


def serialize_embedding(vec: list[float]) -> str:
    return json.dumps([round(v, 6) for v in vec])


def deserialize_embedding(data: str) -> list[float]:
    return json.loads(data)

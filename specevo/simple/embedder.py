"""Description embedder.

Embeds LLM-emitted ``## Description`` blocks via litellm. Default config uses
OpenAI's ``text-embedding-3-small`` reached through OpenRouter, but any
litellm-compatible model works. Results are cached by SHA1 of the input text
so repeated descriptions cost nothing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import litellm
import numpy as np

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    model: str = "openrouter/openai/text-embedding-3-small"
    dim: int = 1536
    api_key_env: str = "OPENAI_API_KEY"
    cache_size: int = 4096
    request_timeout: float = 30.0


@dataclass
class DescriptionEmbedder:
    config: EmbedderConfig = field(default_factory=EmbedderConfig)
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _api_key(self) -> str | None:
        return os.getenv(self.config.api_key_env)

    def _evict_if_full(self) -> None:
        if len(self._cache) > self.config.cache_size:
            # Drop oldest insertion order (dict preserves insertion order).
            for k in list(self._cache.keys())[: len(self._cache) - self.config.cache_size]:
                self._cache.pop(k, None)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single description. Returns L2-normalized vector."""
        text = (text or "").strip()
        if not text:
            return np.zeros(self.config.dim, dtype=np.float32)
        cache_key = self._key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        resp = litellm.embedding(
            model=self.config.model,
            input=[text],
            api_key=self._api_key(),
            timeout=self.config.request_timeout,
        )
        vec = np.asarray(resp.data[0]["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        self._cache[cache_key] = vec
        self._evict_if_full()
        return vec

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed many descriptions; uses cache for hits, batches the rest."""
        out: list[np.ndarray | None] = [None] * len(texts)
        to_call: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = np.zeros(self.config.dim, dtype=np.float32)
                continue
            k = self._key(t)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                to_call.append((i, t))

        if to_call:
            resp = litellm.embedding(
                model=self.config.model,
                input=[t for _, t in to_call],
                api_key=self._api_key(),
                timeout=self.config.request_timeout,
            )
            # Sort by `index` defensively: OpenAI-compatible endpoints return
            # entries in request order, but the field is the spec'd guarantee.
            entries = sorted(resp.data, key=lambda d: int(d.get("index", 0)))
            for (i, t), entry in zip(to_call, entries, strict=True):
                vec = np.asarray(entry["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                self._cache[self._key(t)] = vec
                out[i] = vec
            self._evict_if_full()

        return [v if v is not None else np.zeros(self.config.dim, dtype=np.float32) for v in out]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already L2-normalized vectors (dot product)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))

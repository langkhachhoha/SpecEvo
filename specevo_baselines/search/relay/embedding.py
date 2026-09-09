"""Candidate embeddings for the relay bank's diversity-coverage term.

The paper represents each candidate by a *code* embedding and a *textual
metadata* embedding, and blends the two cosine similarities with a weight
``eta`` (Eq. 3).  Two backends provide those vectors:

``hash`` (default)
    A local, deterministic, free bag-of-tokens hashing embedding.  No API
    call, no extra latency, no spend charged against the run's dollar
    budget — which matters because the relay bank is re-scored after every
    block.

``api``
    Any OpenAI-compatible ``/embeddings`` endpoint (e.g. OpenAI's
    ``text-embedding-3-small``).  Falls back to ``hash`` — permanently, and
    with one warning — the first time the endpoint errors, so a bad
    embedding endpoint can never abort a run.

Both return L2-normalised ``float32`` vectors, so cosine similarity is a
dot product.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*|[^\sA-Za-z_0-9]")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


class CandidateEmbedder:
    """Embeds candidate code and metadata text, caching by content."""

    def __init__(
        self,
        backend: str = "hash",
        dim: int = 512,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_chars: int = 20000,
    ):
        self.backend = backend if backend in ("hash", "api") else "hash"
        self.dim = int(dim)
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.max_chars = max_chars
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._client = None
        self._api_broken = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Embed many strings, reusing cached vectors."""
        cleaned = [(t or "")[: self.max_chars] for t in texts]
        out: List[Optional[np.ndarray]] = [None] * len(cleaned)
        missing: List[int] = []

        with self._lock:
            for i, t in enumerate(cleaned):
                hit = self._cache.get(t)
                if hit is not None:
                    out[i] = hit
                else:
                    missing.append(i)

        if missing:
            wanted = [cleaned[i] for i in missing]
            vectors = self._embed_uncached(wanted)
            with self._lock:
                for i, vec in zip(missing, vectors):
                    self._cache[cleaned[i]] = vec
                    out[i] = vec

        return [v if v is not None else np.zeros(self.dim, dtype=np.float32) for v in out]

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _embed_uncached(self, texts: List[str]) -> List[np.ndarray]:
        if self.backend == "api" and not self._api_broken:
            try:
                return self._embed_api(texts)
            except Exception as exc:  # noqa: BLE001 — never let this kill a run
                self._api_broken = True
                logger.warning(
                    "Embedding API failed (%s); falling back to the local hashing "
                    "embedder for the rest of this run.",
                    exc,
                )
        return [self._embed_hash(t) for t in texts]

    def _embed_hash(self, text: str) -> np.ndarray:
        """Hashed bag-of-tokens with sublinear term weighting."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec
        counts: Dict[int, float] = {}
        for tok in tokens:
            idx = hash(_stable(tok)) % self.dim
            counts[idx] = counts.get(idx, 0.0) + 1.0
        for idx, count in counts.items():
            vec[idx] = 1.0 + np.log(count)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _embed_api(self, texts: List[str]) -> List[np.ndarray]:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key or "sk-none",
                base_url=self.api_base,
                timeout=self.timeout,
            )
        resp = self._client.embeddings.create(model=self.model, input=texts)
        entries = sorted(resp.data, key=lambda d: int(getattr(d, "index", 0)))
        vectors = []
        for entry in entries:
            vec = np.asarray(entry.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            vectors.append(vec / norm if norm > 0 else vec)
        return vectors


def _stable(token: str) -> int:
    """Deterministic token hash (Python's ``hash(str)`` is salted per process)."""
    h = 2166136261
    for ch in token:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-normalised vectors."""
    if a is None or b is None or a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.dot(a, b))

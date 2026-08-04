"""Production embedding providers — Sentence Transformers, OpenAI, BGE, E5, Nomic, Ollama.

HashingEmbeddingProvider remains the offline fallback. Planner semantic retrieval
uses whichever provider ``build_embedding_provider`` selects from settings.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import Any, Literal, Protocol

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)

EmbeddingProviderName = Literal[
    "hashing",
    "sentence_transformers",
    "openai",
    "bge",
    "e5",
    "nomic",
    "ollama",
]


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


class HashingEmbeddingProvider:
    """Deterministic local embeddings (offline fallback)."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "hashing"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        import re

        token_re = re.compile(r"[a-z0-9_]{3,}", re.I)
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = token_re.findall((text or "").lower()) or ["empty"]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class SentenceTransformerEmbeddingProvider:
    """sentence-transformers / BGE / E5 / Nomic via local models."""

    MODEL_ALIASES: dict[str, str] = {
        "bge": "BAAI/bge-small-en-v1.5",
        "e5": "intfloat/e5-small-v2",
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
        "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = self.MODEL_ALIASES.get(model_name, model_name)
        self._model: Any = None
        self._dim = 384
        self._load()

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        kwargs: dict[str, Any] = {}
        if "nomic" in self.model_name.lower():
            kwargs["trust_remote_code"] = True
        self._model = SentenceTransformer(self.model_name, **kwargs)
        try:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            probe = self._model.encode(["dim"], normalize_embeddings=True)
            self._dim = int(len(probe[0]))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"sentence_transformers:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        prepared = []
        for t in texts:
            raw = t or ""
            # E5 models expect query:/passage: prefixes for best quality
            if "e5" in self.model_name.lower() and not raw.lower().startswith(
                ("query:", "passage:")
            ):
                raw = f"query: {raw}"
            prepared.append(raw)
        vectors = self._model.encode(prepared, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-* API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI embedding provider requires api_key")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors = [list(map(float, item["embedding"])) for item in items]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


class OllamaEmbeddingProvider:
    """Ollama /api/embeddings (nomic-embed-text, mxbai-embed-large, etc.)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dim: int = 768,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text or ""}).encode(
                "utf-8"
            )
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vec = list(map(float, data.get("embedding") or []))
            if not vec:
                raise RuntimeError(f"Ollama returned empty embedding for model={self.model}")
            self._dim = len(vec)
            out.append(vec)
        return out


def build_embedding_provider(
    provider: EmbeddingProviderName | str = "hashing",
    *,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    dim: int = 256,
    openai_base_url: str = "https://api.openai.com/v1",
) -> EmbeddingProvider:
    """Factory — falls back to hashing if the requested provider cannot load."""
    name = (provider or "hashing").strip().lower()
    try:
        if name in {"hashing", "hash", "local"}:
            return HashingEmbeddingProvider(dim=dim)

        if name == "openai":
            return OpenAIEmbeddingProvider(
                api_key=api_key,
                model=model or "text-embedding-3-small",
                base_url=openai_base_url or "https://api.openai.com/v1",
            )

        if name == "ollama":
            return OllamaEmbeddingProvider(
                base_url=base_url or "http://localhost:11434",
                model=model or "nomic-embed-text",
            )

        if name in {"sentence_transformers", "bge", "e5", "nomic", "minilm"}:
            model_name = model or SentenceTransformerEmbeddingProvider.MODEL_ALIASES.get(
                name, "sentence-transformers/all-MiniLM-L6-v2"
            )
            return SentenceTransformerEmbeddingProvider(model_name)

        logger.warning("Unknown embedding provider %s — using hashing", name)
        return HashingEmbeddingProvider(dim=dim)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Embedding provider %s unavailable (%s) — falling back to hashing",
            name,
            exc,
        )
        return HashingEmbeddingProvider(dim=dim)

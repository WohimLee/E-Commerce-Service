# embedder.py
from __future__ import annotations

import os
import math
import time
import json
import hashlib

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Any

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

class Embedder:
    """
    Unified embedder interface.

    embed(texts) -> list of vectors (list[float])
    """

    def load_model(self, *args, **kwargs):
        pass

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


# ----------------------------
# 1) Dummy embedder (deterministic)
# ----------------------------

@dataclass
class DummyEmbedder(Embedder):
    """
    Deterministic pseudo-embedding for pipeline wiring.

    - No external dependency
    - Same input text -> same vector
    - Produces unit-length-ish vectors (cosine-friendly)
    """
    dims: int = 1024

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        # 1) Hash the text to bytes
        h = hashlib.sha256(text.encode("utf-8")).digest()

        # 2) Expand bytes into dims floats deterministically.
        #    We'll generate a stream of hashes: sha256(h + counter)
        out: List[float] = []
        counter = 0
        while len(out) < self.dims:
            counter_bytes = counter.to_bytes(4, "big", signed=False)
            block = hashlib.sha256(h + counter_bytes).digest()  # 32 bytes
            # map each byte to [-1, 1]
            for b in block:
                out.append((b / 255.0) * 2.0 - 1.0)
                if len(out) >= self.dims:
                    break
            counter += 1

        # 3) L2 normalize (avoid all-zero)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


# ----------------------------
# 2) HTTP embedder (internal service)
# ----------------------------

@dataclass
class HttpEmbedder(Embedder):
    """
    Calls an HTTP embedding service.

    Expected request:
      POST {base_url}/embed
      {"texts": [...], "dims": 1024? (optional), "model": "..."? (optional)}

    Expected response (one of):
      {"embeddings": [[...], ...]}
      or {"data": [{"embedding": [...]}, ...]}  (OpenAI-like)

    You can adapt parse logic as needed.
    """
    base_url: str
    timeout: int = 30
    model: Optional[str] = None
    dims: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    max_batch_size: int = 64
    retry: int = 2
    backoff: float = 0.5

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = self.base_url.rstrip("/") + "/embed"
        payload: Dict[str, Any] = {"texts": texts}
        if self.model:
            payload["model"] = self.model
        if self.dims is not None:
            payload["dims"] = self.dims

        last_exc: Optional[Exception] = None
        for attempt in range(self.retry + 1):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                obj = resp.json()
                return self._parse_embeddings(obj, expected=len(texts))
            except Exception as e:
                last_exc = e
                if attempt >= self.retry:
                    break
                time.sleep(self.backoff * (2 ** attempt))

        raise RuntimeError(f"HTTP embedding failed: {last_exc}") from last_exc

    @staticmethod
    def _parse_embeddings(obj: Dict[str, Any], expected: int) -> List[List[float]]:
        if isinstance(obj, dict):
            if "embeddings" in obj:
                embs = obj["embeddings"]
            elif "data" in obj and isinstance(obj["data"], list) and obj["data"] and "embedding" in obj["data"][0]:
                embs = [d["embedding"] for d in obj["data"]]
            else:
                raise ValueError(f"Unrecognized embedding response format: keys={list(obj.keys())}")
        else:
            raise ValueError("Embedding response must be a JSON object.")

        if not isinstance(embs, list) or (embs and not isinstance(embs[0], list)):
            raise ValueError("Embeddings must be a list of list[float].")
        if len(embs) != expected:
            raise ValueError(f"Embedding count mismatch: got {len(embs)} expected {expected}")
        return embs


# ----------------------------
# 3) OpenAI embedder (stub / optional)
# ----------------------------

@dataclass
class OpenAIEmbedder(Embedder):
    """
    Optional: OpenAI embeddings.
    We keep this as a stub so your pipeline doesn't depend on openai package yet.

    To enable:
      - install official SDK
      - implement embed() accordingly
    """
    api_key: str
    model: str = "text-embedding-3-large"
    dims: Optional[int] = None  # if your model supports dimension truncation
    max_batch_size: int = 128

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "OpenAIEmbedder is a stub. Implement with OpenAI SDK when you decide to use it."
        )


# ----------------------------
# 4) Local embedder (sentence-transformers)
# ----------------------------

@dataclass
class LocalEmbedder(Embedder):
    """
    Local embedding via sentence-transformers.

    Env suggestions:
      EMBEDDER_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2
      EMBEDDER_LOCAL_DEVICE=cpu|cuda|mps (optional)
      EMBEDDER_LOCAL_BATCH=64
      EMBEDDER_LOCAL_NORMALIZE=true|false
      EMBEDDER_LOCAL_CACHE_DIR=/path/to/cache (optional)

    Notes:
      - If `dims` is set and differs from model output dimension, we will
        truncate or pad with zeros to match `dims`.
      - By default we return L2-normalized vectors if normalize=True.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 64
    normalize: bool = True
    dims: Optional[int] = None
    cache_dir: Optional[str] = None

    # internal lazy model
    _model: Any = None

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            raise RuntimeError(
                "LocalEmbedder requires `sentence-transformers`.\n"
                "Install with: pip install sentence-transformers"
            ) from e

        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir

        # self._model = SentenceTransformer(self.model_name, **kwargs)
        self._model = BGEM3FlagModel(self.model_name, use_fp16=True) 

        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self.load_model()

        # sentence-transformers returns numpy.ndarray when convert_to_numpy=True
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        # Ensure python lists
        vectors: List[List[float]] = [v.astype(float).tolist() for v in vecs]

        # Optionally match dims by truncation/padding
        if self.dims is not None:
            vectors = [self._match_dims(v, self.dims) for v in vectors]

        return vectors

    @staticmethod
    def _match_dims(v: List[float], dims: int) -> List[float]:
        if len(v) == dims:
            return v
        if len(v) > dims:
            return v[:dims]
        # pad with zeros
        return v + [0.0] * (dims - len(v))

# ----------------------------
# Factory
# ----------------------------

def get_embedder() -> Embedder:
    """
    Create embedder based on env vars.

    EMBEDDER_TYPE:
      - dummy (default)
      - http
      - local

    For http:
      EMBEDDER_HTTP_URL=http://localhost:8000
      EMBEDDER_HTTP_MODEL=...
      EMBEDDER_DIMS=1024
      EMBEDDER_HTTP_HEADERS='{"Authorization":"Bearer xxx"}' (optional)
    """
    t = (os.getenv("EMBEDDER_TYPE") or "dummy").strip().lower()
    dims = int(os.getenv("EMBEDDER_DIMS", "1024"))

    if t == "dummy":
        return DummyEmbedder(dims=dims)

    if t == "http":
        base_url = os.getenv("EMBEDDER_HTTP_URL")
        if not base_url:
            raise ValueError("EMBEDDER_HTTP_URL is required when EMBEDDER_TYPE=http")

        model = os.getenv("EMBEDDER_HTTP_MODEL")
        headers_json = os.getenv("EMBEDDER_HTTP_HEADERS")
        headers = json.loads(headers_json) if headers_json else None

        max_batch_size = int(os.getenv("EMBEDDER_HTTP_BATCH", "64"))
        timeout = int(os.getenv("EMBEDDER_HTTP_TIMEOUT", "30"))

        return HttpEmbedder(
            base_url=base_url,
            model=model,
            dims=dims,
            headers=headers,
            max_batch_size=max_batch_size,
            timeout=timeout,
        )
    
    if t == "local":
        model_name = os.getenv("EMBEDDER_LOCAL_MODEL")
        device = os.getenv("EMBEDDER_LOCAL_DEVICE", "cpu")  # e.g. "cpu", "cuda", "mps"
        batch_size = int(os.getenv("EMBEDDER_LOCAL_BATCH", "64"))
        normalize_str = (os.getenv("EMBEDDER_LOCAL_NORMALIZE") or "true").strip().lower()
        normalize = normalize_str in ("1", "true", "yes", "y", "on")
        cache_dir = os.getenv("EMBEDDER_LOCAL_CACHE_DIR")

        return LocalEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            dims=dims,  # reuse EMBEDDER_DIMS
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unknown EMBEDDER_TYPE: {t}")


if __name__ == "__main__":

    embedder = get_embedder()
    embedder.load_model()

    pass
   


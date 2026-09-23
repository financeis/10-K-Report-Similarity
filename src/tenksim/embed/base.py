"""청크 임베딩 공통 인터페이스와 디스크 캐시."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.where(norms == 0, 1, norms)).astype(np.float32)


class EmbeddingCache:
    """(모델 식별자, 텍스트 해시) → 벡터.

    같은 텍스트는 다시 모델/API를 호출하지 않는다. 청크 방식이나 평균 방식을 바꿔
    다시 돌려도, 이미 임베딩한 청크는 비용 없이 재사용된다.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            " model TEXT NOT NULL, key TEXT NOT NULL, vec BLOB NOT NULL,"
            " PRIMARY KEY (model, key))"
        )

    def get_many(self, model: str, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, vec FROM vectors WHERE model = ? AND key IN ({placeholders})",
                [model, *batch],
            )
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, model: str, items: list[tuple[str, np.ndarray]]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (model, key, vec) VALUES (?, ?, ?)",
                [(model, k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DenseEmbedder(ABC):
    """청크를 받아 고정 길이 벡터를 내는 모델 (sentence-transformers, OpenAI)."""

    model_id: str
    """캐시 네임스페이스. 벡터 값을 바꾸는 설정(모델명, 차원 등)을 모두 담아야 한다."""
    max_tokens: int
    batch_size: int

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray: ...

    def _batches(self, items: list[tuple[str, str]]) -> Iterator[list[tuple[str, str]]]:
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]

    def embed(self, texts: list[str], cache: EmbeddingCache) -> np.ndarray:
        """L2 정규화된 float32 벡터를 texts 순서대로 돌려준다."""
        keys = [text_key(t) for t in texts]
        unique = dict(zip(keys, texts, strict=True))
        found = cache.get_many(self.model_id, list(unique))
        todo = [(k, t) for k, t in unique.items() if k not in found]
        log.info(
            "%s: %d unique chunks, %d cached, %d to embed",
            self.model_id,
            len(unique),
            len(found),
            len(todo),
        )
        if todo:
            with tqdm(total=len(todo), desc=self.model_id) as bar:
                for batch in self._batches(todo):
                    vecs = l2_normalize(self._embed([t for _, t in batch]))
                    items = [(k, v) for (k, _), v in zip(batch, vecs, strict=True)]
                    cache.put_many(self.model_id, items)
                    found.update(items)
                    bar.update(len(batch))
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

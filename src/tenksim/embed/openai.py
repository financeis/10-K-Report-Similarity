"""OpenAI 임베딩 API (유료, OPENAI_API_KEY 필요).

text-embedding-3는 앞쪽 차원만 잘라 써도 되도록 학습된 모델이다(Matryoshka).
그래서 dimensions를 바꾼 실험은 API가 잘라 준 벡터를 그대로 받는다.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np

from .base import DenseEmbedder


class OpenAIEmbedder(DenseEmbedder):
    MAX_INPUTS_PER_REQUEST = 2048
    # 요청당 입력 토큰 합계에도 한도가 있어 여유 있게 잡는다
    MAX_TOKENS_PER_REQUEST = 250_000

    def __init__(
        self,
        model: str,
        *,
        dimensions: int | None = None,
        batch_size: int = 256,
        max_tokens: int | None = None,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 필요합니다. .env에 넣거나, 설정에서 openai method를 빼세요."
            )
        import tiktoken
        from openai import OpenAI

        # 429/5xx 응답은 SDK가 지수 백오프로 재시도한다
        self._client = OpenAI(max_retries=6)
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(batch_size, self.MAX_INPUTS_PER_REQUEST)
        # 모델 입력 한도는 8191 토큰
        self.max_tokens = max_tokens or 8000
        self.model_id = f"openai:{model}:{dimensions or 'native'}"

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def _batches(self, items: list[tuple[str, str]]) -> Iterator[list[tuple[str, str]]]:
        batch: list[tuple[str, str]] = []
        tokens = 0
        for key, text in items:
            n = self.count_tokens(text)
            if batch and (
                len(batch) >= self.batch_size or tokens + n > self.MAX_TOKENS_PER_REQUEST
            ):
                yield batch
                batch, tokens = [], 0
            batch.append((key, text))
            tokens += n
        if batch:
            yield batch

    def _embed(self, texts: list[str]) -> np.ndarray:
        kwargs: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(**kwargs)
        data = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([d.embedding for d in data], dtype=np.float32)

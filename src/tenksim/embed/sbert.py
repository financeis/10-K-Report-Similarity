"""sentence-transformers 로컬 모델. 무료이고 결과 재현이 쉽다.

기본값 all-mpnet-base-v2는 참고 논문(Vamvourellis et al.)의 SBERT-PT와 같은 모델이다.
논문에서 GPT(ada-002)와 대등한 성능을 냈다.
"""

from __future__ import annotations

import numpy as np

from .base import DenseEmbedder


class SbertEmbedder(DenseEmbedder):
    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 32,
        max_tokens: int | None = None,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        # max_seq_length는 특수 토큰(<s>, </s>)을 포함한 길이다
        limit = self._model.max_seq_length - 2
        self.max_tokens = min(max_tokens, limit) if max_tokens else limit
        self.batch_size = batch_size
        self.model_id = f"sbert:{model}"
        self._tokenizer = self._model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

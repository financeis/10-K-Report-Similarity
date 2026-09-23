"""기업 벡터를 만드는 방법들: TF-IDF 기준선과 청크 임베딩 모델."""

from __future__ import annotations

from ..config import MethodConfig
from .base import DenseEmbedder, EmbeddingCache, l2_normalize, text_key
from .tfidf import TfidfMethod

__all__ = [
    "DenseEmbedder",
    "EmbeddingCache",
    "TfidfMethod",
    "build_method",
    "l2_normalize",
    "text_key",
]


def build_method(cfg: MethodConfig) -> TfidfMethod | DenseEmbedder:
    if cfg.kind == "tfidf":
        return TfidfMethod(min_df=cfg.min_df, max_df=cfg.max_df)
    if cfg.kind == "sbert":
        from .sbert import SbertEmbedder

        return SbertEmbedder(
            cfg.model, batch_size=cfg.batch_size, max_tokens=cfg.max_tokens, device=cfg.device
        )
    if cfg.kind == "openai":
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder(
            cfg.model,
            dimensions=cfg.dimensions,
            batch_size=cfg.batch_size,
            max_tokens=cfg.max_tokens,
        )
    raise ValueError(f"unknown method kind: {cfg.kind}")

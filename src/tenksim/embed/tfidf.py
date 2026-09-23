"""단어 빈도(TF-IDF) 기준선. Hoberg & Phillips(2016)의 방식을 단순화했다.

원 논문은 명사·고유명사만 쓰고, 전체 문서의 25% 넘게 등장하는 흔한 단어를 뺀다.
여기서는 품사 필터 없이 max_df로 흔한 단어만 뺀다. LLM 임베딩이 이 기준선을
못 넘으면 비싼 모델을 쓸 이유가 없다.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfMethod:
    def __init__(self, *, min_df: int = 2, max_df: float = 0.25):
        self.min_df = min_df
        self.max_df = max_df

    def fit_transform(self, docs: list[str]) -> sparse.csr_matrix:
        """문서마다 L2 정규화된 희소 벡터를 만든다 (행 순서 = docs 순서)."""
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",  # 숫자는 빼고 3글자 이상 단어만
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        try:
            return vectorizer.fit_transform(docs).tocsr()
        except ValueError as exc:
            raise ValueError(
                f"TF-IDF 어휘가 비었습니다 (문서 {len(docs)}개, min_df={self.min_df}, "
                f"max_df={self.max_df}). 문서 수가 너무 적으면 max_df를 올리세요."
            ) from exc

"""회사 벡터 풀링, 유사도 행렬, 이웃 찾기, 유사 근거(청크 쌍) 찾기."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from .embed.base import l2_normalize


def pool_chunks(
    chunk_vectors: np.ndarray, owners: np.ndarray, weights: np.ndarray, n_companies: int
) -> np.ndarray:
    """청크 벡터를 토큰 수로 가중 평균해 회사 벡터를 만든다 (L2 정규화)."""
    pooled = np.zeros((n_companies, chunk_vectors.shape[1]), dtype=np.float64)
    np.add.at(pooled, owners, chunk_vectors * weights[:, None])
    return l2_normalize(pooled)


def cosine_matrix(vectors: np.ndarray | sparse.spmatrix, *, center: bool = False) -> np.ndarray:
    """코사인 유사도 행렬.

    center=True면 전체 평균 벡터를 뺀 뒤 계산한다. 모든 10-K가 공유하는 공시 문체
    성분을 걷어내서, 점수가 0.8 근처에 몰리는 현상을 줄이려는 것이다.
    """
    if sparse.issparse(vectors):
        if center:
            raise ValueError("centering은 dense 벡터에만 적용합니다")
        x = sparse.csr_matrix(vectors)
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1))).ravel()
        x = sparse.diags(1 / np.where(norms == 0, 1, norms)) @ x
        return np.asarray((x @ x.T).todense(), dtype=np.float32)
    x = np.asarray(vectors, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    x = l2_normalize(x)
    return (x @ x.T).astype(np.float32)


def top_k(sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """행마다 자기 자신을 뺀 상위 k개 (인덱스, 점수)."""
    s = sim.astype(np.float64, copy=True)
    np.fill_diagonal(s, -np.inf)
    k = min(k, s.shape[0] - 1)
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(s, idx, axis=1)


def pair_percentiles(sim: np.ndarray) -> np.ndarray:
    """각 기업쌍 점수가 전체 기업쌍 중 몇 백분위인지 (0~100).

    코사인 값 자체는 모델마다 분포가 달라 해석하기 어렵다. 같은 유니버스 안에서의
    상대 위치로 바꾸면 '0~100 점수'로 읽을 수 있다. 대각선은 NaN.
    """
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    pct = np.full((n, n), np.nan, dtype=np.float32)
    if len(iu[0]):
        ranks = rankdata(sim[iu], method="average")
        values = (ranks - 1) / max(len(ranks) - 1, 1) * 100
        pct[iu] = values
        pct[(iu[1], iu[0])] = values
    return pct


def explain_pair(
    chunks_a: list[str],
    vectors_a: np.ndarray,
    chunks_b: list[str],
    vectors_b: np.ndarray,
    top: int = 3,
) -> list[tuple[float, str, str]]:
    """두 회사에서 가장 비슷한 청크 쌍. 왜 비슷하다고 나왔는지 근거 문단을 보여준다."""
    sim = vectors_a @ vectors_b.T
    flat = np.argsort(-sim, axis=None)
    out: list[tuple[float, str, str]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for pos in flat:
        i, j = np.unravel_index(pos, sim.shape)
        # 한 청크가 여러 쌍을 독차지하지 않도록 청크당 한 번만 쓴다
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((float(sim[i, j]), chunks_a[i], chunks_b[j]))
        if len(out) == top:
            break
    return out

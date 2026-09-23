import numpy as np
from scipy import sparse

from tenksim.similarity import (
    cosine_matrix,
    ensemble_similarity,
    explain_pair,
    pair_percentiles,
    pool_chunks,
    top_k,
)


def test_pool_chunks_is_token_weighted_mean():
    chunks = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    owners = np.array([0, 0, 1])
    weights = np.array([3.0, 1.0, 5.0])
    pooled = pool_chunks(chunks, owners, weights, 2)
    expected0 = np.array([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
    np.testing.assert_allclose(pooled[0], expected0, rtol=1e-6)
    np.testing.assert_allclose(pooled[1], [1.0, 0.0])


def test_cosine_matrix_dense_and_sparse_agree():
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(size=(6, 20)))
    dense = cosine_matrix(x)
    sp = cosine_matrix(sparse.csr_matrix(x))
    np.testing.assert_allclose(dense, sp, atol=1e-5)
    np.testing.assert_allclose(np.diag(dense), 1.0, atol=1e-5)


def test_centering_removes_shared_component():
    # 모든 문서가 같은 '공시 문체' 성분을 크게 공유하면 원래 코사인은 전부 1에 가깝다
    rng = np.random.default_rng(1)
    shared = rng.normal(size=64) * 10
    x = shared + rng.normal(size=(30, 64))
    off = ~np.eye(30, dtype=bool)
    assert cosine_matrix(x)[off].mean() > 0.9
    assert abs(cosine_matrix(x, center=True)[off].mean()) < 0.1


def test_top_k_excludes_self_and_sorts():
    sim = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.5], [0.9, 0.5, 1.0]])
    idx, scores = top_k(sim, 2)
    assert idx.tolist() == [[2, 1], [2, 0], [0, 1]]
    np.testing.assert_allclose(scores[0], [0.9, 0.2])
    assert top_k(sim, 10)[0].shape == (3, 2)  # k가 회사 수보다 크면 잘린다


def test_pair_percentiles():
    sim = np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.5], [0.9, 0.5, 1.0]])
    pct = pair_percentiles(sim)
    assert np.isnan(np.diag(pct)).all()
    np.testing.assert_allclose(pct, pct.T, equal_nan=True)
    assert pct[0, 1] == 0 and pct[0, 2] == 100 and pct[1, 2] == 50


def test_explain_pair_uses_each_chunk_once():
    a = np.eye(3, dtype=np.float32)
    b = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    pairs = explain_pair(["a0", "a1", "a2"], a, ["b0", "b1", "b2"], b, top=2)
    assert [(ta, tb) for _, ta, tb in pairs] == [("a0", "b0"), ("a2", "b2")]
    assert pairs[0][0] >= pairs[1][0]


def test_ensemble_similarity_averages_ranks():
    a = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.5], [0.1, 0.5, 1.0]])
    b = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.2, 0.1, 1.0]])  # 척도가 전혀 다르다
    e = ensemble_similarity([a, b])
    np.testing.assert_allclose(np.diag(e), 100)
    np.testing.assert_allclose(e, e.T)
    # a 순위: (0,1)=100 (1,2)=50 (0,2)=0 / b 순위: (0,2)=100 (1,2)=50 (0,1)=0
    assert e[0, 1] == 50 and e[1, 2] == 50 and e[0, 2] == 50
    heavy = ensemble_similarity([a, b], weights=[3, 1])
    assert heavy[0, 1] > heavy[0, 2]

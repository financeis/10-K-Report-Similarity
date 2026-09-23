import numpy as np
import pandas as pd
import pytest

from tenksim.evaluate import (
    agreement,
    bootstrap_means,
    label_frame,
    label_metrics,
    pair_auc,
    precision_at_k,
    precision_per_firm,
)


def block_similarity(labels: list[str], noise: float = 0.0, seed: int = 0) -> np.ndarray:
    lab = np.array(labels)
    sim = (lab[:, None] == lab[None, :]).astype(float)
    rng = np.random.default_rng(seed)
    n = rng.normal(scale=noise, size=sim.shape)
    return sim + (n + n.T) / 2


def test_precision_at_k_perfect_blocks():
    labels = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    sim = block_similarity(list(labels), noise=0.01)
    r = precision_at_k(sim, labels, k=2)
    assert r["precision"] == 1.0
    assert r["random"] == pytest.approx(2 / 5)  # 다른 5개 중 같은 산업 2개
    assert r["n"] == 6


def test_precision_skips_missing_labels():
    labels = np.array(["a", "a", None, "b", "b", np.nan], dtype=object)
    sim = block_similarity(["a", "a", "x", "b", "b", "y"], noise=0.01)
    assert precision_at_k(sim, labels, k=1)["n"] == 4
    assert precision_at_k(sim, labels, k=1)["precision"] == 1.0


def test_pair_auc():
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    assert pair_auc(block_similarity(list(labels), noise=0.01), labels) == 1.0
    assert np.isnan(pair_auc(np.eye(3), np.array(["a", "a", "a"], dtype=object)))
    rng = np.random.default_rng(0)
    big = np.array(list("abcd") * 50, dtype=object)
    noise = rng.normal(size=(200, 200))
    assert abs(pair_auc(noise + noise.T, big) - 0.5) < 0.05


def test_label_metrics_shape():
    labels = pd.DataFrame({"sector": ["a", "a", "b", "b"], "sic2": ["10", "20", "30", "30"]})
    sim = block_similarity(["a", "a", "b", "b"], noise=0.01)
    m = label_metrics(sim, labels, [1, 2])
    assert set(m) == {"sector", "sic2"}
    assert set(m["sector"]) == {"auc", "p@1", "random@1", "p@2", "random@2", "n"}


def test_label_frame_uses_sic_from_filings():
    universe = pd.DataFrame(
        {
            "cik": [1, 2],
            "ticker": ["A", "B"],
            "gics_sector": ["IT", "Energy"],
            "gics_sub_industry": ["Semis", "Oil"],
        }
    )
    records = pd.DataFrame({"cik": [1, 1, 2], "sic": ["3674", "3674", "911"]})
    labels = label_frame(universe, records)
    assert labels.loc[1, "sic3"] == "367" and labels.loc[2, "sic4"] == "0911"
    assert labels.loc[2, "sic2"] == "09"


def test_agreement_identity():
    sim = block_similarity(list("aabbcc"), noise=0.1)
    a = agreement(sim, sim, k=2)
    assert a["spearman"] == 1.0 and a["jaccard@2"] == 1.0


def test_precision_per_firm_marks_missing_labels():
    labels = np.array(["a", "a", None, "b", "b"], dtype=object)
    sim = block_similarity(["a", "a", "x", "b", "b"], noise=0.01)
    per = precision_per_firm(sim, labels, k=1)
    assert np.isnan(per[2]) and np.nanmean(per) == 1.0


def test_bootstrap_paired_difference_is_tighter():
    rng = np.random.default_rng(0)
    difficulty = rng.normal(0, 1, 400)  # 회사마다 다른 난이도가 두 방법에 똑같이 들어간다
    a = difficulty + 0.10 + rng.normal(0, 0.1, 400)
    b = difficulty + rng.normal(0, 0.1, 400)
    out = bootstrap_means({"b": b, "a": a}, n_boot=500, reference="b")
    assert out["a"]["lo"] <= out["a"]["mean"] <= out["a"]["hi"]
    assert "diff" not in out["b"]
    assert 0.05 < out["a"]["diff_lo"] < out["a"]["diff"] < out["a"]["diff_hi"] < 0.15
    # 짝지은 차이의 구간이 각 평균의 구간보다 훨씬 좁다
    assert out["a"]["diff_hi"] - out["a"]["diff_lo"] < (out["a"]["hi"] - out["a"]["lo"]) / 3


def test_bootstrap_ignores_nan():
    out = bootstrap_means({"x": np.array([1.0, np.nan, 3.0, np.nan])}, n_boot=200)
    assert out["x"]["mean"] == 2.0

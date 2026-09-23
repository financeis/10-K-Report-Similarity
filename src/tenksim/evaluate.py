"""산업분류를 정답으로 쓴 평가 지표, 부트스트랩 신뢰구간, 방법 간 일치도.

- precision@k: 텍스트로 찾은 상위 k개 이웃 중 같은 산업(label)인 비율.
  random은 무작위로 k개를 뽑았을 때의 기댓값이다.
- pair AUC: 임의의 '같은 산업 쌍'이 임의의 '다른 산업 쌍'보다 점수가 높을 확률.
  0.5면 무작위, 1이면 완벽하다. k를 정하지 않아도 되는 전체 순위 지표다.

산업분류 재현은 '텍스트가 사업 내용을 담고 있다'는 최소 조건을 확인하는 용도다.
분류를 완벽히 재현하는 유사도는 분류표 이상의 정보가 없다는 뜻이기도 하다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .similarity import top_k


def label_frame(universe: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """cik별 정답 레이블. GICS는 universe에서, SIC는 EDGAR 회사 정보에서 가져온다."""
    sic = (
        records.dropna(subset=["sic"])
        .drop_duplicates("cik")
        .set_index("cik")["sic"]
        .astype(str)
        .str.zfill(4)
    )
    labels = universe.set_index("cik")[["gics_sector", "gics_sub_industry"]].copy()
    labels["sic4"] = sic
    labels["sic3"] = sic.str[:3]
    labels["sic2"] = sic.str[:2]
    return labels


def _valid(labels: np.ndarray) -> np.ndarray:
    return np.array([isinstance(v, str) and v != "" for v in labels])


def precision_per_firm(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """회사별로 상위 k 이웃 중 같은 레이블인 비율. 레이블이 없는 회사는 NaN."""
    out = np.full(len(labels), np.nan)
    idx = np.flatnonzero(_valid(labels))
    if len(idx) < 2:
        return out
    lab = labels[idx]
    neighbors, _ = top_k(sim[np.ix_(idx, idx)], k)
    out[idx] = (lab[neighbors] == lab[:, None]).mean(axis=1)
    return out


def precision_at_k(sim: np.ndarray, labels: np.ndarray, k: int) -> dict:
    idx = np.flatnonzero(_valid(labels))
    if len(idx) < 2:
        return {"precision": np.nan, "random": np.nan, "n": int(len(idx))}
    lab = labels[idx]
    same = (lab[:, None] == lab[None, :]).sum(axis=1) - 1
    return {
        "precision": float(np.nanmean(precision_per_firm(sim, labels, k))),
        "random": float((same / (len(idx) - 1)).mean()),
        "n": int(len(idx)),
    }


def bootstrap_means(
    values: dict[str, np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    reference: str | None = None,
) -> dict[str, dict]:
    """회사 단위로 재표집한 평균의 95% 구간.

    values의 배열은 모두 같은 회사 순서여야 한다. 같은 재표집 표본을 모든 방법에 쓰므로
    reference와의 차이도 짝지어(paired) 비교된다. 회사마다 난이도가 달라서, 짝지은 차이의
    구간이 각 평균의 구간보다 훨씬 좁다. NaN(해당 없는 회사)은 평균에서 뺀다.
    """
    n = len(next(iter(values.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    ref = values.get(reference) if reference else None
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 전부 NaN인 재표집 표본
        for name, v in values.items():
            v = np.asarray(v, dtype=np.float64)
            lo, hi = np.nanpercentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])
            entry = {"mean": float(np.nanmean(v)), "lo": float(lo), "hi": float(hi)}
            if ref is not None and name != reference:
                d = v - ref
                dlo, dhi = np.nanpercentile(np.nanmean(d[idx], axis=1), [2.5, 97.5])
                entry.update(diff=float(np.nanmean(d)), diff_lo=float(dlo), diff_hi=float(dhi))
            out[name] = entry
    return out


def pair_auc(sim: np.ndarray, labels: np.ndarray) -> float:
    idx = np.flatnonzero(_valid(labels))
    iu = np.triu_indices(len(idx), 1)
    lab = labels[idx]
    y = lab[iu[0]] == lab[iu[1]]
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, sim[np.ix_(idx, idx)][iu]))


def label_metrics(sim: np.ndarray, labels: pd.DataFrame, ks: list[int]) -> dict:
    """labels: 행 순서가 sim과 같은 DataFrame (컬럼 = label 이름)."""
    out = {}
    for name in labels.columns:
        values = labels[name].to_numpy(dtype=object)
        entry = {"auc": pair_auc(sim, values)}
        for k in ks:
            r = precision_at_k(sim, values, k)
            entry[f"p@{k}"] = r["precision"]
            entry[f"random@{k}"] = r["random"]
        entry["n"] = int(_valid(values).sum())
        out[name] = entry
    return out


def agreement(sim_a: np.ndarray, sim_b: np.ndarray, k: int) -> dict:
    """두 방법이 얼마나 같은 답을 내는지: 전체 쌍 순위 상관과 상위 k 이웃 겹침(Jaccard)."""
    iu = np.triu_indices(sim_a.shape[0], 1)
    rho = spearmanr(sim_a[iu], sim_b[iu]).statistic
    na, _ = top_k(sim_a, k)
    nb, _ = top_k(sim_b, k)
    jaccard = np.mean(
        [
            len(set(a) & set(b)) / len(set(a) | set(b))
            for a, b in zip(na.tolist(), nb.tolist(), strict=True)
        ]
    )
    return {"spearman": float(rho), f"jaccard@{k}": float(jaccard)}

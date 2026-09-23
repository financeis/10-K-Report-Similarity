"""산업분류를 정답으로 쓴 평가 지표와 방법 간 일치도.

- precision@k: 텍스트로 찾은 상위 k개 이웃 중 같은 산업(label)인 비율.
  random은 무작위로 k개를 뽑았을 때의 기댓값이다.
- pair AUC: 임의의 '같은 산업 쌍'이 임의의 '다른 산업 쌍'보다 점수가 높을 확률.
  0.5면 무작위, 1이면 완벽하다. k를 정하지 않아도 되는 전체 순위 지표다.
"""

from __future__ import annotations

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


def precision_at_k(sim: np.ndarray, labels: np.ndarray, k: int) -> dict:
    ok = _valid(labels)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return {"precision": np.nan, "random": np.nan, "n": int(len(idx))}
    s = sim[np.ix_(idx, idx)]
    lab = labels[idx]
    neighbors, _ = top_k(s, k)
    precision = float((lab[neighbors] == lab[:, None]).mean())
    same = (lab[:, None] == lab[None, :]).sum(axis=1) - 1
    random = float((same / (len(idx) - 1)).mean())
    return {"precision": precision, "random": random, "n": int(len(idx))}


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

"""관계 유형별 주가 동조성 (docs/relation-map-plan.md 7.6 '연구 분석', 단계 2-3).

2024년 10-K가 밝힌 관계가 이듬해(2025년) 일별 주가에도 드러나는지 잰다. 잔차 상관은 기존 평가와
같다(returns.py): 종목마다 동일가중 시장(자기 제외)을 OLS로 뺀 일별 잔차 수익률의 상관이다.

1) 쌍 평균: 관계 유형별, GICS 위치(같은 서브산업 / 같은 섹터 / 다른 섹터)별 잔차 상관의 평균.
2) 기업쌍 회귀: 같은 GICS, 텍스트 유사도, 회사 고정효과(규모·공시 성향처럼 회사마다 다른 성질)를
   통제한 뒤에도 관계가 동조성을 설명하는지.

한 회사가 여러 쌍에 나오므로 쌍끼리 독립이 아니다. 구간은 회사 단위 재표집으로 구한다: 회사를
복원추출하고, 뽑힌 회사끼리의 쌍을 뽑힌 횟수의 곱만큼 쓴다(returns.incremental_effect와 같은 방식).
모든 계산을 회사×회사 행렬로 해서 쌍 목록(약 11만 개)을 재표집마다 만들지 않는다.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..config import Config
from ..pipeline import similarity_for
from ..returns import correlation_matrices, load_prices
from .reviews import edge_state
from .stages import apply_edge_reviews, open_graph, relations_config

log = logging.getLogger(__name__)

RELATIONS = ("competitor", "business", "equity")
SHOWN = ("accepted", "confirmed")
"""관계도에 선으로 보이는 상태 (모델 채택, 검수로 확인)."""

GROUPS = {
    "competitor": "경쟁",
    "business": "공급·협력",
    "equity": "지분 (검증 전)",
    "uncertain": "불확실 (검수 대기)",
    "mention_only": "회사 언급, 관계 아님",
    "name_only": "이름만 겹침 (다른 회사·다른 뜻)",
    "similar_only": "유사도 상위 {k}, 언급 없음",
    "same_sub": "같은 서브산업 전체",
    "all": "모든 쌍",
}
"""쌍 묶음 이름. {k}는 유사도 후보 수(relations.top_k)."""
POSITIONS = {
    "same_sub": "같은 서브산업",
    "same_sector": "같은 섹터, 다른 서브산업",
    "other_sector": "다른 섹터",
}
COEFS = {
    "competitor": "경쟁",
    "business": "공급·협력",
    "equity": "지분 (검증 전)",
    "uncertain": "불확실 (검수 대기)",
    "mention_only": "회사 언급, 관계 아님",
    "name_only": "이름만 겹침",
    "same_sub": "같은 서브산업 (같은 섹터에 더해)",
    "same_sector": "같은 섹터",
    "top20": "유사도 상위 {k} 이웃",
}
"""보고서 회귀표에 싣는 계수 (유사도 구간 더미는 '포함'으로만 적는다)."""
MODELS = {
    "gics": "(1) GICS",
    "text": "(2) + 텍스트 유사도",
    "firm": "(3) + 회사 고정효과",
}
"""보고서 회귀표의 모형. linear·linear_firm(유사도를 직선 하나로 통제)은 표 아래에 따로 적는다."""


@dataclass
class PairData:
    """분석 대상 회사(행렬 순서)와 회사×회사 행렬들."""

    firms: pd.DataFrame
    """node_id, ticker, name, gics_sector, gics_sub_industry."""
    corr: np.ndarray
    """잔차 상관."""
    sim: np.ndarray
    """텍스트 유사도 (관계도 후보에 쓴 방법, 기업쌍 백분위)."""
    usable: np.ndarray
    """분석에 쓰는 쌍: 두 회사 모두 수익률 관측치가 충분하고 i ≠ j."""
    pairs: dict[str, np.ndarray]
    """쌍 표시(대칭 bool): competitor, business, equity, uncertain, mentioned,
    entity_confirmed(판정 모델이 그 회사로 확인한 언급이 있음), top20, similar_only,
    same_sub, same_sector, gics_known."""
    info: dict


# ---------------------------------------------------------------- 계산 (행렬만 받는다)


def pair_groups(pairs: dict[str, np.ndarray], usable: np.ndarray) -> dict[str, np.ndarray]:
    """보고서의 쌍 묶음 (GROUPS). 관계 유형끼리는 겹칠 수 있다(경쟁이면서 공급·협력).
    '불확실'과 언급 묶음은 보이는 관계가 없는 쌍만이다. 관계로 채택되지 않은 언급은 판정 모델이
    그 회사로 확인했는지(회사 언급, 관계 아님)와 못 했는지(이름만 겹침: 제품명·일반 단어)로 나눈다."""
    related = pairs["competitor"] | pairs["business"] | pairs["equity"]
    out = {r: pairs[r] for r in RELATIONS}
    out["uncertain"] = pairs["uncertain"] & ~related
    not_related = pairs["mentioned"] & ~related & ~pairs["uncertain"]
    out["mention_only"] = not_related & pairs["entity_confirmed"]
    out["name_only"] = not_related & ~pairs["entity_confirmed"]
    out["similar_only"] = pairs["similar_only"]
    out["same_sub"] = pairs["same_sub"]
    out["all"] = np.ones_like(usable)
    return {k: v & usable for k, v in out.items()}


def positions(pairs: dict[str, np.ndarray], usable: np.ndarray) -> dict[str, np.ndarray]:
    """GICS 위치 (POSITIONS). 분류를 모르는 회사가 낀 쌍은 어디에도 넣지 않는다."""
    return {
        "same_sub": pairs["same_sub"] & usable,
        "same_sector": pairs["same_sector"] & ~pairs["same_sub"] & usable,
        "other_sector": pairs["gics_known"] & ~pairs["same_sector"] & usable,
    }


def boot_weights(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n): 재표집마다 회사가 뽑힌 횟수."""
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    return np.stack([np.bincount(d, minlength=n) for d in draws]).astype(np.float64)


def _boot_means(y: np.ndarray, g: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """재표집마다 묶음 g의 쌍 평균 (그 묶음의 쌍이 하나도 안 뽑힌 표본은 NaN).
    쌍 (i,j)를 두 번((i,j),(j,i)) 세지만 분자·분모에 똑같이 들어가 평균은 같다."""
    yg = np.where(g, y, 0.0)
    gf = g.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((weights @ yg) * weights).sum(1) / ((weights @ gf) * weights).sum(1)


def _interval(boots: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(boots)
    lo, hi = np.percentile(boots[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    return float(lo), float(hi)


def group_means(y: np.ndarray, groups: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    """묶음마다 쌍 평균과 회사 단위 재표집 95% 구간.

    groups: 이름 → 쌍 표시(n×n, 대칭, 대각 False). 재표집에서는 쌍마다 w_i·w_j를 곱한다.
    """
    out = {}
    finite = np.isfinite(y)
    for name, g in groups.items():
        g = g & finite
        n_pairs = int(g.sum()) // 2
        entry: dict = {"n_pairs": n_pairs, "n_firms": int(g.any(axis=1).sum())}
        if n_pairs == 0:
            out[name] = {**entry, "mean": None, "lo": None, "hi": None}
            continue
        lo, hi = _interval(_boot_means(y, g, weights))
        out[name] = {**entry, "mean": float(y[g].mean()), "lo": lo, "hi": hi}
    return out


def contrast(y: np.ndarray, a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> dict:
    """두 묶음의 평균 차이(a − b)와 회사 단위 재표집 95% 구간. 같은 재표집 표본으로 짝지어 잰다."""
    finite = np.isfinite(y)
    a, b = a & finite, b & finite
    if not a.any() or not b.any():
        return {"diff": None, "lo": None, "hi": None}
    lo, hi = _interval(_boot_means(y, a, weights) - _boot_means(y, b, weights))
    return {"diff": float(y[a].mean() - y[b].mean()), "lo": lo, "hi": hi}


def dyadic_ols(
    y: np.ndarray,
    covariates: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    firm_effects: bool = False,
) -> dict[str, float]:
    """기업쌍 회귀 y_ij = Σ β_p·x_p,ij (+ α_i + α_j) 의 β (쌍마다 한 번씩 센다).

    covariates: 이름 → n×n 대칭 행렬. firm_effects가 False면 절편(const)을 넣고, True면 회사마다
    α를 두어 규모·공시 성향 같은 회사 성질이 쌍에 더해지는 몫을 흡수한다. weights(회사별 뽑힌 횟수)가
    있으면 쌍 (i,j)를 w_i·w_j번 쓴 것과 같다. 정규방정식을 n×n 행렬 연산으로 바로 만든다.
    """
    n = y.shape[0]
    m = mask & np.isfinite(y)
    for c in covariates.values():
        m = m & np.isfinite(c)
    w = np.ones(n) if weights is None else weights
    omega = np.outer(w, w) * m
    # 쓰이는 쌍에서 값이 모두 0인 변수(쌍이 하나도 없는 묶음 등)는 추정할 수 없으므로 None으로 둔다
    values = {k: np.where(m, c, 0.0) for k, c in covariates.items()}
    names = [k for k, v in values.items() if (omega * v != 0).any()]
    missing = {k: None for k in covariates if k not in names}
    xs = [values[k] for k in names]
    if not firm_effects:
        names, xs = ["const", *names], [m, *xs]
    x = np.stack(xs).astype(np.float64).reshape(len(xs), -1)  # (변수, n·n)
    wx = x * omega.reshape(-1)
    yv = np.where(m, y, 0.0).reshape(-1)
    # 행렬 원소를 모두 더하면 쌍마다 (i,j)와 (j,i) 두 번 세므로 2로 나눈다
    xtx, xty = wx @ x.T / 2, wx @ yv / 2
    k = len(names)
    if firm_effects:
        # α 블록: 쌍 (i,j)의 설계 행은 e_i + e_j. 뽑히지 않은 회사는 뺀다.
        keep = omega.sum(1) > 0
        o = omega[np.ix_(keep, keep)]
        x_a = wx.reshape(k, n, n).sum(2)[:, keep].T
        a_a = o + np.diag(o.sum(1))
        full = np.block([[xtx, x_a.T], [x_a, a_a]])
        rhs = np.concatenate([xty, (omega.reshape(-1) * yv).reshape(n, n).sum(1)[keep]])
    else:
        full, rhs = xtx, xty
    coef, *_ = np.linalg.lstsq(full, rhs, rcond=None)
    return dict(zip(names, coef[:k].tolist(), strict=True)) | missing


def pair_regression(
    y: np.ndarray,
    covariates: dict[str, np.ndarray],
    mask: np.ndarray,
    weights: np.ndarray,
    *,
    firm_effects: bool = False,
) -> dict:
    """dyadic_ols의 계수와 회사 단위 재표집 95% 구간. 추정할 수 없는 계수는 None.
    재표집에서 그 변수가 빠진 표본(해당 쌍의 회사가 하나도 안 뽑힘)은 그 계수의 구간에서만 뺀다."""
    point = dyadic_ols(y, covariates, mask, firm_effects=firm_effects)
    boots = [dyadic_ols(y, covariates, mask, weights=w, firm_effects=firm_effects) for w in weights]
    coefs = {}
    for name, value in point.items():
        draws = [b[name] for b in boots if b[name] is not None]
        if value is None or not draws:
            coefs[name] = {"coef": value, "lo": None, "hi": None, "n_boot": len(draws)}
            continue
        lo, hi = np.percentile(draws, [2.5, 97.5])
        coefs[name] = {"coef": value, "lo": float(lo), "hi": float(hi), "n_boot": len(draws)}
    n_pairs = int((mask & np.isfinite(y)).sum()) // 2
    return {"coefs": coefs, "n_pairs": n_pairs, "n_boot": len(weights)}


def analyze(
    data: PairData, *, n_boot: int = 1000, n_boot_regression: int = 200, seed: int = 0
) -> dict:
    y, usable, pairs = data.corr, data.usable, data.pairs
    groups = pair_groups(pairs, usable)
    where = positions(pairs, usable)
    weights = boot_weights(len(data.firms), n_boot, seed)

    means = group_means(y, groups, weights)
    for name, g in groups.items():
        n = int(g.sum())
        means[name]["share_same_sub"] = float((g & pairs["same_sub"]).sum() / n) if n else None
        means[name]["share_top20"] = float((g & pairs["top20"]).sum() / n) if n else None
    by_position = {
        name: group_means(y, {p: g & at for p, at in where.items()}, weights)
        for name, g in groups.items()
        if name != "same_sub"
    }
    top20 = pairs["top20"]
    by_similarity = {
        name: group_means(y, {"in": groups[name] & top20, "out": groups[name] & ~top20}, weights)
        for name in (*RELATIONS, "uncertain", "mention_only", "name_only", "all")
    }

    # 판정 모델이 관계로 채택한 쌍이, 그 회사로 확인했지만 관계가 아니라고 본 언급 쌍보다 더 같이 움직이나
    contrasts = {
        f"{r}{suffix}": contrast(y, groups[r] & at, groups["mention_only"] & at, weights)
        for r in ("competitor", "business")
        for suffix, at in (("", usable), ("_other_sector", where["other_sector"]))
    }

    pct = pair_percentile(data.sim, usable)
    for name, g in groups.items():
        upper = np.triu(g, 1)
        means[name]["median_similarity_pct"] = float(np.median(pct[upper])) if upper.any() else None

    # 유사도 통제: 백분위 구간 더미. 관계 쌍은 유사도 최상위에 몰려 있고 그 구간에서 동조성이
    # 가파르게 오르므로, 유사도를 직선 하나로 넣으면 그 몫이 관계 계수로 새어 나온다(linear 모형).
    finite = usable & np.isfinite(data.sim)
    z = (data.sim - data.sim[finite].mean()) / data.sim[finite].std()
    base = {r: groups[r] for r in (*RELATIONS, "uncertain", "mention_only", "name_only")}
    base |= {"same_sub": pairs["same_sub"], "same_sector": pairs["same_sector"]}
    text = similarity_bins(pct) | {"top20": pairs["top20"]}
    linear = {"sim_z": z, "top20": pairs["top20"]}
    specs = {
        "gics": (base, False),
        "text": (base | text, False),
        "firm": (base | text, True),
        "linear": (base | linear, False),
        "linear_firm": (base | linear, True),
    }
    reg_weights = weights[:n_boot_regression]
    regression = {}
    for name, (cov, fe) in specs.items():
        cov = {k: v.astype(np.float64) for k, v in cov.items()}
        regression[name] = pair_regression(y, cov, usable, reg_weights, firm_effects=fe)
        log.info("회귀 %s 완료", name)

    return {
        "info": data.info,
        "n_boot": n_boot,
        "sim_bins": list(SIM_BINS),
        "groups": means,
        "positions": by_position,
        "contrasts": contrasts,
        "similarity": by_similarity,
        "regression": regression,
        "examples": examples(data, pct, groups, where),
    }


SIM_BINS = (50, 75, 90, 95, 98, 99, 99.5)
"""유사도 구간의 경계 (기업쌍 백분위). 50 미만이 기준 구간이다."""


def pair_percentile(sim: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """쌍 유사도의 백분위 (분석 대상 쌍 안에서 0~100, 대칭). 대상 밖 쌍은 NaN."""
    upper = np.triu(usable & np.isfinite(sim), 1)
    values = sim[upper]
    ranks = np.empty(len(values))
    ranks[values.argsort(kind="stable")] = np.arange(len(values))
    pct = np.full(sim.shape, np.nan)
    pct[upper] = 100 * ranks / max(len(values) - 1, 1)
    return np.where(np.isnan(pct), pct.T, pct)


def similarity_bins(pct: np.ndarray) -> dict[str, np.ndarray]:
    """SIM_BINS 구간마다 쌍 표시 (백분위 [경계, 다음 경계))."""
    edges = [*SIM_BINS, np.inf]
    with np.errstate(invalid="ignore"):
        return {
            f"sim_{lo:g}": (pct >= lo) & (pct < hi)
            for lo, hi in zip(edges, edges[1:], strict=False)
        }


def examples(
    data: PairData, pct: np.ndarray, groups: dict[str, np.ndarray], where: dict[str, np.ndarray]
) -> dict:
    related = groups["competitor"] | groups["business"] | groups["equity"]
    return {
        "cross_sector_business": top_pairs(
            data, pct, groups["business"] & where["other_sector"], 10
        ),
        "beyond_similarity": top_pairs(data, pct, related & ~data.pairs["top20"], 10),
        "negative_competitors": top_pairs(data, pct, groups["competitor"], 5, lowest=True),
        "mention_only": top_pairs(data, pct, groups["mention_only"], 10),
    }


def top_pairs(
    data: PairData, pct: np.ndarray, mask: np.ndarray, n: int, *, lowest: bool = False
) -> list[dict]:
    """쌍 표시에서 잔차 상관이 가장 높은(lowest면 낮은) 쌍 n개. pct: 쌍 유사도 백분위."""
    a, b = np.nonzero(np.triu(mask & np.isfinite(data.corr), 1))
    values = data.corr[a, b]
    order = np.argsort(values if lowest else -values)[:n]
    firms = data.firms
    out = []
    for i, j in zip(a[order], b[order], strict=True):
        rel = [r for r in (*RELATIONS, "uncertain") if data.pairs[r][i, j]]
        out.append(
            {
                "a": _firm(firms.iloc[i]),
                "b": _firm(firms.iloc[j]),
                "corr": float(data.corr[i, j]),
                "similarity_pct": float(pct[i, j]),
                "top20": bool(data.pairs["top20"][i, j]),
                "relations": rel,
            }
        )
    return out


def _firm(row: pd.Series) -> dict:
    return {k: row[k] for k in ("ticker", "name", "gics_sector", "gics_sub_industry")}


# ---------------------------------------------------------------- 데이터 읽기


def shown_edges(cfg: Config, con) -> pd.DataFrame:
    """graph.db의 관계에 reviews.sqlite의 최신 검수를 덧씌우고 최종 상태(state)를 붙인다.
    화면(app/server.py)과 같은 규칙이라, export를 다시 하지 않아도 검수가 반영된다."""
    edges = pd.read_sql(
        "SELECT edge_id, src, dst, relation, decision, review_state FROM edges", con
    )
    evidence = pd.read_sql("SELECT edge_id, span_id FROM edge_evidence", con)
    apply_edge_reviews(cfg, con, SimpleNamespace(edges=edges, edge_evidence=evidence))
    edges["state"] = [
        edge_state(d, r) for d, r in zip(edges["decision"], edges["review_state"], strict=True)
    ]
    return edges


def _matrix(n: int, pos: dict[str, int], pairs) -> np.ndarray:
    """(node_a, node_b) 목록 → 대칭 bool 행렬. 분석 대상 밖 회사가 낀 쌍은 뺀다."""
    m = np.zeros((n, n), dtype=bool)
    for a, b in pairs:
        i, j = pos.get(a), pos.get(b)
        if i is not None and j is not None and i != j:
            m[i, j] = m[j, i] = True
    return m


def _coverage(edges: pd.DataFrame, kind: dict, pos: dict, usable: np.ndarray) -> dict:
    """관계도에 보이는 관계 가운데 분석에 들어간 것과 빠진 이유별 개수.
    external: 한쪽이 외부·익명 기업 / outside_universe: 텍스트 유사도가 없는 회사(10-K 추출 문제 등) /
    no_returns: 주가 관측치가 부족함."""
    out = {"shown": len(edges), "used": 0, "external": 0, "outside_universe": 0, "no_returns": 0}
    for a, b in zip(edges["src"], edges["dst"], strict=True):
        if kind.get(a) != "company" or kind.get(b) != "company":
            out["external"] += 1
        elif a not in pos or b not in pos:
            out["outside_universe"] += 1
        elif not usable[pos[a], pos[b]]:
            out["no_returns"] += 1
        else:
            out["used"] += 1
    return out


def _split(pair_keys) -> list[tuple[str, str]]:
    return [tuple(k.split("|")) for k in pair_keys]


def load_pair_data(cfg: Config) -> PairData:
    rc = cfg.evaluation.returns
    if rc is None:
        raise SystemExit("설정에 evaluation.returns(수익률 기간)가 없습니다")
    rel = relations_config(cfg)
    if rel.similarity is None:
        raise SystemExit("relations.similarity가 없습니다 (통제 변수로 쓸 텍스트 유사도)")
    companies, sim = similarity_for(cfg, rel.similarity)
    tickers = companies["ticker"].tolist()
    prices = load_prices(
        tickers, rc.market, rc.start, rc.end, cfg.run_dir / f"prices_{rc.start}_{rc.end}.parquet"
    )
    _, corr, enough = correlation_matrices(prices, tickers, rc.market, rc.min_obs)

    with closing(open_graph(cfg)) as con:
        nodes = pd.read_sql(
            "SELECT node_id, kind, ticker, name, gics_sector, gics_sub_industry FROM nodes", con
        )
        edges = shown_edges(cfg, con)
        units = pd.read_sql(
            "SELECT pair_key, MAX(d_is_entity = 'yes') AS entity FROM unit_judgements "
            "GROUP BY pair_key",
            con,
        )
        cands = pd.read_sql("SELECT pair_key, source, status FROM candidates", con)
        meta = {k: v for k, v in con.execute("SELECT key, value FROM meta")}

    firms = companies[["cik", "ticker"]].assign(node_id=[f"cik:{c}" for c in companies["cik"]])
    companies_only = nodes[nodes["kind"] == "company"].drop(columns=["kind", "ticker"])
    firms = firms.merge(companies_only, on="node_id", how="left")
    n = len(firms)
    pos = {node: i for i, node in enumerate(firms["node_id"])}
    usable = enough[:, None] & enough[None, :] & ~np.eye(n, dtype=bool) & np.isfinite(corr)
    kind = dict(zip(nodes["node_id"], nodes["kind"], strict=True))

    pairs: dict[str, np.ndarray] = {}
    coverage = {}
    for r in RELATIONS:
        shown = edges[(edges["relation"] == r) & edges["state"].isin(SHOWN)]
        pairs[r] = _matrix(n, pos, zip(shown["src"], shown["dst"], strict=True))
        coverage[r] = _coverage(shown, kind, pos, usable)
    uncertain = edges[edges["state"] == "uncertain"]
    pairs["uncertain"] = _matrix(n, pos, zip(uncertain["src"], uncertain["dst"], strict=True))
    pairs["mentioned"] = _matrix(n, pos, _split(units["pair_key"]))
    # 판정 모델이 그 회사로 확인한 언급이 있는 쌍. 관계가 만들어진 쌍(검수로 거절한 것 포함)도 확인된 것으로 본다
    confirmed = set(units.loc[units["entity"] == 1, "pair_key"])
    confirmed |= {f"{a}|{b}" for a, b in zip(edges["src"], edges["dst"], strict=True)}
    pairs["entity_confirmed"] = _matrix(n, pos, _split(confirmed))
    top20 = cands[cands["source"] != "mention"]["pair_key"]
    pairs["top20"] = _matrix(n, pos, _split(top20))
    pairs["similar_only"] = _matrix(n, pos, _split(cands[cands["status"] == "similar"]["pair_key"]))

    # 분석 대상 밖이라 관계가 빠진 S&P 500 회사 (텍스트 유사도가 없음)
    ticker = dict(zip(nodes["node_id"], nodes["ticker"], strict=True))
    shown_all = edges[edges["state"].isin(SHOWN)]
    outside = {
        node
        for a, b in zip(shown_all["src"], shown_all["dst"], strict=True)
        for node in (a, b)
        if kind.get(node) == "company" and node not in pos
    }

    sub = firms["gics_sub_industry"].to_numpy(dtype=object)
    sec = firms["gics_sector"].to_numpy(dtype=object)
    known = np.array([isinstance(s, str) for s in sub]) & np.array(
        [isinstance(s, str) for s in sec]
    )
    pairs["gics_known"] = known[:, None] & known[None, :]
    pairs["same_sub"] = (sub[:, None] == sub[None, :]) & pairs["gics_known"]
    pairs["same_sector"] = (sec[:, None] == sec[None, :]) & pairs["gics_known"]

    judge = json.loads(meta.get("judge", "{}"))
    info = {
        "config": cfg.name,
        "filings_year": cfg.filings.year,
        "start": str(rc.start),
        "end": str(rc.end),
        "market": rc.market,
        "min_obs": rc.min_obs,
        "similarity": rel.similarity,
        "top_k": rel.top_k,
        "n_firms": int(enough.sum()),
        "n_firms_similarity": n,
        "n_pairs": int(usable.sum()) // 2,
        "graph_created": meta.get("created"),
        "judge_model": judge.get("model"),
        "question_version": judge.get("question_version"),
        "validated": json.loads(meta.get("validated_relations", "[]")),
        "coverage": coverage,
        "outside_companies": sorted(str(ticker[x]) for x in outside),
        "reviews": {k: int(v) for k, v in edges["review_state"].value_counts().items()},
    }
    return PairData(firms, corr, sim, usable, pairs, info)


# ---------------------------------------------------------------- 보고서


def _num(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{digits}f}"


MIN_CI_PAIRS = 5
"""쌍이 이보다 적으면 구간을 싣지 않는다 (쌍 하나면 재표집 구간이 한 점이 된다)."""


def _ci(e: dict) -> str:
    if e.get("mean") is None:
        return "–"
    if e.get("n_pairs", MIN_CI_PAIRS) < MIN_CI_PAIRS:
        return f"{e['mean']:.3f}"
    return f"{e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


def _diff(e: dict) -> str:
    """차이와 구간. 구간이 0을 포함하지 않으면 굵게."""
    if e.get("diff") is None:
        return "–"
    text = f"{e['diff']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
    return f"**{text}**" if _significant(e) else text


def _cell(e: dict) -> str:
    """'평균 [구간] · 쌍 수'"""
    return "–" if e["mean"] is None else f"{_ci(e)} · {e['n_pairs']:,}"


def _estimated(e: dict | None) -> bool:
    return bool(e) and e.get("coef") is not None and e.get("lo") is not None


def _coef(e: dict | None) -> str:
    """계수와 구간. 구간이 0을 포함하지 않으면 굵게. 추정할 수 없으면(쌍이 없음) '–'."""
    if not _estimated(e):
        return "–"
    text = f"{e['coef']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
    return f"**{text}**" if _significant(e) else text


def _significant(e: dict) -> bool:
    """구간이 0을 포함하지 않는가. 보고서에 적는 자릿수(소수 셋째 자리)로 반올림해 판단한다
    (그래야 '[+0.000, …]'이 굵게 나오는 일이 없다)."""
    return round(e["lo"], 3) > 0 or round(e["hi"], 3) < 0


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _name(f: dict) -> str:
    return f"{f['name']} ({f['ticker']})"


def _label(key: str, top_k: int) -> str:
    return GROUPS[key].format(k=top_k)


def _short(key: str) -> str:
    return GROUPS[key].split(" (")[0]


def _dropped(c: dict) -> str:
    """'62/564 (외부 기업 39, 분석 대상 밖 S&P 500 23)'"""
    parts = [
        f"{label} {c[key]}"
        for key, label in (
            ("external", "외부 기업"),
            ("outside_universe", "분석 대상 밖 S&P 500"),
            ("no_returns", "주가 부족"),
        )
        if c[key]
    ]
    n = c["shown"] - c["used"]
    return f"{n}/{c['shown']}" + (f" ({', '.join(parts)})" if parts else "")


def format_report(result: dict) -> str:
    info, g = result["info"], result["groups"]
    cov, top_k = info["coverage"], info["top_k"]
    bins = "·".join(f"{b:g}" for b in result["sim_bins"])
    outside = info.get("outside_companies") or []
    outside_text = f"(이번에는 {', '.join(outside)})" if outside else ""
    lines = [
        f"# {info['config']} 관계 유형별 주가 동조성",
        "",
        f"- 생성: {datetime.now().isoformat(timespec='seconds')}",
        f"- 관계: {info['filings_year']}년 제출 10-K의 Item 1·1A에서 판정한 관계입니다 "
        f"(`graph.db` {info['graph_created']}, 판정 `{info['judge_model']}` 질문 "
        f"{info['question_version']}). 관계도에 보이는 관계(모델 채택 + 검수로 확인, 검수로 거절한 것은 뺌)만 씁니다.",
        f"- 수익률: {info['start']} ~ {info['end']} 일별 수익률에서 동일가중 시장(자기 제외)을 OLS로 뺀 "
        "잔차의 상관입니다. 관계를 밝힌 10-K보다 뒤의 기간입니다.",
        f"- 대상: 텍스트 유사도를 계산한 {info['n_firms_similarity']}개사 중 거래일이 "
        f"{info['min_obs']}일 이상인 {info['n_firms']}개사, {info['n_pairs']:,}쌍입니다. "
        "그 밖의 회사와의 관계는 뺍니다: 외부 기업(해외 기업, 익명 고객)과, "
        f"10-K 추출 품질 문제로 텍스트 유사도가 없는 S&P 500 회사{outside_text}입니다.",
        f"- 텍스트 유사도는 관계도 후보에 쓴 `{info['similarity']}`이고, '유사도 백분위'는 분석 대상 "
        "모든 쌍 안에서의 순위입니다(100이 가장 비슷함).",
        f"- 괄호 안 구간은 회사 단위 부트스트랩 95% 구간입니다(평균 {result['n_boot']}회, "
        f"회귀 {result['regression']['gics']['n_boot']}회). 한 회사가 여러 쌍에 나오므로 회사를 "
        f"복원추출하고, 뽑힌 회사끼리의 쌍으로 다시 계산했습니다. 쌍이 {MIN_CI_PAIRS}개 미만이면 "
        "평균만 싣습니다.",
        "- 지분 관계는 확인 표본에 정답이 없어 판정 정확도를 검증하지 못했습니다(검증 전). 수치는 참고만 하세요.",
        "",
        "## 1. 요약",
        "",
        *_summary(result),
        "",
        "## 2. 유형별 평균 잔차 상관",
        "",
        "쌍 하나를 한 번씩 센 평균입니다. 관계 유형끼리는 겹칠 수 있습니다(경쟁이면서 공급·협력인 쌍). "
        "관계로 채택되지 않은 이름 언급은 둘로 나눴습니다.",
        "",
        "- '회사 언급, 관계 아님': 판정 모델이 그 회사를 가리킨다고 확인했지만 세 관계 모두 채택하지 않은 쌍"
        "(검수로 거절한 관계 포함). 예: 임원 약력의 전 직장, 소송·협의 상대",
        "- '이름만 겹침': 이름은 나왔지만 판정 모델이 그 회사가 아니라고 보거나 확인하지 못한 쌍. "
        "예: 제품명(Ciena의 'Coherent ELS')이나 일반 단어가 회사명과 같은 경우",
        f"- '유사도 상위 {top_k}, 언급 없음': 텍스트 유사도로 어느 한쪽의 상위 {top_k}에 들지만 "
        "서로 이름을 적지 않은 쌍",
        "",
    ]
    rows = [
        [_label(key, top_k), _ci(e := g[key]), f"{e['n_pairs']:,}", f"{e['n_firms']:,}",
         _pct(e.get("share_same_sub")), _pct(e.get("share_top20")),
         _num(e.get("median_similarity_pct"), 1)]
        for key in GROUPS
    ]  # fmt: skip
    lines += _table(
        ["쌍 묶음", "평균 잔차 상관 [95% 구간]", "쌍", "회사", "같은 서브산업",
         f"유사도 상위 {top_k}", "유사도 백분위 중앙값"],
        rows,
    )  # fmt: skip
    dropped = ", ".join(f"{_short(r)} {_dropped(cov[r])}" for r in RELATIONS)
    con = result["contrasts"]
    lines += [
        "",
        "- 관계 쌍과 '회사 언급, 관계 아님' 쌍의 차이(같은 재표집 표본으로 짝지어 잼): "
        f"경쟁 {_diff(con['competitor'])}, 공급·협력 {_diff(con['business'])}. "
        f"섹터가 다른 쌍만 보면 경쟁 {_diff(con['competitor_other_sector'])}, "
        f"공급·협력 {_diff(con['business_other_sector'])}. 판정 모델이 관계로 채택한 쌍이, 그 회사로 "
        "확인했지만 관계가 아니라고 본 쌍보다 더 같이 움직이는지를 봅니다.",
        f"- 관계도에 보이는 관계 가운데 분석에서 빠진 것: {dropped}.",
        f"- '같은 서브산업', '유사도 상위 {top_k}' 칸은 그 묶음의 쌍 가운데 GICS 서브산업이 같은 쌍, "
        "텍스트 유사도 후보에도 들어 있던 쌍의 비율입니다.",
        "",
        "## 3. GICS 위치별",
        "",
        "관계가 GICS 분류가 이미 묶는 연결을 다시 찾은 것인지 보려고, 같은 위치의 모든 쌍과 나란히 놓습니다. "
        "칸마다 평균 [95% 구간] · 쌍 수입니다.",
        "",
    ]
    rows = [
        [_label(key, top_k), *(_cell(by_pos[p]) for p in POSITIONS)]
        for key, by_pos in result["positions"].items()
    ]
    lines += _table(["쌍 묶음", *POSITIONS.values()], rows)
    lines += [
        "",
        "## 4. 텍스트 유사도 후보 안·밖",
        "",
        f"텍스트 유사도로 어느 한쪽의 상위 {top_k}에 들었는지로 나눕니다. '밖' 열은 유사도만으로 "
        "후보를 만들었다면 놓쳤을 쌍입니다. 칸마다 평균 [95% 구간] · 쌍 수입니다.",
        "",
    ]
    rows = [
        [_label(key, top_k), *(_cell(e) for e in split.values())]
        for key, split in result["similarity"].items()
    ]
    lines += _table(["쌍 묶음", f"상위 {top_k} 안", f"상위 {top_k} 밖"], rows)
    reg = result["regression"]
    lines += [
        "",
        "## 5. 기업쌍 회귀: 분류·유사도·회사 성질을 통제하면",
        "",
        "모든 쌍(i, j)의 잔차 상관을 관계 표시와 통제 변수로 회귀했습니다. 계수는 다른 조건이 같을 때 "
        "그 표시가 있는 쌍의 잔차 상관이 얼마나 높은지입니다. 구간이 0을 포함하지 않으면 **굵게** 표시합니다.",
        "",
        "- 관계·언급 행(경쟁부터 이름만 겹침까지)의 비교 기준은 **관계도 이름 언급도 없는 쌍**입니다.",
        "- 같은 섹터와 유사도 상위 이웃 행은 그 표시가 없는 쌍과 비교한 값입니다. 같은 서브산업 행은 같은 섹터 "
        "위에 더해지는 몫이라, 같은 서브산업 쌍의 효과는 두 계수의 합입니다.",
        "- 쌍이 하나도 없는 묶음은 추정할 수 없어 '–'로 둡니다.",
        "",
        "통제 변수:",
        "",
        "- (1) GICS: 같은 서브산업, 같은 섹터",
        f"- (2) + 텍스트 유사도: 유사도 백분위 구간 더미(경계 {bins}, 50 미만이 기준)와 "
        f"'유사도 상위 {top_k} 이웃' 표시. 유사도로 이미 가까운 회사끼리라서 같이 움직이는 몫을 뺍니다",
        "- (3) + 회사 고정효과: 회사마다 상수를 두어 규모, 변동성, 이름을 많이 적는 공시 성향처럼 회사 한쪽의 "
        "성질에서 오는 몫을 뺍니다. 시가총액 자료가 없어 규모는 이 방식으로만 통제합니다",
        "",
    ]
    # 모형에 없는 변수는 빈칸, 모형에 있지만 추정할 수 없는 변수(쌍 없음)는 '–'
    rows = [
        [label.format(k=top_k),
         *("" if key not in reg[m]["coefs"] else _coef(reg[m]["coefs"][key]) for m in MODELS)]
        for key, label in COEFS.items()
    ]  # fmt: skip
    rows.append([f"유사도 구간 ({len(result['sim_bins'])}개)", "", "포함", "포함"])
    rows.append(["쌍 수", *(f"{reg[m]['n_pairs']:,}" for m in MODELS)])
    lines += _table(["", *MODELS.values()], rows)
    lin, lin_fe = reg["linear"]["coefs"], reg["linear_firm"]["coefs"]
    lines += [
        "",
        "**유사도를 직선 하나로만 넣으면 결과가 달라집니다.** 유사도를 표준화한 값 하나(와 상위 이웃 표시)로 "
        f"통제하면 경쟁 {_coef(lin.get('competitor'))}, 공급·협력 {_coef(lin.get('business'))}"
        f"(회사 고정효과를 더하면 경쟁 {_coef(lin_fe.get('competitor'))}, "
        f"공급·협력 {_coef(lin_fe.get('business'))})입니다. "
        "관계 쌍은 유사도 최상위에 몰려 있고(2장의 백분위 중앙값), 그 구간에서 동조성이 가파르게 오릅니다. "
        "직선은 이 굴곡을 따라가지 못해 유사도의 몫이 관계 계수로 넘어가므로, 구간 더미를 쓴 (2)·(3)을 기준으로 읽습니다.",
    ]
    ex = result["examples"]
    lines += [
        "",
        "## 6. 예시",
        "",
        "### 다른 섹터의 공급·협력 관계, 잔차 상관 상위",
        "",
        *_example_table(ex["cross_sector_business"]),
        "",
        f"### 텍스트 유사도 상위 {top_k}에 없던 관계, 잔차 상관 상위",
        "",
        "유사도만으로 후보를 만들었다면 놓쳤을 관계입니다.",
        "",
        *_example_table(ex["beyond_similarity"]),
        "",
        "### 잔차 상관이 가장 낮은 경쟁 관계",
        "",
        *_example_table(ex["negative_competitors"]),
        "",
        "### 회사 언급, 관계 아님: 잔차 상관 상위",
        "",
        "판정 모델이 그 회사로 확인했지만 관계로 채택하지 않은 언급 가운데 주가가 가장 같이 움직인 쌍입니다. "
        "판정 누락을 점검할 때 먼저 볼 후보입니다. 같은 산업이라 같이 움직일 뿐 관계 문장이 아닌 경우도 많습니다"
        "(예: 임원 약력의 전 직장).",
        "",
        *_example_table(ex["mention_only"], relation=False),
        "",
        "## 7. 해석할 때 주의할 점",
        "",
        "- 일별 잔차 상관은 두 회사가 같은 뉴스·수요·원가 요인에 함께 노출됐는지를 봅니다. 관계가 주가를 "
        "움직인다는 인과를 뜻하지 않고, 관계의 방향(누가 누구에게 파는지)도 모릅니다.",
        "- 경쟁 관계는 한 회사의 호재가 경쟁사에 악재가 되는 효과(반대 방향)와, 같은 산업 충격(같은 방향)이 "
        "섞입니다. 일별 상관의 부호만으로는 둘을 나누기 어렵습니다. 실적 발표일 반응을 보는 이벤트 스터디는 "
        "3단계에서 합니다(계획서 7.6).",
        "- 5장의 (2)·(3)은 '텍스트 유사도가 이미 알려주는 것 위에 관계가 더 알려주는 것이 있나'를 묻습니다. "
        "경쟁사·거래처끼리 사업 설명이 비슷한 것은 자연스러우므로, 계수가 0에 가깝다는 것은 관계가 "
        "무의미하다는 뜻이 아니라 같은 날 동조성에 관해서는 유사도와 같은 정보라는 뜻입니다.",
        "- 같은 날의 상관만 봅니다. 고객사 소식이 며칠 뒤 공급사 주가에 반영되는 것 같은 시차 효과"
        "(Cohen & Frazzini 2008)는 이 분석에 잡히지 않습니다.",
        "- 관계는 이름을 적은 공시에서만 나옵니다. 이름을 밝히지 않는 고객·공급사, S&P 500 밖 회사와의 관계는 "
        "빠져 있습니다.",
        "- 쌍 평균은 관계가 많은 회사(예: Microsoft)의 비중이 큽니다. 회사 단위 재표집 구간이 그 불확실성을 반영합니다.",
    ]
    return "\n".join(lines)


def _summary(result: dict) -> list[str]:
    g, pos, reg = result["groups"], result["positions"], result["regression"]
    top_k, con = result["info"]["top_k"], result["contrasts"]
    cross = {k: pos[k]["other_sector"] for k in ("competitor", "business", "mention_only", "all")}
    moves = all(
        g[r]["lo"] is not None and g[r]["lo"] > g["all"]["hi"] for r in ("competitor", "business")
    )
    lead = "**관계가 있는 쌍은 주가가 같이 움직입니다.** " if moves else ""
    out = [
        f"- {lead}평균 잔차 상관: 경쟁 {_ci(g['competitor'])} "
        f"({g['competitor']['n_pairs']:,}쌍), 공급·협력 {_ci(g['business'])} ({g['business']['n_pairs']:,}쌍), "
        f"회사 언급·관계 아님 {_ci(g['mention_only'])}, 이름만 겹침 {_num(g['name_only']['mean'])}. "
        f"비교: 모든 쌍 {_num(g['all']['mean'])}, 같은 서브산업 전체 {_num(g['same_sub']['mean'])}, "
        f"유사도 상위 {top_k}·언급 없음 {_num(g['similar_only']['mean'])}.",
        f"- 섹터가 다른 쌍만 보면: 공급·협력 {_ci(cross['business'])}, 경쟁 {_num(cross['competitor']['mean'])}, "
        f"회사 언급·관계 아님 {_num(cross['mention_only']['mean'])}, 섹터가 다른 모든 쌍 "
        f"{_num(cross['all']['mean'])}.",
        "- 관계 쌍 − 회사 언급·관계 아님 쌍(2장): "
        f"경쟁 {_diff(con['competitor'])}, 공급·협력 {_diff(con['business'])}, "
        f"섹터가 다른 쌍의 공급·협력 {_diff(con['business_other_sector'])}.",
        f"- 관계 쌍은 텍스트 유사도 최상위에 몰려 있습니다(유사도 백분위 중앙값: 경쟁 "
        f"{_num(g['competitor']['median_similarity_pct'], 1)}, 공급·협력 "
        f"{_num(g['business']['median_similarity_pct'], 1)}).",
    ]
    for r in ("competitor", "business"):
        if g[r]["n_pairs"] == 0:
            out.append(f"- {GROUPS[r]}: 분석할 쌍이 없습니다.")
            continue
        steps = " → ".join(
            f"{MODELS[m].split(' ')[0]} {_coef(reg[m]['coefs'].get(r))}" for m in MODELS
        )
        out.append(f"- {GROUPS[r]}의 초과 동조성(회귀 계수, 5장): {steps}. {_verdict(reg, r)}")
    return out


def _verdict(reg: dict, r: str) -> str:
    """(1)·(2)·(3)의 구간을 함께 보고 한 문장으로 읽는다."""
    gics, text, firm = (reg[m]["coefs"].get(r) for m in MODELS)
    if not all(_estimated(e) for e in (gics, text, firm)):
        return "추정할 수 없는 모형이 있습니다(쌍이 너무 적음)."
    if _significant(firm):
        return (
            "모두 통제해도 관계가 없는 쌍보다 잔차 상관이 뚜렷하게 "
            f"{'높습니다' if firm['lo'] > 0 else '낮습니다'}."
        )
    if not _significant(gics):
        return "GICS만 통제해도 구간이 0을 포함해, 관계가 없는 쌍과 뚜렷한 차이가 없습니다."
    if not _significant(text):
        return (
            "GICS만 통제했을 때 보이던 초과 동조성이 텍스트 유사도를 더하면 사라집니다. "
            "같이 움직이는 몫은 대부분 GICS와 텍스트 유사도로 설명됩니다."
        )
    return "텍스트 유사도까지 통제해도 남지만, 회사 고정효과를 더하면 구간이 0을 포함합니다."


def _example_table(items: list[dict], *, relation: bool = True) -> list[str]:
    if not items:
        return ["해당하는 쌍이 없습니다."]
    rows = []
    for x in items:
        row = [
            _name(x["a"]),
            x["a"]["gics_sector"] or "",
            _name(x["b"]),
            x["b"]["gics_sector"] or "",
        ]
        if relation:
            row.append(" + ".join(_short(r) for r in x["relations"]))
        rows.append([*row, f"{x['corr']:.2f}", _num(x["similarity_pct"], 1)])
    headers = [
        "회사 A",
        "섹터",
        "회사 B",
        "섹터",
        *(["관계"] if relation else []),
        "잔차 상관",
        "유사도 백분위",
    ]
    return _table(headers, rows)


def report_path(cfg: Config) -> Path:
    return cfg.reports_dir / f"relations_{cfg.name}.md"


def stage_research(cfg: Config, *, n_boot: int | None = None) -> tuple[dict, Path]:
    """분석을 돌려 결과(relations/comovement.json)와 보고서(reports/relations_<name>.md)를 쓴다."""
    data = load_pair_data(cfg)
    n_boot = n_boot or cfg.evaluation.n_boot
    result = analyze(data, n_boot=n_boot, n_boot_regression=max(50, n_boot // 5))
    out = cfg.relations_dir / "comovement.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(result) + "\n", encoding="utf-8")
    return result, path

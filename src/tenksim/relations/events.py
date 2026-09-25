"""실적 발표 때의 주가 전이: 이벤트 스터디 (docs/relation-map-plan.md 10장, 단계 3-1).

2-3(research.py)은 같은 날의 일별 동조성을 봤다. 여기서는 한 회사가 실적을 발표할 때(8-K Item 2.02,
earnings.py) 관계가 있는 상대 회사의 주가가 어떻게 반응하는지 본다(기업 간 정보 전이, Foster 1981).

- 뉴스 크기: 발표 회사의 누적 비정상 수익률 CAR[0,+1] (0일 = 반응 첫 거래일). 애널리스트 컨센서스가
  없어 발표 회사 주가의 반응 자체를 뉴스의 크기로 쓴다.
- 상대 반응: 같은 창의 상대 회사 CAR. 비정상 수익률은 2-3과 같은 시장 모형 잔차다(동일가중 시장,
  자기 제외, 수익률 기간 전체로 추정). 상대의 시장 평균에서는 발표 회사를 뺀다(announcer_out).
- 반응 계수: 상대 CAR을 발표 회사 CAR에 회귀한 기울기. 양수면 같은 방향(전염), 음수면 반대 방향.
- 상대 회사가 창이 겹치게 자기 실적을 발표한 이벤트-쌍은 뺀다(상대의 반응이 자기 뉴스로 오염됨).
- 평소 날 비교: 발표일의 발표 회사 CAR에는 뉴스 말고도 그날의 업종 공통 움직임이 섞인다. 같은 회사·같은
  상대를 발표와 무관한 날(발표일 ±10·±20거래일, 가짜 발표일)에 재면 평소 동조성의 기울기가 나온다.
  발표일에 늘어난 공분산을 늘어난 분산으로 나눈 값((cov_발표 − cov_평소)/(var_발표 − var_평소))을
  '평소 동조성을 뺀 몫'으로 함께 싣는다(발표 뉴스에 대한 반응이 일정하다고 볼 때의 추정치).

한 회사가 여러 이벤트·쌍에 나오므로 구간은 research.py와 같은 회사 단위 재표집이다: (발표 회사 i,
상대 j) 이벤트-쌍마다 w_i·w_j를 곱한다. 계산은 쌍마다 이벤트를 모은 합(모멘트) 행렬로 하고, 쌍 표시가
같은 쌍끼리 묶은 칸(cell)별 합에서 묶음 기울기와 회귀를 바로 푼다. 그래서 재표집마다 이벤트-쌍 목록
(약 80만 개)을 다시 만들지 않는다.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..earnings import day_zero, load_earnings_filings, session
from ..returns import EQUAL_WEIGHT, equal_weight_market, load_prices, residual_returns
from . import research

log = logging.getLogger(__name__)

OWN_WINDOW = (0, 1)
"""상대 회사 자신의 실적 반응 창. 이 창이 분석 창과 겹치면 그 이벤트-쌍을 뺀다."""
TYPICAL_EVENTS = 4
"""회사당 한 해 분기 실적 발표 수. 이보다 많은 회사는 실적 예고 등을 Item 2.02로 더 낸 회사다."""


@dataclass(frozen=True)
class Spec:
    """이벤트-쌍 하나의 뉴스(발표 회사 CAR 창 x)와 반응(상대 CAR 창 y), 그리고 표본 조건."""

    x: tuple[int, int]
    y: tuple[int, int]
    sign: int = 0
    """+1 좋은 소식(발표 회사 CAR > 0)만, −1 나쁜 소식만, 0 모두."""
    winsor: float = 0.0
    """발표 회사 CAR과 상대 CAR을 양쪽 이 비율에서 자른다(극단값 제한). 0이면 그대로."""
    solo: bool = False
    """발표 회사와 같은 GICS 서브산업의 다른 회사가 창이 겹치게 발표한 이벤트를 뺀다."""
    few: bool = False
    """한 해 이벤트가 TYPICAL_EVENTS건 이하인 발표 회사만 (실적 예고 등이 섞인 회사를 뺌)."""
    min_abs: float = 0.0
    """발표 회사 |CAR|이 이 값 이상인 이벤트만 (큰 소식)."""
    placebo: tuple[int, ...] = ()
    """비어 있지 않으면 실제 발표일 대신 이만큼 옮긴 가짜 발표일(거래일)을 쓴다."""


PLACEBO_SHIFTS = (-20, -10, 10, 20)
SPECS = {
    "d01": Spec((0, 1), (0, 1)),
    "placebo": Spec((0, 1), (0, 1), placebo=PLACEBO_SHIFTS),
    "d11": Spec((-1, 1), (-1, 1)),
    "d02": Spec((0, 2), (0, 2)),
    "wins": Spec((0, 1), (0, 1), winsor=0.01),
    "solo": Spec((0, 1), (0, 1), solo=True),
    "few": Spec((0, 1), (0, 1), few=True),
    "big": Spec((0, 1), (0, 1), min_abs=0.05),
    "good": Spec((0, 1), (0, 1), sign=+1),
    "bad": Spec((0, 1), (0, 1), sign=-1),
    "drift": Spec((0, 1), (2, 5)),
}
MAIN = "d01"
ROBUST = {
    "d11": "[−1, +1]",
    "d02": "[0, +2]",
    "wins": "극단값 1% 제한",
    "solo": "같은 업종 동시 발표 제외",
    "few": f"이벤트 {TYPICAL_EVENTS}건 이하 회사만",
}
"""주 창 [0, +1]과 결론이 같은지 보는 변형."""
PROFILE_DAYS = tuple(range(-3, 6))
"""발표 회사 자신의 날짜별 반응을 보는 날 (0일 확인용)."""

GROUPS = {
    **{k: v for k, v in research.GROUPS.items() if k not in ("same_sub", "all")},
    "same_sub": "같은 서브산업 전체",
    "same_sub_unrelated": "같은 서브산업, 관계·언급 없음",
    "all": "모든 쌍 (계산상 0에 가까움)",
}
"""표에 싣는 쌍 묶음 (research.GROUPS에 '같은 서브산업, 관계·언급 없음'을 더함). {k}는 relations.top_k."""
POSITIONS = {
    "same_sub": ("같은 서브산업", [("same_sub", True)]),
    "same_sector": ("같은 섹터, 다른 서브산업", [("same_sector", True), ("same_sub", False)]),
    "other_sector": ("다른 섹터", [("gics_known", True), ("same_sector", False)]),
}
POSITION_ROWS = {
    "competitor": "경쟁",
    "business": "공급·협력",
    "mention_only": "회사 언급, 관계 아님",
    "unrelated": "관계·언급 없음",
}
"""GICS 위치별 표의 행. 관계 묶음은 GICS 위치 구성이 달라, 같은 위치끼리 비교해야 공정하다."""
BASE = ("competitor", "business", "equity", "uncertain", "mention_only", "name_only",
        "same_sub", "same_sector")  # fmt: skip
TEXT = (*(f"sim_{b:g}" for b in research.SIM_BINS), "top20")
MODELS = {"gics": BASE, "text": BASE + TEXT}
MODEL_LABELS = {"gics": "(1) GICS", "text": "(2) + 텍스트 유사도"}
COEFS = {
    "x": "기준 쌍의 기울기 (아래 설명)",
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
"""회귀표에 싣는 반응 계수. x는 기준 쌍의 기울기, 나머지는 그 표시가 있을 때 기울기에 더해지는 몫."""
MOMENTS = ("n", "x", "xx", "y", "xy")


def group_defs() -> dict[str, list[tuple[str, bool]]]:
    """묶음 이름 → 조건 [(쌍 표시, 있어야 하는가)]. 모든 묶음은 칸(cell)을 나누는 표시로 정해진다."""
    defs = {k: [(k, True)] for k in GROUPS if k not in ("same_sub_unrelated", "all")}
    defs["same_sub_unrelated"] = [("same_sub", True), ("unrelated", True)]
    defs["all"] = []
    for row in POSITION_ROWS:
        for pos, (_, cond) in POSITIONS.items():
            defs[f"{row}@{pos}"] = [(row, True), *cond]
    return defs


CONTRASTS = {
    "competitor_vs_mention": ("competitor", "mention_only"),
    "business_vs_mention": ("business", "mention_only"),
    **{
        f"{r}_vs_unrelated@{pos}": (f"{r}@{pos}", f"unrelated@{pos}")
        for r in ("competitor", "business")
        for pos in POSITIONS
    },
}
"""같은 재표집 표본으로 짝지어 재는 묶음 차이. 관계 묶음은 같은 GICS 위치의 '관계·언급 없음'과 비교한다."""


@dataclass
class Market:
    """시장 모형의 재료 (거래일 × 회사, 회사 순서는 pair.firms). 상대 회사의 시장에서 발표 회사를 빼는 데 쓴다."""

    returns: np.ndarray
    """일별 수익률."""
    market: np.ndarray
    """회사마다 자기를 뺀 동일가중 시장 수익률 (잔차를 낸 그 시장). 수익률 이력이 부족한 회사는 NaN."""
    others: np.ndarray
    """그날 그 회사의 시장에 들어간 회사 수."""
    beta: np.ndarray
    """(회사,) 시장 베타."""
    in_market: np.ndarray
    """(회사,) 다른 회사의 시장 평균에 들어가는가 (수익률 이력이 충분함)."""


@dataclass
class EventData:
    pair: research.PairData
    resid: np.ndarray
    """(거래일, 회사) 시장 모형 잔차. 회사 순서는 pair.firms."""
    calendar: pd.DatetimeIndex
    """resid 행의 날짜."""
    events: pd.DataFrame
    """firm(행렬 위치), day(resid 행), accepted(UTC), session, accession_number, ticker."""
    info: dict
    market: Market | None = None
    """없으면 상대 회사의 시장에서 발표 회사를 빼지 않는다(시험용)."""


# ---------------------------------------------------------------- 계산 (배열만 받는다)


def window_car(resid: np.ndarray, days: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """(이벤트, 회사): 이벤트마다 [0일+lo, 0일+hi] 잔차의 합. 창이 기간을 벗어나거나 빠진 날이 있으면 NaN."""
    t = resid.shape[0]
    out = np.full((len(days), resid.shape[1]), np.nan)
    for k, d in enumerate(days):
        if d + lo >= 0 and d + hi < t:
            out[k] = resid[d + lo : d + hi + 1].sum(axis=0)
    return out


def announcer_out(m: Market, firm: np.ndarray, days: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """(이벤트, 회사): 상대 j의 시장에서 발표 회사 i를 뺐을 때 j의 CAR에 더해지는 몫.

    j의 잔차는 e_j = r_j − α_j − β_j·m_j이고, m_j(j를 뺀 동일가중 평균)에는 i도 1/K 비중으로 들어 있다.
    그래서 i가 x만큼 움직이면 e_j가 기계적으로 β_j·x/K만큼 반대로 움직인다(약 −0.002·x). i를 뺀 시장
    m_j' = (K·m_j − r_i)/(K − 1)로 바꾸면 e_j' = e_j + β_j·(r_i − m_j)/(K − 1)이다(α, β는 그대로 둔다).
    """
    t, n = m.returns.shape
    out = np.zeros((len(days), n))
    for k, (i, d) in enumerate(zip(firm, days, strict=True)):
        if not m.in_market[i] or d + lo < 0 or d + hi >= t:
            continue
        rows = slice(d + lo, d + hi + 1)
        r_i = m.returns[rows, i][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            shift = (r_i - m.market[rows]) / (m.others[rows] - 1)
        shift = np.where(np.isfinite(r_i) & (m.others[rows] > 1), shift, 0.0)
        out[k] = m.beta * shift.sum(axis=0)
    return out


def winsorize(values: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
    """mask 안 값의 양쪽 q 비율에서 자른다 (mask 밖은 그대로)."""
    if not q or not mask.any():
        return values
    lo, hi = np.quantile(values[mask], [q, 1 - q])
    return np.where(mask, np.clip(values, lo, hi), values)


def own_event_overlap(
    event_days: np.ndarray, days: np.ndarray, lo: int, hi: int, own: tuple[int, int] = OWN_WINDOW
) -> np.ndarray:
    """(이벤트, 회사): 그 회사 자신의 실적 반응 창 [d+own0, d+own1]이 [0일+lo, 0일+hi]와 겹치는가.

    event_days: (회사, 거래일) bool, 그날이 그 회사 실적 발표의 0일인가."""
    n, t = event_days.shape
    counts = np.concatenate([np.zeros((n, 1)), np.cumsum(event_days, axis=1)], axis=1)
    first = np.clip(days + lo - own[1], 0, t)  # 겹치는 d의 범위: [0일+lo−own1, 0일+hi−own0]
    last = np.clip(days + hi - own[0] + 1, 0, t)
    return (counts[:, last] - counts[:, first]).T > 0


def event_moments(
    x: np.ndarray, firm: np.ndarray, y: np.ndarray, valid: np.ndarray, n: int
) -> dict:
    """쌍(발표 회사 i, 상대 j)마다 쓸 수 있는 이벤트를 모은 합: n(개수), x, xx, y, xy. 각 (n, n)."""
    a = np.zeros((len(x), n))
    a[np.arange(len(x)), firm] = 1.0
    v = valid.astype(np.float64)
    y0 = np.where(valid, y, 0.0)
    xv = np.where(np.isfinite(x), x, 0.0)[:, None]
    return {
        "n": a.T @ v,
        "x": a.T @ (v * xv),
        "xx": a.T @ (v * xv**2),
        "y": a.T @ y0,
        "xy": a.T @ (y0 * xv),
    }


@dataclass
class Cells:
    """분석하는 쌍(발표 회사 ii, 상대 jj)과, 쌍 표시가 모두 같은 쌍끼리 묶은 칸."""

    ii: np.ndarray
    jj: np.ndarray
    cell: np.ndarray
    """쌍마다 칸 번호."""
    patterns: np.ndarray
    """(칸, 표시) 0/1."""
    names: list[str]


def make_cells(indicators: dict[str, np.ndarray], mask: np.ndarray) -> Cells:
    ii, jj = np.nonzero(mask)
    names = list(indicators)
    stack = np.stack([indicators[k][ii, jj] for k in names], axis=1).astype(np.int8)
    patterns, cell = np.unique(stack, axis=0, return_inverse=True)
    return Cells(ii, jj, cell.reshape(-1), patterns.astype(np.float64), names)


def cell_sums(cells: Cells, values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """(모멘트, 칸): 칸마다 쌍의 모멘트 합. values: (모멘트, 쌍) = 모멘트 행렬을 cells 쌍에서 뽑은 것.
    weights(회사별 뽑힌 횟수)가 있으면 쌍 (i, j)에 w_i·w_j를 곱한다."""
    c = len(cells.patterns)
    omega = 1.0 if weights is None else weights[cells.ii] * weights[cells.jj]
    return np.stack([np.bincount(cells.cell, weights=omega * v, minlength=c) for v in values])


def slope(s: np.ndarray) -> float:
    """모멘트 합 (n, x, xx, y, xy) → y를 x에 회귀한 기울기 (절편 포함). 쓸 수 없으면 NaN."""
    n, sx, sxx, sy, sxy = s
    if n <= 1:
        return math.nan
    var = sxx - sx * sx / n
    return float((sxy - sx * sy / n) / var) if var > 0 else math.nan


def transfer(event: np.ndarray, normal: np.ndarray) -> float:
    """발표일에 늘어난 공분산 ÷ 늘어난 분산 (평소 동조성을 뺀 반응). 두 모멘트 합은 같은 묶음의 것."""
    if event[0] <= 1 or normal[0] <= 1:
        return math.nan

    def cov_var(s: np.ndarray) -> tuple[float, float]:
        n, sx, sxx, sy, sxy = s
        return sxy / n - (sx / n) * (sy / n), sxx / n - (sx / n) ** 2

    ce, ve = cov_var(event)
    cn, vn = cov_var(normal)
    return float((ce - cn) / (ve - vn)) if ve > vn else math.nan


def group_membership(cells: Cells, defs: dict[str, list[tuple[str, bool]]]) -> np.ndarray:
    """(칸, 묶음) 0/1. 묶음마다 조건(표시가 있음/없음)을 모두 만족하는 칸."""
    cols = []
    for cond in defs.values():
        col = np.ones(len(cells.patterns))
        for name, present in cond:
            v = cells.patterns[:, cells.names.index(name)]
            col = col * (v if present else 1 - v)
        cols.append(col)
    return np.stack(cols, axis=1)


def spill_ols(sums: np.ndarray, patterns: np.ndarray, names: list[str]) -> dict[str, float | None]:
    """y = α + Σγ_k·D_k + (β + Σβ_k·D_k)·x 를 칸별 모멘트 합으로 푼다.

    D_k: 쌍 표시(patterns의 열, names 순서). 돌려주는 값: 'x' = β(표시가 모두 없는 쌍의 기울기),
    각 이름 = β_k(그 표시가 있을 때 기울기에 더해지는 몫). 쓰이는 이벤트-쌍이 없는 표시는 None.
    칸 c의 설계 행은 [1, D_c] ⊗ [1, x] 꼴이라 정규방정식이 칸별 합의 크로네커 곱의 합이 된다.
    """
    n, sx, sxx, sy, sxy = sums
    has = (n @ patterns) > 0
    keep = [k for k, h in enumerate(has) if h]
    q = np.column_stack([np.ones(len(n)), patterns[:, keep]])
    k = q.shape[1]
    m2 = np.array([[n, sx], [sx, sxx]])  # (2, 2, 칸)
    xtx = np.einsum("abc,cl,cm->albm", m2, q, q).reshape(2 * k, 2 * k)
    xty = np.einsum("ac,cl->al", np.array([sy, sxy]), q).reshape(2 * k)
    coef, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    out: dict[str, float | None] = {"x": float(coef[k])}
    for pos, idx in enumerate(keep):
        out[names[idx]] = float(coef[k + 1 + pos])
    return out | {names[i]: None for i in range(len(names)) if not has[i]}


def _interval(draws) -> tuple[float | None, float | None]:
    values = np.array([d for d in draws if d is not None and np.isfinite(d)])
    if not len(values):
        return None, None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _none(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


@dataclass
class SpecFit:
    """한 Spec의 결과(result, 보고서·JSON용)와 재표집마다의 묶음 모멘트 합(boot, 다른 Spec과 짝지을 때)."""

    result: dict
    point: np.ndarray
    """(모멘트, 묶음) 묶음별 모멘트 합."""
    boot: np.ndarray
    """(재표집, 모멘트, 묶음)."""


def analyze_spec(
    moments: dict, cells: Cells, defs: dict, weights: np.ndarray, contrasts: dict
) -> SpecFit:
    """한 Spec의 묶음별 기울기·짝지은 차이·회귀와 회사 단위 재표집 구간."""
    names = list(defs)
    values = np.stack([moments[m][cells.ii, cells.jj] for m in MOMENTS])
    member = group_membership(cells, defs)
    columns = {m: [cells.names.index(c) for c in cols] for m, cols in MODELS.items()}

    def estimate(sums: np.ndarray) -> tuple[np.ndarray, dict]:
        reg = {
            m: spill_ols(sums, cells.patterns[:, cols], [cells.names[c] for c in cols])
            for m, cols in columns.items()
        }
        return sums @ member, reg

    point_sums = cell_sums(cells, values)
    point, reg = estimate(point_sums)
    boots = [estimate(cell_sums(cells, values, w)) for w in weights]
    boot = np.stack([b[0] for b in boots]) if boots else np.zeros((0, *point.shape))
    slopes = {g: slope(point[:, k]) for k, g in enumerate(names)}
    boot_slopes = {g: [slope(b[:, k]) for b in boot] for k, g in enumerate(names)}

    pairs_any = (moments["n"] > 0) | (moments["n"] > 0).T
    out_groups = {}
    for k, name in enumerate(names):
        lo, hi = _interval(boot_slopes[name])
        in_group = np.zeros_like(pairs_any)
        in_group[cells.ii, cells.jj] = member[cells.cell, k] > 0
        out_groups[name] = {
            "slope": _none(slopes[name]),
            "lo": lo,
            "hi": hi,
            "n_event_pairs": int(point[0, k]),
            "n_pairs": int((pairs_any & in_group).sum()) // 2,
            "n_announcers": int(((moments["n"] * in_group).sum(axis=1) > 0).sum()),
        }
    out_contrasts = {}
    for c, (a, b) in contrasts.items():
        draws = [x - y for x, y in zip(boot_slopes[a], boot_slopes[b], strict=True)]
        lo, hi = _interval(draws)
        diff = slopes[a] - slopes[b]
        out_contrasts[c] = {"diff": _none(diff), "lo": lo, "hi": hi}
    out_reg = {}
    for m in MODELS:
        coefs = {}
        for name, value in reg[m].items():
            draws = [b[1][m][name] for b in boots if b[1][m][name] is not None]
            lo, hi = _interval(draws) if value is not None else (None, None)
            coefs[name] = {"coef": value, "lo": lo, "hi": hi, "n_boot": len(draws)}
        out_reg[m] = {"coefs": coefs, "n_event_pairs": int(point_sums[0].sum())}
    result = {"groups": out_groups, "contrasts": out_contrasts, "regression": out_reg}
    return SpecFit(result, point, boot)


def paired(a: SpecFit, b: SpecFit, names: list[str], fn) -> dict:
    """두 Spec의 같은 묶음 모멘트 합에 fn(a 합, b 합)을 적용한 값과, 같은 재표집끼리 짝지은 95% 구간."""
    out = {}
    for k, g in enumerate(names):
        value = _none(fn(a.point[:, k], b.point[:, k]))
        draws = [fn(x[:, k], y[:, k]) for x, y in zip(a.boot, b.boot, strict=True)]
        lo, hi = _interval(draws) if value is not None else (None, None)
        out[g] = {"diff": value, "lo": lo, "hi": hi}
    return out


def lead_lag(resid: np.ndarray) -> np.ndarray:
    """(n, n): L[i, j] = corr(e_i,t, e_j,t+1). 하루 뒤 j의 잔차가 오늘 i의 잔차를 따라가는가."""
    a, b = resid[:-1], resid[1:]
    ma, mb = np.isfinite(a), np.isfinite(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        za = np.where(ma, (a - np.nanmean(a, 0)) / np.nanstd(a, 0), 0.0)
        zb = np.where(mb, (b - np.nanmean(b, 0)) / np.nanstd(b, 0), 0.0)
        joint = ma.astype(np.float64).T @ mb.astype(np.float64)
        out = (np.nan_to_num(za).T @ np.nan_to_num(zb)) / joint
    out[joint < 2] = np.nan
    np.fill_diagonal(out, np.nan)
    return out


# ---------------------------------------------------------------- 분석


def indicators(pair: research.PairData) -> dict[str, np.ndarray]:
    """칸을 나눌 쌍 표시 전체. 묶음(group_defs)과 회귀 변수는 모두 이 표시로 정해진다."""
    p, usable = pair.pairs, pair.usable
    groups = research.pair_groups(p, usable)
    related = p["competitor"] | p["business"] | p["equity"] | p["uncertain"] | p["mentioned"]
    marks = {k: groups[k] for k in research.GROUPS if k != "all"}
    marks["unrelated"] = ~related
    marks |= {k: p[k] for k in ("same_sector", "gics_known", "top20")}
    marks |= research.similarity_bins(research.pair_percentile(pair.sim, usable))
    return {k: v & usable for k, v in marks.items()}


def group_matrices(marks: dict[str, np.ndarray], defs: dict, usable: np.ndarray) -> dict:
    """묶음 정의 → 쌍 표시 행렬 (lead_lag·예시처럼 행렬로 계산하는 곳에서 쓴다)."""
    out = {}
    for name, cond in defs.items():
        m = usable.copy()
        for mark, present in cond:
            m &= marks[mark] if present else ~marks[mark]
        out[name] = m
    return out


def event_arrays(
    data: EventData, spec: Spec, *, drop_overlap: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(발표 회사 CAR x, 발표 회사 위치, 상대 CAR y, 쓸 수 있는 이벤트-쌍) — 이벤트 순서.

    spec.placebo가 있으면 실제 발표일을 그만큼 옮긴 가짜 발표일마다 한 줄씩이다. 가짜 발표일의 창이
    발표 회사 자신의 실적 창과 겹치면 뺀다."""
    ev = data.events
    n, t = len(data.pair.firms), data.resid.shape[0]
    firm, days = ev["firm"].to_numpy(), ev["day"].to_numpy()
    own = np.zeros((n, t), dtype=bool)
    own[firm, days] = True
    if spec.placebo:
        firm = np.tile(firm, len(spec.placebo))
        days = np.concatenate([days + s for s in spec.placebo])
        inside = (days >= 0) & (days < t)
        firm, days = firm[inside], days[inside]
    rows = np.arange(len(firm))
    x = window_car(data.resid, days, *spec.x)[rows, firm]
    y = window_car(data.resid, days, *spec.y)
    if data.market is not None:
        y = y + announcer_out(data.market, firm, days, *spec.y)
    lo, hi = min(spec.x[0], spec.y[0]), max(spec.x[1], spec.y[1])
    overlap = own_event_overlap(own, days, lo, hi)
    valid = np.isfinite(y) & np.isfinite(x)[:, None] & data.pair.usable[firm]
    if drop_overlap:
        valid &= ~overlap
    valid[rows, firm] = False
    if spec.placebo:
        valid &= ~overlap[rows, firm][:, None]
    if spec.sign:
        valid &= (np.sign(x) == spec.sign)[:, None]
    if spec.min_abs:
        valid &= (np.abs(x) >= spec.min_abs)[:, None]
    if spec.few:
        counts = np.bincount(ev["firm"].to_numpy(), minlength=n)
        valid &= (counts[firm] <= TYPICAL_EVENTS)[:, None]
    if spec.solo:
        peers = data.pair.pairs["same_sub"] & ~np.eye(n, dtype=bool)
        valid &= ~(overlap & peers[firm]).any(axis=1)[:, None]
    if spec.winsor:
        used = valid.any(axis=1)
        x = winsorize(x, used, spec.winsor)
        y = winsorize(y, valid, spec.winsor)
    return x, firm, y, valid


def analyze(data: EventData, *, n_boot: int = 1000, seed: int = 0) -> dict:
    pair = data.pair
    n = len(pair.firms)
    marks = indicators(pair)
    defs = group_defs()
    names = list(defs)
    cells = make_cells(marks, pair.usable)
    weights = research.boot_weights(n, n_boot, seed)

    fits, specs = {}, {}
    for name, spec in SPECS.items():
        x, firm, y, valid = event_arrays(data, spec)
        fits[name] = analyze_spec(event_moments(x, firm, y, valid, n), cells, defs, weights,
                                  CONTRASTS)  # fmt: skip
        used = valid.any(axis=1)
        specs[name] = fits[name].result | {
            "n_events": int(used.sum()),
            "mean_abs_news": _none(float(np.nanmean(np.abs(x[used])))) if used.any() else None,
        }
        log.info("이벤트 창 %s 완료", name)

    return {
        "info": data.info,
        "n_boot": n_boot,
        "specs": specs,
        # 발표일 기울기 − 평소 날 기울기, 발표일에 늘어난 공분산 ÷ 늘어난 분산
        "vs_placebo": paired(fits[MAIN], fits["placebo"], names, lambda a, b: slope(a) - slope(b)),
        "transfer": paired(fits[MAIN], fits["placebo"], names, transfer),
        # 나쁜 소식 − 좋은 소식
        "asymmetry": paired(fits["bad"], fits["good"], names, lambda a, b: slope(a) - slope(b)),
        "exclusion": exclusion(data, marks, defs),
        "profile": profile(data),
        "lead_lag": research.group_means(
            lead_lag(data.resid), group_matrices(marks, defs, pair.usable), weights
        ),
        "examples": examples(data, group_matrices(marks, defs, pair.usable)),
    }


def exclusion(data: EventData, marks: dict, defs: dict) -> dict:
    """주 창에서 상대 회사의 자기 발표 때문에 빠진 이벤트-쌍 비율과, 통째로 빠진 관계 쌍."""
    spec = SPECS[MAIN]
    _, firm, _, before = event_arrays(data, spec, drop_overlap=False)
    _, _, _, after = event_arrays(data, spec)
    n = len(data.pair.firms)
    mats = group_matrices(marks, defs, data.pair.usable)
    out: dict = {"share": {}, "dropped_pairs": {}}
    count_b = np.zeros((n, n))
    count_a = np.zeros((n, n))
    np.add.at(count_b, firm, before.astype(float))
    np.add.at(count_a, firm, after.astype(float))
    for g in ("competitor", "business", "mention_only", "same_sub_unrelated", "all"):
        m = mats[g]
        total = float((count_b * m).sum())
        out["share"][g] = (total - float((count_a * m).sum())) / total if total else None
    tickers = data.pair.firms["ticker"].to_numpy()
    for g in ("competitor", "business"):
        m = np.triu(mats[g], 1)
        had = (count_b + count_b.T) > 0
        kept = (count_a + count_a.T) > 0
        i, j = np.nonzero(m & had & ~kept)
        out["dropped_pairs"][g] = [f"{tickers[a]}–{tickers[b]}" for a, b in zip(i, j, strict=True)]
    return out


def profile(data: EventData) -> dict:
    """발표 회사 자신의 날짜별 평균 |잔차| (0일이 맞게 잡혔는지 확인) + 평소 날의 평균 |잔차|."""
    ev = data.events
    firm, days = ev["firm"].to_numpy(), ev["day"].to_numpy()
    t = data.resid.shape[0]
    rows = {}
    for k in PROFILE_DAYS:
        ok = (days + k >= 0) & (days + k < t)
        vals = np.abs(data.resid[days[ok] + k, firm[ok]])
        rows[str(k)] = _none(float(np.nanmean(vals))) if ok.any() else None
    own = np.zeros(data.resid.shape, dtype=bool)
    for f, d in zip(firm, days, strict=True):
        own[max(d - 3, 0) : d + 6, f] = True
    announcers = np.zeros(data.resid.shape[1], dtype=bool)
    announcers[firm] = True
    normal = np.abs(data.resid[:, announcers][~own[:, announcers]])
    return {"by_day": rows, "normal": _none(float(np.nanmean(normal)))}


PER_EVENT = 2
"""예시에서 이벤트 하나당 싣는 상대 회사 수."""


def examples(data: EventData, groups: dict[str, np.ndarray], n: int = 10) -> dict:
    """관계 쌍의 이벤트 가운데 발표 회사 반응이 가장 컸던 것 (주 창 [0, +1])."""
    x, firm, y, valid = event_arrays(data, SPECS[MAIN])
    firms = data.pair.firms
    dates = data.calendar[data.events["day"].to_numpy()]
    out = {}
    for rel in ("competitor", "business"):
        rows, per_event = [], {}
        e_idx, j_idx = np.nonzero(valid & groups[rel][firm])
        # 발표 회사 반응이 큰 이벤트부터, 이벤트마다 상대 반응이 큰 2곳까지
        order = np.lexsort((-np.abs(y[e_idx, j_idx]), -np.abs(x[e_idx])))
        for e, j in zip(e_idx[order], j_idx[order], strict=True):
            if per_event.get(e, 0) == PER_EVENT:
                continue
            per_event[e] = per_event.get(e, 0) + 1
            rows.append(
                {
                    "date": str(dates[e].date()),
                    "announcer": research._firm(firms.iloc[firm[e]]),
                    "news": float(x[e]),
                    "partner": research._firm(firms.iloc[j]),
                    "response": float(y[e, j]),
                }
            )
            if len(rows) == n:
                break
        out[rel] = rows
    return out


# ---------------------------------------------------------------- 데이터 읽기


def load_event_data(cfg: Config, *, refresh: bool = False) -> EventData:
    pair = research.load_pair_data(cfg)
    rc = cfg.evaluation.returns
    tickers = pair.firms["ticker"].tolist()
    prices = load_prices(
        tickers, rc.market, rc.start, rc.end, cfg.run_dir / f"prices_{rc.start}_{rc.end}.parquet"
    )
    stock, resid, enough = residual_returns(prices, tickers, rc.market, rc.min_obs)
    filings = load_earnings_filings(
        pair.firms[["cik", "ticker"]], rc.start, rc.end,
        cfg.run_dir / f"earnings_8k_{rc.start}_{rc.end}.parquet", refresh=refresh,
    )  # fmt: skip
    data = build_event_data(pair, resid, pd.DatetimeIndex(prices.index), filings)
    if rc.market == EQUAL_WEIGHT:
        data.market = market_parts(stock, resid, enough)
    return data


def market_parts(stock: pd.DataFrame, resid: pd.DataFrame, enough: np.ndarray) -> Market:
    """residual_returns와 같은 동일가중 시장(자기 제외)과 회사별 베타."""
    usable = stock.loc[:, enough]
    market = equal_weight_market(usable).reindex(columns=stock.columns).to_numpy()
    present = usable.notna()
    others = (
        present.sum(axis=1).to_numpy()[:, None]
        - present.reindex(columns=stock.columns, fill_value=False).to_numpy()
    )
    r, e = stock.to_numpy(), resid.to_numpy()
    beta = np.full(r.shape[1], np.nan)
    for j in np.flatnonzero(enough):
        ok = np.isfinite(r[:, j]) & np.isfinite(market[:, j]) & np.isfinite(e[:, j])
        m = market[ok, j]
        beta[j] = np.cov(r[ok, j], m, bias=True)[0, 1] / np.var(m)
    return Market(r, market, others.astype(np.float64), beta, np.asarray(enough, dtype=bool))


def build_event_data(
    pair: research.PairData,
    resid: pd.DataFrame,
    price_days: pd.DatetimeIndex,
    filings: pd.DataFrame,
) -> EventData:
    """8-K 목록 → 이벤트 표. price_days는 가격이 있는 거래일(resid는 그 둘째 날부터)."""
    pos_of = {int(c): i for i, c in enumerate(pair.firms["cik"])}
    tickers = pair.firms["ticker"].to_numpy()
    ev = filings[filings["status"] == "event"].copy()
    ev["firm"] = ev["cik"].map(pos_of)
    ev = ev[ev["firm"].notna()].copy()
    ev["firm"] = ev["firm"].astype(int)
    ev["session"] = session(ev["accepted"]).to_numpy()
    report = ev["report_date"] if "report_date" in ev else None
    pos, early = day_zero(ev["accepted"], price_days, report)
    ev["day"] = pos - 1  # resid 행 = 가격 위치 − 1 (첫날은 수익률이 없음)
    ev["by_report_date"] = early
    n_filings = len(ev)
    outside = int((pos < 0).sum())
    ev = ev[pos >= 0]
    # 같은 회사가 같은 0일에 여러 건(보도자료와 정정 등)을 냈으면 가장 먼저 낸 것 하나로 센다
    ev = ev.sort_values(["firm", "day", "accepted"]).drop_duplicates(["firm", "day"])
    enough = np.isfinite(resid.to_numpy()).sum(axis=0) > 0
    no_returns = int((~enough[ev["firm"].to_numpy()]).sum())
    ev = ev[enough[ev["firm"].to_numpy()]].reset_index(drop=True)
    ev["ticker"] = tickers[ev["firm"].to_numpy()]
    fetched = filings.groupby("cik")["status"].agg(
        lambda s: "error" if (s == "error").any() else "ok"
    )
    per_firm = ev.groupby("firm").size().reindex(range(len(pair.firms)), fill_value=0)
    info = pair.info | {
        "events": {
            "firms": len(pair.firms),
            "firms_failed": int((fetched == "error").sum()),
            "filings": n_filings,
            "outside_period": outside,
            "duplicates": n_filings - outside - no_returns - len(ev),
            "no_returns": no_returns,
            "events": len(ev),
            "by_report_date": int(ev["by_report_date"].sum()),
            "announcers": int((per_firm > 0).sum()),
            "per_firm": {str(k): int(v) for k, v in per_firm.value_counts().sort_index().items()},
            "without_events": sorted(str(t) for t in tickers[(per_firm == 0).to_numpy()]),
            "few_events": sorted(
                str(t) for t in tickers[((per_firm > 0) & (per_firm < TYPICAL_EVENTS)).to_numpy()]
            ),
            "many_events": int((per_firm > TYPICAL_EVENTS).sum()),
            "sessions": {str(k): int(v) for k, v in ev["session"].value_counts().items()},
        }
    }
    return EventData(pair, resid.to_numpy(), pd.DatetimeIndex(resid.index), ev, info)


# ---------------------------------------------------------------- 보고서


def _signed(v: float) -> str:
    """부호를 붙인 소수 셋째 자리. 반올림해 0이면 부호 없이 '0.000' (−0.000을 쓰지 않음)."""
    text = f"{v:+.3f}"
    return "0.000" if text in ("+0.000", "-0.000") else text


def _with_ci(value: float | None, e: dict, *, points_only: bool = False) -> str:
    """값 [구간]. 구간이 0을 포함하지 않으면 굵게. 구간이 없거나 points_only면 값만."""
    if value is None:
        return "–"
    if points_only or e.get("lo") is None:
        return _signed(value)
    text = f"{_signed(value)} [{_signed(e['lo'])}, {_signed(e['hi'])}]"
    return f"**{text}**" if research._significant(e) else text


def _slope(e: dict) -> str:
    return _with_ci(e.get("slope"), e, points_only=e.get("n_pairs", 0) < research.MIN_CI_PAIRS)


def _mean(e: dict) -> str:
    return _with_ci(e.get("mean"), e, points_only=e.get("n_pairs", 0) < research.MIN_CI_PAIRS)


def _diff(e: dict | None) -> str:
    return "–" if not e else _with_ci(e.get("diff"), e)


def _coef(e: dict | None) -> str:
    return "–" if not research._estimated(e) else _with_ci(e["coef"], e)


def _cell(e: dict) -> str:
    """'기울기 [구간] · 쌍 수'"""
    return "–" if e.get("slope") is None else f"{_slope(e)} · {e['n_pairs']:,}"


def _label(key: str, top_k: int) -> str:
    return GROUPS[key].format(k=top_k)


def _pct(v: float | None, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}%}"


def _sig(e: dict | None) -> bool:
    return bool(e) and e.get("lo") is not None and research._significant(e)


TABLE_GROUPS = list(GROUPS)


def format_report(result: dict) -> str:
    info, specs = result["info"], result["specs"]
    ev = info["events"]
    main = specs[MAIN]
    g = main["groups"]
    prof = result["profile"]
    sessions = ev["sessions"]
    n_ev = max(ev["events"], 1)
    per = g["all"]["n_event_pairs"] / max(g["all"]["n_pairs"], 1)
    lines = [
        f"# {info['config']} 실적 발표 때의 주가 전이 (이벤트 스터디)",
        "",
        f"- 생성: {datetime.now().isoformat(timespec='seconds')}",
        f"- 관계: {info['filings_year']}년 제출 10-K에서 판정한 관계입니다(`graph.db` "
        f"{info['graph_created']}, 판정 `{info['judge_model']}` 질문 {info['question_version']}). "
        "관계도에 보이는 관계(모델 채택 + 검수로 확인)만 씁니다. 쌍 묶음의 정의는 2-3 리포트와 같고, "
        "'같은 서브산업, 관계·언급 없음'을 더했습니다. 표의 '쌍'은 쓸 수 있는 이벤트가 하나라도 있는 쌍만 "
        "세므로 2-3보다 조금 적습니다.",
        f"- 이벤트: {info['start']} ~ {info['end']}의 8-K 가운데 Item 2.02(실적 발표)가 든 것입니다. "
        "접수 시각이 뉴욕 장 마감(16:00, 조기 폐장일은 13:00) 이후면 다음 거래일, 그 전이면 그날(휴장일이면 "
        "다음 거래일)을 0일로 둡니다. 8-K의 보고일이 휴장일이고 접수일보다 1~4일 앞서면(토요일에 실적을 "
        "내는 버크셔 해서웨이 등) 보고일 뒤 첫 거래일을 0일로 둡니다.",
        "- 비정상 수익률: 2-3과 같은 시장 모형 잔차입니다(동일가중 시장, 자기 제외, 수익률 기간 전체로 "
        "추정). CAR은 창 안의 잔차 합입니다. 상대 회사의 시장 평균에는 발표 회사도 들어 있어, 발표 회사가 "
        "크게 움직이면 상대의 잔차가 기계적으로 반대로 움직입니다(약 −0.002배). 그래서 상대 회사 CAR은 "
        "시장 평균에서 발표 회사를 뺀 값으로 다시 계산했습니다.",
        "- 반응 계수: 상대 회사 CAR을 발표 회사 CAR에 회귀한 기울기입니다. **+0.05면 발표 회사 주가가 "
        "10% 움직일 때 상대가 평균 0.5% 같은 방향으로** 움직였다는 뜻입니다. 음수면 평균적인 종목보다 "
        "반대로 움직였다는 뜻입니다(경쟁 효과일 수 있음).",
        "- 상대 회사가 창이 겹치게(자기 반응 창 [0, +1] 기준) 자기 실적을 발표한 이벤트-쌍은 뺍니다.",
        f"- 괄호 안 구간은 회사 단위 부트스트랩 95% 구간입니다({result['n_boot']}회). 발표 회사와 상대 "
        "회사를 함께 복원추출합니다(2-3과 같은 방식). 구간이 0을 포함하지 않으면 **굵게** 표시합니다. 구간 "
        "끝이 0.000으로 반올림되면 0을 포함한 것으로 봅니다.",
        "",
        "## 1. 요약",
        "",
        *_summary(result),
        "",
        "## 2. 이벤트와 0일 확인",
        "",
        f"- 8-K(Item 2.02) {ev['filings']:,}건 → 이벤트 {ev['events']:,}건, 발표 회사 "
        f"{ev['announcers']}곳(분석 대상 {ev['firms']}곳 중). 기간 밖 {ev['outside_period']}건, 같은 0일 "
        f"중복 {ev['duplicates']}건, 주가 부족 {ev['no_returns']}건을 뺐습니다. 보고일로 0일을 앞당긴 "
        f"이벤트는 {ev['by_report_date']}건입니다(휴장일 보고일).",
        "- 회사당 이벤트 수: "
        + ", ".join(f"{k}건 {v}곳" for k, v in ev["per_firm"].items() if k != "0")
        + ".",
        *_firm_lists(ev),
        f"- Item 2.02에는 분기 실적 외에 잠정 실적·가이던스·인도량 같은 운영 지표 공시도 들어갑니다"
        f"(예: Tesla 분기 인도량). 이벤트가 {TYPICAL_EVENTS + 1}건 이상인 회사 {ev['many_events']}곳은 "
        "대개 이런 공시가 더해진 경우입니다. 회사 고유 뉴스라 분석에 남기고, 5장에서 이 회사들의 발표를 뺀 결과를 "
        "함께 봅니다.",
        f"- 접수 시각: 장 전 {sessions.get('pre', 0) / n_ev:.0%}, 장중 "
        f"{sessions.get('intraday', 0) / n_ev:.0%}, 장 마감 뒤 {sessions.get('after', 0) / n_ev:.0%}.",
        f"- 발표 회사의 평균 |CAR[0, +1]|은 {_pct(main['mean_abs_news'])}입니다.",
        "",
        "발표 회사 자신의 날짜별 평균 |비정상 수익률|입니다. 0일이 가장 크고 +1일이 그다음이면 0일을 맞게 "
        "잡은 것입니다.",
        "",
    ]
    days = list(prof["by_day"])
    lines += research._table(
        ["", *(f"{int(d):+d}일" if int(d) else "0일" for d in days), "평소"],
        [["평균 \\|잔차\\|", *(_pct(prof["by_day"][d]) for d in days), _pct(prof["normal"])]],
    )
    lines += _section_groups(result, per)
    lines += _section_positions(result)
    lines += _section_robust(result)
    lines += _section_news(result)
    lines += _section_regression(result)
    lines += _section_lag(result)
    lines += _section_examples(result)
    lines += _section_caveats()
    return "\n".join(lines)


def _firm_lists(ev: dict) -> list[str]:
    out = []
    if ev.get("without_events"):
        out.append(
            f"- 이벤트가 없는 회사: {', '.join(ev['without_events'])}. "
            f"{TYPICAL_EVENTS}건보다 적은 회사: {', '.join(ev.get('few_events') or ['없음'])}. "
            "SEC 8-K 목록에서 그 분기 8-K에 Item 2.02 표시가 없는 경우입니다(실적 자료를 9.01로만 표시하는 "
            "등). 지주회사 전환 등으로 CIK가 바뀐 회사는 두 CIK를 모두 조회했습니다. 이 회사들이 상대일 "
            "때는 빠진 발표일이 제외되지 않아 그 이벤트-쌍이 남을 수 있습니다."
        )
    if ev.get("firms_failed"):
        out.append(f"- 8-K 목록을 받지 못한 회사 {ev['firms_failed']}곳은 빠졌습니다.")
    return out


def _section_groups(result: dict, per: float) -> list[str]:
    specs, top_k = result["specs"], result["info"]["top_k"]
    main, placebo = specs[MAIN]["groups"], specs["placebo"]["groups"]
    trans, big = result["transfer"], specs["big"]["groups"]
    exc = result["exclusion"]
    lines = [
        "",
        "## 3. 관계 유형별 반응 계수",
        "",
        "쌍 묶음마다 이벤트-쌍(발표 회사의 이벤트 하나 × 상대 회사 하나)을 모아 상대 CAR을 발표 회사 CAR에 "
        f"회귀한 기울기입니다. 양쪽 회사가 번갈아 발표 회사가 되므로 쌍 하나에 이벤트가 평균 {per:.1f}건 "
        "들어갑니다(상대가 같은 때 발표한 이벤트를 뺀 뒤).",
        "",
        "**평소 날과 비교.** 발표일의 발표 회사 CAR에는 실적 뉴스 말고도 그날의 업종 공통 움직임이 섞여 "
        "있습니다. 같은 회사와 같은 상대를 발표와 무관한 날(발표일 ±10·±20거래일, 두 회사 모두 발표 창이 "
        "아닌 날)에 재면 '평소 날 기울기'가 나옵니다. 평소에는 발표 회사 움직임에서 업종 공통 움직임의 "
        "비중이 발표일보다 커서(같은 업종 쌍의 평소 날 상관 약 0.3) 이 기울기가 큽니다. 발표일에는 발표 "
        "회사 고유의 뉴스가 크게 더해지므로 기울기가 작아집니다. '평소 동조성을 뺀 몫'은 발표일에 늘어난 "
        "공분산을 늘어난 분산으로 나눈 값으로, 실적 뉴스 자체에 대한 반응을 어림합니다. 뉴스에 대한 반응이 "
        "이벤트마다 같다는 가정에 기댄 추정치입니다.",
        "",
    ]
    rows = [
        [_label(k, top_k), _slope(main[k]), _slope(placebo[k]), _diff(trans[k]),
         f"{main[k]['n_event_pairs']:,}", f"{main[k]['n_pairs']:,}"]
        for k in TABLE_GROUPS
    ]  # fmt: skip
    lines += research._table(
        [
            "쌍 묶음",
            "반응 계수 [0, +1]",
            "평소 날 기울기",
            "평소 동조성을 뺀 몫",
            "이벤트-쌍",
            "쌍",
        ],
        rows,
    )
    share = exc["share"]
    dropped = exc["dropped_pairs"]
    con = specs[MAIN]["contrasts"]
    lines += [
        "",
        "- '모든 쌍'이 0에 가까운 것은 결과가 아니라 계산의 성질입니다. 잔차는 같은 회사들의 동일가중 "
        "시장을 뺀 값이라, 한 이벤트에서 나머지 회사의 CAR을 모두 더하면 거의 0이 됩니다. 그래서 같은 업종 "
        "쌍의 기울기가 양수인 만큼 나머지 쌍은 조금 음수가 됩니다. 묶음끼리의 차이로 읽습니다.",
        f"- 큰 소식(발표 회사 |CAR| ≥ 5%)만 쓰면 경쟁 {_slope(big['competitor'])}, 공급·협력 "
        f"{_slope(big['business'])}로 전체와 거의 같습니다. 큰 소식에서는 평소 움직임이 발표 회사 CAR에서 "
        "차지하는 비중이 작으므로, 평소 동조성이 반응 계수를 크게 부풀린 것은 아닙니다. 위 가정대로라면 "
        "큰 소식의 기울기가 더 작아져야 하므로, '평소 동조성을 뺀 몫'은 뉴스 반응을 낮게 어림했을 수 있습니다.",
        f"- 상대 회사가 창이 겹치게 발표해 빠진 이벤트-쌍: 경쟁 {_pct(share['competitor'], 0)}, 공급·협력 "
        f"{_pct(share['business'], 0)}, 같은 서브산업·관계·언급 없음 {_pct(share['same_sub_unrelated'], 0)}, "
        f"모든 쌍 {_pct(share['all'], 0)}."
        + (
            f" 늘 창이 겹치게 함께 발표해 통째로 빠진 경쟁 쌍 {len(dropped['competitor'])}개: "
            f"{', '.join(dropped['competitor'])}."
            if dropped["competitor"]
            else ""
        ),
        f"- 관계 쌍 − '회사 언급, 관계 아님' 쌍(짝지은 차이): 경쟁 {_diff(con['competitor_vs_mention'])}, "
        f"공급·협력 {_diff(con['business_vs_mention'])}. 두 묶음은 GICS 위치 구성이 달라, 4장에서 같은 "
        "위치끼리 다시 비교합니다.",
    ]
    return lines


def _section_positions(result: dict) -> list[str]:
    g = result["specs"][MAIN]["groups"]
    con = result["specs"][MAIN]["contrasts"]
    lines = [
        "",
        "## 4. GICS 위치별",
        "",
        "경쟁 쌍은 같은 서브산업에 많고, 공급·협력 쌍은 대부분 다른 서브산업입니다. 그래서 관계 묶음을 "
        "'같은 서브산업, 관계·언급 없음' 하나와 비교하면 업종 구성의 차이가 섞입니다. 같은 GICS 위치끼리 "
        "나란히 놓습니다. 칸마다 반응 계수 [95% 구간] · 쌍 수입니다.",
        "",
    ]
    rows = [
        [label, *(_cell(g[f"{row}@{pos}"]) for pos in POSITIONS)]
        for row, label in POSITION_ROWS.items()
    ]
    lines += research._table(["쌍 묶음", *(v[0] for v in POSITIONS.values())], rows)
    lines += [
        "",
        "같은 위치의 '관계·언급 없음'과의 차이(짝지은 차이):",
        "",
    ]
    rows = [
        [label, *(_diff(con[f"{r}_vs_unrelated@{pos}"]) for pos in POSITIONS)]
        for r, label in (
            ("competitor", "경쟁 − 관계·언급 없음"),
            ("business", "공급·협력 − 관계·언급 없음"),
        )
    ]
    lines += research._table(["", *(v[0] for v in POSITIONS.values())], rows)
    return lines


def _section_robust(result: dict) -> list[str]:
    specs, top_k = result["specs"], result["info"]["top_k"]
    lines = [
        "",
        "## 5. 같은 결론이 나오는지 (강건성)",
        "",
        "창을 넓히거나(보도자료보다 8-K가 늦은 회사), 발표 회사·상대 회사 CAR을 양쪽 1%에서 잘라 극단적인 "
        "이벤트(예: 이틀 −50%)의 영향을 줄이거나, 발표 회사와 같은 서브산업의 다른 회사가 창이 겹치게 함께 "
        f"발표한 이벤트를 빼거나, 이벤트가 {TYPICAL_EVENTS}건보다 많은(실적 예고 등이 섞인) 회사의 발표를 "
        "뺍니다(상대 회사로는 남음).",
        "",
    ]
    rows = [
        [_label(k, top_k), *(_slope(specs[s]["groups"][k]) for s in ROBUST)] for k in TABLE_GROUPS
    ]
    lines += research._table(["쌍 묶음", *ROBUST.values()], rows)
    lines += [
        "",
        f"'같은 업종 동시 발표 제외'는 이벤트가 {specs['solo']['n_events']:,}건 남습니다(전체 "
        f"{specs[MAIN]['n_events']:,}건). 실적 시즌에는 같은 업종이 몰려 발표하므로, 남는 것은 대부분 회사 "
        "수가 적은 서브산업의 이벤트입니다. 업종 공통 뉴스를 줄이는 대신 표본 구성이 달라집니다.",
    ]
    return lines


def _section_news(result: dict) -> list[str]:
    specs, top_k = result["specs"], result["info"]["top_k"]
    asym = result["asymmetry"]
    lines = [
        "",
        "## 6. 좋은 소식과 나쁜 소식",
        "",
        "발표 회사 CAR[0, +1]이 양수인 이벤트(좋은 소식)와 음수인 이벤트(나쁜 소식)를 나눠 같은 기울기를 "
        "잽니다. 경쟁 효과가 한쪽 소식에서 더 강하다면(예: 경쟁사의 나쁜 소식이 내게는 점유율 기회) 그쪽 "
        "기울기가 작아집니다. 좋은 소식과 나쁜 소식에 똑같이 작용하는 경쟁 효과는 이 비교로 드러나지 "
        "않습니다. 마지막 열은 같은 재표집 표본으로 짝지은 차이입니다.",
        "",
    ]
    rows = [
        [_label(k, top_k), _slope(specs["good"]["groups"][k]), _slope(specs["bad"]["groups"][k]),
         _diff(asym[k])]
        for k in TABLE_GROUPS
    ]  # fmt: skip
    lines += research._table(["쌍 묶음", "좋은 소식", "나쁜 소식", "나쁜 − 좋은"], rows)
    lines += [
        "",
        f"좋은 소식 {specs['good']['n_events']:,}건, 나쁜 소식 {specs['bad']['n_events']:,}건입니다. "
        "나눈 표본은 이벤트 몇 건(한 해에 하루 ±30% 안팎)에 크게 좌우될 수 있어, 차이의 구간이 0을 포함하면 "
        "해석하지 않습니다.",
    ]
    return lines


def _section_regression(result: dict) -> list[str]:
    specs, top_k = result["specs"], result["info"]["top_k"]
    reg = specs[MAIN]["regression"]
    lines = [
        "",
        "## 7. 통제 회귀: 분류·유사도를 통제하면",
        "",
        "모든 이벤트-쌍을 한 회귀에 넣고, 쌍 표시마다 '반응 계수에 더해지는 몫'을 추정했습니다"
        "(상대 CAR = α + Σγ·표시 + (β + Σβ_표시·표시) × 발표 회사 CAR). 구간이 0을 포함하지 않으면 "
        "**굵게** 표시합니다.",
        "",
        "- 첫 행 β는 기준 쌍의 기울기입니다. (1)에서는 관계·언급이 없고 섹터가 다른 쌍, (2)에서는 여기에 "
        f"유사도 백분위가 50 미만이고 상위 {top_k} 이웃이 아닌 쌍입니다. 기준이 조금 음수인 것은 3장의 "
        "'모든 쌍' 설명과 같은 계산의 성질입니다(같은 업종 쌍이 양수인 만큼 나머지가 음수).",
        "- 나머지 행은 그 표시가 있을 때 기울기에 더해지는 몫입니다. 같은 서브산업 행은 같은 섹터 위에 "
        "더해지는 몫입니다.",
        "- (2)는 2-3과 같은 유사도 백분위 구간 더미와 상위 이웃 표시를 더합니다. 2-3의 회사 고정효과 모형은 "
        "여기서는 쓰지 않습니다(10장).",
        "",
    ]
    rows = [
        [label.format(k=top_k),
         *("" if key not in reg[m]["coefs"] else _coef(reg[m]["coefs"][key]) for m in MODELS)]
        for key, label in COEFS.items()
    ]  # fmt: skip
    rows.append([f"유사도 구간 ({len(research.SIM_BINS)}개)", "", "포함"])
    rows.append(["이벤트-쌍", *(f"{reg[m]['n_event_pairs']:,}" for m in MODELS)])
    lines += research._table(["", *MODEL_LABELS.values()], rows)
    lines += ["", "경쟁·공급·협력의 추가 반응 계수를 다른 창과 표본에서도 봅니다.", ""]
    variants = {MAIN: "[0, +1]", **ROBUST, "good": "좋은 소식", "bad": "나쁜 소식",
                "drift": "시차 [+2, +5]"}  # fmt: skip
    rows = [
        [label,
         *(_coef(specs[s]["regression"][m]["coefs"].get(r))
           for r in ("competitor", "business") for m in MODELS)]
        for s, label in variants.items()
    ]  # fmt: skip
    lines += research._table(
        ["창·표본", "경쟁 (1)", "경쟁 (2)", "공급·협력 (1)", "공급·협력 (2)"], rows
    )
    return lines


def _section_lag(result: dict) -> list[str]:
    specs, top_k = result["specs"], result["info"]["top_k"]
    drift, ll = specs["drift"]["groups"], result["lead_lag"]
    lines = [
        "",
        "## 8. 시차: 발표 뒤에도 이어지나",
        "",
        "공급망 소식이 며칠 늦게 반영된다면(Cohen & Frazzini 2008), 발표 뒤 [+2, +5]의 상대 CAR이 발표 "
        "회사 CAR[0, +1]을 따라갑니다. 오른쪽 열은 이벤트와 상관없이 오늘 한 회사의 잔차와 다음 날 상대 "
        "잔차의 평균 상관입니다. 방향 없이 양쪽을 평균하고 하루 단위로 보므로, 고객 → 공급사로 한 달에 걸쳐 "
        "번지는 효과(Cohen & Frazzini의 검정)와는 다릅니다.",
        "",
    ]
    rows = [
        [_label(k, top_k), _slope(drift[k]), f"{drift[k]['n_event_pairs']:,}", _mean(ll[k])]
        for k in TABLE_GROUPS
    ]
    lines += research._table(
        ["쌍 묶음", "반응 계수 [+2, +5]", "이벤트-쌍", "하루 뒤 잔차 상관"], rows
    )
    return lines


def _section_examples(result: dict) -> list[str]:
    ex = result["examples"]
    lines = [
        "",
        "## 9. 예시: 발표 회사 반응이 컸던 이벤트",
        "",
        f"관계 상대의 반응을 나란히 봅니다(이벤트마다 반응이 큰 상대 {PER_EVENT}곳까지). 같은 방향이면 "
        "전염, 반대 방향이면 경쟁 효과에 가깝습니다. 상대 CAR은 시장에서 발표 회사를 뺀 값입니다.",
        "",
    ]
    for rel, title in (("competitor", "경쟁"), ("business", "공급·협력")):
        lines += [f"### {title}", "", *_example_table(ex[rel]), ""]
    return lines


def _section_caveats() -> list[str]:
    return [
        "## 10. 해석할 때 주의할 점",
        "",
        "- 뉴스의 크기를 발표 회사 주가 반응으로 잽니다. 컨센서스 대비 실적 서프라이즈가 아니므로 발표 회사 "
        "주가가 그날의 업종 공통 움직임으로 움직인 몫도 섞입니다. 3장의 평소 날 비교가 그 크기를 보여 줍니다. "
        "'같은 업종 동시 발표 제외'나 GICS 통제로는 이 몫이 없어지지 않습니다.",
        "- 반응 계수는 전염(같은 방향)과 경쟁 효과(반대 방향)가 합쳐진 순효과입니다. 양수라도 경쟁 효과가 "
        "없다는 뜻은 아닙니다.",
        "- OLS 기울기는 발표 회사 반응이 큰 이벤트에 무게가 많이 실립니다. 5장의 '극단값 1% 제한'으로 몇몇 "
        "이벤트가 결론을 좌우하지 않는지 봅니다.",
        "- 관계에 방향이 없어(누가 누구에게 파는지 묻지 않음) 고객 → 공급사 같은 방향별 전이는 보지 못합니다.",
        "- 0일은 8-K 접수 시각(과 보고일)으로 잡습니다. 장 전에 보도자료를 내고 장 마감 뒤에 8-K를 낸 "
        "회사는 0일이 하루 늦습니다. 드문 일이라 2장의 −1일 평균 반응이 평소와 비슷하고, 5장의 [−1, +1] "
        "창이 이런 경우를 덮습니다.",
        "- 회사 고정효과(회사마다 다른 반응 크기)는 통제하지 않았습니다. 2-3에서는 회사 고정효과가 결론을 "
        "바꾸지 않았습니다.",
        "- 이벤트는 한 해(2025년)뿐이고 S&P 500 대형주만 봅니다. 시차 효과는 관심이 적은 소형주에서 크다고 "
        "알려져 있어, 여기서 보이지 않았다고 일반화할 수 없습니다.",
        "- 시장 모형은 이벤트 날을 포함한 한 해 전체로 추정했습니다. 회사당 이벤트 날은 1년에 8일 정도라 "
        "영향이 작습니다.",
    ]


def _summary(result: dict) -> list[str]:
    specs = result["specs"]
    main = specs[MAIN]
    g, reg = main["groups"], main["regression"]
    top_k = result["info"]["top_k"]
    trans, placebo = result["transfer"], specs["placebo"]["groups"]
    out = [
        f"- **반응 계수 [0, +1]:** 경쟁 {_slope(g['competitor'])}, 공급·협력 {_slope(g['business'])}. "
        f"비교: 같은 서브산업·관계·언급 없음 {_slope(g['same_sub_unrelated'])}, 유사도 상위 {top_k}·언급 없음 "
        f"{_slope(g['similar_only'])}, 회사 언급·관계 아님 {_slope(g['mention_only'])}.",
        f"- **평소 동조성을 뺀 몫(3장):** 경쟁 {_diff(trans['competitor'])}, 공급·협력 "
        f"{_diff(trans['business'])}, 같은 서브산업·관계·언급 없음 {_diff(trans['same_sub_unrelated'])}. "
        f"같은 쌍의 평소 날 기울기는 경쟁 {_slope(placebo['competitor'])}, 공급·협력 "
        f"{_slope(placebo['business'])}로, 발표일 반응 계수의 일부는 평소에도 같이 움직이는 몫입니다.",
    ]
    for r, label in (("competitor", "경쟁"), ("business", "공급·협력")):
        steps = " → ".join(
            f"{MODEL_LABELS[m].split(' ')[0]} {_coef(reg[m]['coefs'].get(r))}" for m in MODELS
        )
        out.append(f"- {label}: 통제 회귀의 추가 반응 계수 {steps}. {_verdict(result, r)}")
    drift = specs["drift"]["groups"]
    out.append(
        f"- 시차: 발표 뒤 [+2, +5] 반응 계수 경쟁 {_slope(drift['competitor'])}, 공급·협력 "
        f"{_slope(drift['business'])}. 구간이 넓어 작은 시차 반응까지 배제하지는 못합니다."
    )
    out.append(f"- **결론:** {_conclusion(result)}")
    return out


def _robust_significant(result: dict, model: str, r: str, sign: int) -> list[str]:
    """model의 r 계수가 sign 쪽으로 0과 구별되는 창·표본 (주 창 포함)."""
    specs = result["specs"]
    labels = {MAIN: "[0, +1]", **ROBUST}
    out = []
    for s, label in labels.items():
        e = specs[s]["regression"][model]["coefs"].get(r)
        if _sig(e) and np.sign(e["coef"]) == sign:
            out.append(label)
    return out


def _verdict(result: dict, r: str) -> str:
    """묶음 기울기, 평소 동조성을 뺀 몫, (1)·(2)의 추가 반응 계수를 함께 보고 읽는다."""
    specs = result["specs"]
    group = specs[MAIN]["groups"][r]
    reg = specs[MAIN]["regression"]
    gics, text = (reg[m]["coefs"].get(r) for m in MODELS)
    if group.get("slope") is None or not all(research._estimated(e) for e in (gics, text)):
        return "추정할 수 없습니다(이벤트-쌍이 너무 적음)."
    trans = result["transfer"][r]
    if _sig(group):
        same = group["slope"] > 0
        direction = "같은 방향(전염)" if same else "반대 방향"
        if _sig(trans) and (trans["diff"] > 0) == same:
            moves = (
                f"상대 주가는 {direction}으로 반응하고, 평소 동조성을 뺀 추정치(가정에 기댐)도 "
                "0과 구별됩니다."
            )
        else:
            moves = (
                f"반응 계수는 {direction}이지만, 평소 동조성을 뺀 추정치(가정에 기댐)는 0과 "
                "구별되지 않습니다."
            )
    else:
        moves = "반응 계수의 구간이 0을 포함합니다."
    if _sig(text):
        more = "더 크게" if text["coef"] > 0 else "더 작게"
        return f"{moves} GICS·텍스트 유사도를 통제해도 관계가 없는 쌍보다 {more} 반응합니다."
    parts = []
    for sign, word in ((1, "크게"), (-1, "작게")):
        where = _robust_significant(result, "gics", r, sign)
        if where:
            parts.append(f"GICS만 통제하면 {', '.join(where)}에서 관계 쌍이 더 {word} 반응합니다")
    text_where = [
        label
        for label in ROBUST.values()
        if label
        in _robust_significant(result, "text", r, 1) + _robust_significant(result, "text", r, -1)
    ]
    tail = (
        "텍스트 유사도를 더하면 주 창과 다른 창·표본 모두에서 0과 구별되지 않습니다."
        if not text_where
        else "텍스트 유사도를 더하면 주 창에서는 0과 구별되지 않지만, "
        f"{', '.join(text_where)}에서는 구별됩니다."
    )
    if parts:
        return f"{moves} {'; '.join(parts)}. {tail}"
    return f"{moves} GICS만 통제해도 관계가 없는 쌍과 뚜렷한 차이가 없습니다."


def _conclusion(result: dict) -> str:
    """관계가 실적 발표 전이를 GICS·유사도 이상으로 설명하는지 한 문장으로 (2-3의 같은 날 동조성과 비교)."""
    specs = result["specs"]
    text = specs[MAIN]["regression"]["text"]["coefs"]
    extra = [r for r in ("competitor", "business") if research._estimated(text.get(r))]
    if len(extra) < 2:
        return "추정할 수 없는 관계 유형이 있습니다(이벤트-쌍이 너무 적음)."
    beyond = [r for r in extra if _sig(text[r])]
    lagged = [r for r in extra if _sig(specs["drift"]["groups"][r])]
    if not beyond:
        first = (
            "이 표본(S&P 500, 2025년, 방향 없는 관계)에서는 실적 발표 때 관계 상대가 같이 움직이는 몫도 "
            "GICS와 텍스트 유사도가 이미 설명합니다. 2-3의 같은 날 동조성과 같은 결론입니다"
        )
    else:
        names = "·".join(GROUPS[r] for r in beyond)
        first = f"{names} 관계는 GICS·텍스트 유사도를 통제해도 실적 발표 반응을 더 설명합니다"
    second = (
        "발표 뒤 며칠에 걸친 시차 반응도 보이지 않습니다"
        if not lagged
        else f"발표 뒤 시차 반응이 보입니다({'·'.join(GROUPS[r] for r in lagged)})"
    )
    return f"{first}. {second}."


def _example_table(items: list[dict]) -> list[str]:
    if not items:
        return ["해당하는 이벤트가 없습니다."]
    rows = [
        [x["date"], research._name(x["announcer"]), f"{x['news']:+.1%}", research._name(x["partner"]),
         f"{x['response']:+.1%}"]
        for x in items
    ]  # fmt: skip
    return research._table(
        ["0일", "발표 회사", "CAR [0, +1]", "상대 회사", "상대 CAR [0, +1]"], rows
    )


def report_path(cfg: Config) -> Path:
    return cfg.reports_dir / f"relations_events_{cfg.name}.md"


def stage_events(
    cfg: Config, *, n_boot: int | None = None, refresh: bool = False
) -> tuple[dict, Path]:
    """이벤트 스터디를 돌려 결과(relations/events.json)와 보고서(reports/relations_events_<name>.md)를 쓴다."""
    data = load_event_data(cfg, refresh=refresh)
    result = analyze(data, n_boot=n_boot or cfg.evaluation.n_boot)
    out = cfg.relations_dir / "events.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    path = report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(result) + "\n", encoding="utf-8")
    return result, path

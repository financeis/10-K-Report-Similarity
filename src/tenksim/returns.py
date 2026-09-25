"""주가 수익률 기반 검증.

사업이 비슷한 회사끼리는 주가도 같이 움직일 것이라는 가정으로, 텍스트 이웃 간의
일별 수익률 상관을 잰다 (참고 논문 Table 1과 같은 지표).

두 가지를 함께 본다.
- raw: 일별 수익률 그대로의 상관. 시장 전체 움직임(베타)이 대부분을 차지한다.
- resid: 시장 모형 r_i = a + b * r_market + e 의 잔차 e끼리의 상관.
  시장 공통 움직임을 걷어내므로 '사업이 비슷해서 같이 움직이는' 부분에 더 가깝다.

시장 수익률은 기본으로 유니버스 종목의 동일가중 평균(자기 자신 제외)을 쓴다.
SPY 같은 시가총액 가중 지수를 쓰면 지수 비중이 큰 대형주끼리 잔차 상관이 음(-)으로
치우친다. 지수가 사실상 그들의 평균이라, 지수를 빼면 그들이 공유하는 움직임까지 빠지기 때문이다.
(2025년 Apple-Microsoft: SPY 기준 -0.09, 동일가중 기준 +0.19)

look-ahead bias를 피하려면 수익률 구간을 공시가 모두 나온 뒤(예: 공시 연도 다음 해)로 잡는다.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EQUAL_WEIGHT = "equal_weight"


def yahoo_symbol(ticker: str) -> str:
    # 위키피디아/SEC의 BRK.B → Yahoo의 BRK-B
    return ticker.replace(".", "-")


def load_prices(
    tickers: list[str], market: str, start: date, end: date, cache_path: Path
) -> pd.DataFrame:
    """수정주가(배당·분할 반영) 종가. 컬럼은 원래 티커, 시장 지수는 market 이름."""
    symbols = {yahoo_symbol(t) for t in tickers}
    if market != EQUAL_WEIGHT:
        symbols.add(market)
    symbols = sorted(symbols)
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if set(symbols) <= set(cached.columns):
            return _rename(cached[symbols], tickers)

    import yfinance as yf

    log.info("Downloading prices for %d symbols (%s ~ %s)", len(symbols), start, end)
    # yfinance의 end는 그 날짜를 포함하지 않는다
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=symbols)
    failed = [s for s in symbols if close[s].notna().sum() == 0]
    if failed:
        log.warning("No price data for %d symbols: %s", len(failed), failed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    return _rename(close, tickers)


def _rename(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    mapping = {yahoo_symbol(t): t for t in tickers}
    return close.rename(columns=mapping)


def equal_weight_market(returns: pd.DataFrame) -> pd.DataFrame:
    """종목마다 '자기 자신을 뺀' 나머지 종목의 동일가중 평균 수익률.

    종목 수가 적으면 자기 자신이 평균에 섞여 베타와 잔차가 왜곡되므로 빼고 계산한다.
    """
    total = returns.sum(axis=1, min_count=1)
    count = returns.notna().sum(axis=1)
    others = count.to_numpy()[:, None] - returns.notna().to_numpy()
    own = returns.fillna(0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (total.to_numpy()[:, None] - own) / others
    loo[others <= 0] = np.nan
    return pd.DataFrame(loo, index=returns.index, columns=returns.columns)


def residualize(
    returns: pd.DataFrame, market: pd.Series | pd.DataFrame, min_obs: int
) -> pd.DataFrame:
    """종목별로 시장 모형을 OLS로 추정하고 잔차를 돌려준다. 관측치가 부족하면 NaN 열.

    market이 DataFrame이면 종목마다 자기 열을 시장 수익률로 쓴다 (동일가중 시장처럼).
    """
    out = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for col in returns.columns:
        r = returns[col].to_numpy()
        m = (market[col] if isinstance(market, pd.DataFrame) else market).to_numpy()
        ok = ~np.isnan(r) & ~np.isnan(m)
        if ok.sum() < min_obs:
            continue
        x = np.column_stack([np.ones(ok.sum()), m[ok]])
        beta, *_ = np.linalg.lstsq(x, r[ok], rcond=None)
        resid = np.full(len(r), np.nan)
        resid[ok] = r[ok] - x @ beta
        out[col] = resid
    return out


def residual_returns(
    prices: pd.DataFrame, tickers: list[str], market: str, min_obs: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """tickers 순서의 (일별 수익률, 시장 모형 잔차, 사용 가능 여부). 수익률 이력이 부족한 종목은
    잔차가 NaN 열이다.

    market: EQUAL_WEIGHT(유니버스 동일가중, 자기 제외) 또는 가격 표에 있는 지수 티커(예: SPY).
    """
    rets = prices.pct_change(fill_method=None).iloc[1:]
    stock = rets.reindex(columns=tickers)
    enough = (stock.notna().sum() >= min_obs).to_numpy()
    # 수익률 이력이 부족한 종목은 시장 평균에도 넣지 않는다
    usable = stock.loc[:, enough]
    factor = equal_weight_market(usable) if market == EQUAL_WEIGHT else rets[market]
    resid = residualize(usable, factor, min_obs).reindex(columns=tickers)
    return stock, resid, enough


def correlation_matrices(
    prices: pd.DataFrame, tickers: list[str], market: str, min_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tickers 순서의 (raw 상관, 잔차 상관, 사용 가능 여부) 행렬 (market은 residual_returns와 같다)."""
    stock, resid, enough = residual_returns(prices, tickers, market, min_obs)
    raw = stock.corr(min_periods=min_obs).to_numpy()
    return raw, resid.corr(min_periods=min_obs).to_numpy(), enough


def peer_corr_per_firm(corr: np.ndarray, peers: list[np.ndarray]) -> np.ndarray:
    """회사별 '자기 peer들과의 평균 상관'. peer가 없으면 NaN."""
    out = np.full(len(peers), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # peer의 수익률이 전부 NaN인 경우
        for i, p in enumerate(peers):
            if len(p):
                out[i] = np.nanmean(corr[i, p])
    return out


def peer_correlation(corr: np.ndarray, peers: list[np.ndarray]) -> float:
    """회사마다 '자기 peer들과의 평균 상관'을 구해, 회사들에 대해 다시 평균낸다."""
    values = peer_corr_per_firm(corr, peers)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def random_per_firm(corr: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """회사별로 허용된 상대 전체와의 평균 상관 = 그중에서 무작위로 골랐을 때의 기댓값."""
    mask = ~np.eye(corr.shape[0], dtype=bool)
    if allowed is not None:
        mask &= allowed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(mask, corr, np.nan), axis=1)


def top_k_within(sim: np.ndarray, allowed: np.ndarray, k: int) -> list[np.ndarray]:
    """allowed[i, j]가 참인 상대 중에서 회사 i와 가장 비슷한 k개 (자기 자신 제외)."""
    s = np.where(allowed, sim.astype(np.float64), -np.inf)
    np.fill_diagonal(s, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return [row[np.isfinite(s[i, row])] for i, row in enumerate(order)]


def incremental_effect(
    sim: np.ndarray,
    corr: np.ndarray,
    same_child: np.ndarray,
    same_parent: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """산업분류를 통제한 뒤에도 텍스트 유사도가 주가 동조성을 설명하는지 (기업쌍 회귀).

        잔차상관_ij = a + b1·같은_서브산업 + b2·같은_섹터 + β·z(유사도_ij)

    β는 같은 분류인지 여부를 고정했을 때 유사도가 1 표준편차 높으면 잔차 상관이 얼마나
    높은지다. β > 0이면 텍스트가 분류표에 없는 연결을 담고 있다는 뜻이다.
    한 회사가 여러 쌍에 등장해 쌍끼리 독립이 아니므로, 구간은 회사 단위 재표집으로 구한다.
    """
    n = sim.shape[0]
    a, b = np.triu_indices(n, 1)
    x_all = sim[a, b].astype(np.float64)
    finite = np.isfinite(x_all) & np.isfinite(corr[a, b])
    mu, sd = x_all[finite].mean(), x_all[finite].std()

    def fit(i: np.ndarray, j: np.ndarray) -> tuple[float, float, float]:
        y = corr[i, j]
        x = (sim[i, j].astype(np.float64) - mu) / sd
        ok = np.isfinite(y) & np.isfinite(x)
        y = y[ok]
        base = np.column_stack([np.ones(ok.sum()), same_child[i, j][ok], same_parent[i, j][ok]])
        full = np.column_stack([base, x[ok]])
        tss = ((y - y.mean()) ** 2).sum()
        r2 = []
        for design in (base, full):
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            r2.append(1 - ((y - design @ coef) ** 2).sum() / tss)
        return float(coef[-1]), float(r2[0]), float(r2[1])

    beta, r2_gics, r2_full = fit(a, b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        r = rng.integers(0, n, size=n)
        ri, rj = r[a], r[b]
        keep = ri != rj  # 같은 회사가 두 번 뽑혀 생긴 자기 자신과의 쌍은 뺀다
        boots.append(fit(ri[keep], rj[keep])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {
        "beta": beta,
        "lo": float(lo),
        "hi": float(hi),
        "r2_gics": r2_gics,
        "r2_full": r2_full,
        "n_pairs": int(finite.sum()),
    }


def label_peers(labels: np.ndarray) -> list[np.ndarray]:
    """같은 레이블을 가진 다른 회사 전부 (GICS·SIC 기준선, 참고 논문의 'dynamic k')."""
    peers = []
    for i, v in enumerate(labels):
        if not isinstance(v, str):
            peers.append(np.array([], dtype=int))
            continue
        same = np.flatnonzero(labels == v)
        peers.append(same[same != i])
    return peers


def random_baseline(corr: np.ndarray) -> float:
    """무작위 peer의 기댓값 = 대각선을 뺀 전체 상관의 평균."""
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(corr[mask]))

"""주가 수익률 기반 검증.

사업이 비슷한 회사끼리는 주가도 같이 움직일 것이라는 가정으로, 텍스트 이웃 간의
일별 수익률 상관을 잰다 (참고 논문 Table 1과 같은 지표).

두 가지를 함께 본다.
- raw: 일별 수익률 그대로의 상관. 시장 전체 움직임(베타)이 대부분을 차지한다.
- resid: 시장 모형 r_i = a + b * r_market + e 의 잔차 e끼리의 상관.
  시장 공통 움직임을 걷어내므로 '사업이 비슷해서 같이 움직이는' 부분에 더 가깝다.

look-ahead bias를 피하려면 수익률 구간을 공시가 모두 나온 뒤(예: 공시 연도 다음 해)로 잡는다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def yahoo_symbol(ticker: str) -> str:
    # 위키피디아/SEC의 BRK.B → Yahoo의 BRK-B
    return ticker.replace(".", "-")


def load_prices(
    tickers: list[str], market: str, start: date, end: date, cache_path: Path
) -> pd.DataFrame:
    """수정주가(배당·분할 반영) 종가. 컬럼은 원래 티커, 시장 지수는 market 이름."""
    symbols = sorted({yahoo_symbol(t) for t in tickers} | {market})
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


def residualize(returns: pd.DataFrame, market: pd.Series, min_obs: int) -> pd.DataFrame:
    """종목별로 시장 모형을 OLS로 추정하고 잔차를 돌려준다. 관측치가 부족하면 NaN 열."""
    out = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    m = market.to_numpy()
    for col in returns.columns:
        r = returns[col].to_numpy()
        ok = ~np.isnan(r) & ~np.isnan(m)
        if ok.sum() < min_obs:
            continue
        x = np.column_stack([np.ones(ok.sum()), m[ok]])
        beta, *_ = np.linalg.lstsq(x, r[ok], rcond=None)
        resid = np.full(len(r), np.nan)
        resid[ok] = r[ok] - x @ beta
        out[col] = resid
    return out


def correlation_matrices(
    prices: pd.DataFrame, tickers: list[str], market: str, min_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tickers 순서의 (raw 상관, 잔차 상관, 사용 가능 여부) 행렬."""
    rets = prices.pct_change(fill_method=None).iloc[1:]
    stock = rets.reindex(columns=tickers)
    enough = (stock.notna().sum() >= min_obs).to_numpy()
    raw = stock.corr(min_periods=min_obs).to_numpy()
    resid = residualize(stock, rets[market], min_obs).corr(min_periods=min_obs).to_numpy()
    return raw, resid, enough


def peer_correlation(corr: np.ndarray, peers: list[np.ndarray]) -> float:
    """회사마다 '자기 peer들과의 평균 상관'을 구해, 회사들에 대해 다시 평균낸다."""
    per_company = [np.nanmean(corr[i, p]) for i, p in enumerate(peers) if len(p)]
    per_company = [v for v in per_company if not np.isnan(v)]
    return float(np.mean(per_company)) if per_company else float("nan")


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

"""분석 대상 기업 목록(universe)을 만든다.

기본값은 위키피디아의 S&P 500 구성종목 표다. GICS 섹터/서브산업이 함께 있어
평가용 정답(label)으로 바로 쓸 수 있다. 단, 현재 시점 구성종목이라 과거 연도에
적용하면 생존 편향(survivorship bias)이 생긴다 (docs/methodology.md 참고).
"""

from __future__ import annotations

import io
import logging

import httpx
import pandas as pd

from .config import UniverseConfig

log = logging.getLogger(__name__)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# 위키피디아는 연락처가 담긴 User-Agent를 요구한다. EDGAR용 개인 식별자는 보내지 않는다.
WIKIPEDIA_USER_AGENT = "tenksim/0.2 (https://github.com/financeis/10-K-Report-Similarity)"
COLUMNS = ["cik", "ticker", "name", "gics_sector", "gics_sub_industry"]


def load_sp500_wikipedia() -> pd.DataFrame:
    resp = httpx.get(
        SP500_URL,
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        follow_redirects=True,
        timeout=30,
    )
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})[0]
    df = table.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "CIK": "cik",
        }
    )
    df["cik"] = df["cik"].astype(int)
    # 의결권 클래스가 여럿인 회사(GOOGL/GOOG 등)는 10-K가 하나이므로 첫 클래스만 남긴다
    df = df.drop_duplicates("cik", keep="first")
    return df[COLUMNS].reset_index(drop=True)


def load_csv(path) -> pd.DataFrame:
    """ticker 또는 cik 컬럼이 있는 CSV를 읽는다. GICS 컬럼은 있으면 쓰고 없으면 비워 둔다."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "cik" not in df.columns and "ticker" not in df.columns:
        raise ValueError(
            f"{path}: 'ticker' 또는 'cik' 컬럼이 필요합니다 (있는 컬럼: {list(df.columns)})"
        )
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["ticker"] = [t.strip().upper() if isinstance(t, str) else None for t in df["ticker"]]

    if df["cik"].isna().any():
        # SEC company_tickers.json 기준 매핑. SEC는 BRK-B 형식, 위키피디아는 BRK.B 형식을 쓴다.
        from edgar import get_ticker_to_cik_lookup

        lookup = get_ticker_to_cik_lookup()
        df["cik"] = [
            cik if pd.notna(cik) else lookup.get(t) or lookup.get(str(t).replace(".", "-"))
            for cik, t in zip(df["cik"], df["ticker"], strict=True)
        ]
        unresolved = df.loc[df["cik"].isna(), "ticker"].tolist()
        if unresolved:
            raise ValueError(f"CIK를 찾지 못한 티커가 있습니다: {unresolved}")
    df["cik"] = df["cik"].astype(int)
    return df[COLUMNS].drop_duplicates("cik", keep="first").reset_index(drop=True)


def build_universe(cfg: UniverseConfig) -> pd.DataFrame:
    df = load_sp500_wikipedia() if cfg.source == "sp500_wikipedia" else load_csv(cfg.path)
    if cfg.tickers:
        wanted = [t.strip().upper() for t in cfg.tickers]
        unknown = sorted(set(wanted) - set(df["ticker"]))
        if unknown:
            raise ValueError(f"universe에 없는 티커입니다: {unknown}")
        df = df[df["ticker"].isin(wanted)]
    if cfg.limit:
        df = df.head(cfg.limit)
    if cfg.cik_overrides:
        overrides = {t.strip().upper(): cik for t, cik in cfg.cik_overrides.items()}
        unknown = sorted(set(overrides) - set(df["ticker"]))
        if unknown:
            log.warning("cik_overrides for tickers not in universe: %s", unknown)
        df = df.assign(
            cik=[overrides.get(t, c) for t, c in zip(df["ticker"], df["cik"], strict=True)]
        )
    log.info("Universe: %d companies (source=%s)", len(df), cfg.source)
    return df.reset_index(drop=True)

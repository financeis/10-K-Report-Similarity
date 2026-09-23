import numpy as np
import pandas as pd

from tenksim.returns import (
    correlation_matrices,
    label_peers,
    peer_correlation,
    random_baseline,
    residualize,
    yahoo_symbol,
)


def synthetic_prices(n_days: int = 400, seed: int = 0) -> pd.DataFrame:
    """A, B는 시장 외에 공통 '산업' 요인을 공유하고, C는 시장만 따른다."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n_days)
    industry = rng.normal(0, 0.01, n_days)
    rets = {
        "SPY": market,
        "A": 1.2 * market + industry + rng.normal(0, 0.005, n_days),
        "B": 0.8 * market + industry + rng.normal(0, 0.005, n_days),
        "C": 1.0 * market + rng.normal(0, 0.01, n_days),
    }
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in rets.items()}, index=idx)


def test_yahoo_symbol():
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("AAPL") == "AAPL"


def test_residualize_removes_market_beta():
    prices = synthetic_prices()
    rets = prices.pct_change(fill_method=None).iloc[1:]
    resid = residualize(rets[["A", "C"]], rets["SPY"], min_obs=50)
    assert abs(np.corrcoef(resid["C"], rets["SPY"])[0, 1]) < 1e-6
    assert rets["C"].corr(rets["SPY"]) > 0.5


def test_residual_correlation_isolates_industry_link():
    raw, resid, enough = correlation_matrices(synthetic_prices(), ["A", "B", "C"], "SPY", 100)
    assert enough.all()
    # 원수익률로는 C도 A와 꽤 상관되지만, 시장을 빼면 A-B만 남는다
    assert raw[0, 2] > 0.3
    assert resid[0, 1] > 0.5 and abs(resid[0, 2]) < 0.15


def test_min_obs_marks_short_histories():
    prices = synthetic_prices()
    prices.loc[prices.index[:-50], "B"] = np.nan
    _, _, enough = correlation_matrices(prices, ["A", "B", "C"], "SPY", 100)
    assert enough.tolist() == [True, False, True]


def test_peer_helpers():
    corr = np.array([[1.0, 0.6, 0.1], [0.6, 1.0, 0.2], [0.1, 0.2, 1.0]])
    peers = label_peers(np.array(["x", "x", "y"], dtype=object))
    assert [p.tolist() for p in peers] == [[1], [0], []]
    assert peer_correlation(corr, peers) == 0.6  # 회사 2는 peer가 없어 빠진다
    assert np.isclose(random_baseline(corr), (0.6 + 0.1 + 0.2) / 3)

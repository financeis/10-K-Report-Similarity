import numpy as np
import pandas as pd

from tenksim.returns import (
    EQUAL_WEIGHT,
    correlation_matrices,
    equal_weight_market,
    incremental_effect,
    label_peers,
    peer_correlation,
    random_baseline,
    random_per_firm,
    residualize,
    top_k_within,
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


def test_top_k_within_and_random_per_firm():
    sim = np.array(
        [[1.0, 0.9, 0.5, 0.1], [0.9, 1.0, 0.2, 0.3], [0.5, 0.2, 1.0, 0.8], [0.1, 0.3, 0.8, 1.0]]
    )
    allowed = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    nb = top_k_within(sim, allowed, 1)
    assert [n.tolist() for n in nb] == [[2], [3], [0], [1]]  # 허용된 상대 중에서만 고른다
    corr = np.array(
        [[1.0, 0.5, 0.2, 0.4], [0.5, 1.0, 0.0, 0.6], [0.2, 0.0, 1.0, 0.3], [0.4, 0.6, 0.3, 1.0]]
    )
    np.testing.assert_allclose(random_per_firm(corr, allowed), [0.3, 0.3, 0.1, 0.5])
    np.testing.assert_allclose(random_per_firm(corr)[0], (0.5 + 0.2 + 0.4) / 3)


def make_linked_world(text_matters: bool, n: int = 60, seed: int = 0):
    """산업(레이블)과 별개로 '텍스트로만 보이는 연결'이 수익률에 영향을 주는지 조절한다."""
    rng = np.random.default_rng(seed)
    industry = np.repeat(np.arange(6), n // 6)
    hidden = rng.normal(size=(n, 3))  # 공급망 같은 숨은 연결
    link = hidden @ hidden.T
    same = industry[:, None] == industry[None, :]
    corr = 0.3 * same + (0.05 * link if text_matters else 0) + rng.normal(0, 0.02, (n, n))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    sim = link + rng.normal(0, 0.5, (n, n))
    sim = (sim + sim.T) / 2
    return sim, corr, same, np.ones((n, n), dtype=bool)


def test_incremental_effect_detects_links_beyond_labels():
    sim, corr, same, parent = make_linked_world(text_matters=True)
    r = incremental_effect(sim, corr, same, parent, n_boot=50)
    assert r["lo"] > 0 and r["r2_full"] > r["r2_gics"]
    sim, corr, same, parent = make_linked_world(text_matters=False)
    r = incremental_effect(sim, corr, same, parent, n_boot=50)
    assert r["lo"] < 0 < r["hi"]  # 레이블로 다 설명되면 텍스트의 추가 효과는 0 근처


def test_equal_weight_market_excludes_self():
    rets = pd.DataFrame(
        {"A": [0.01, 0.02, np.nan], "B": [0.03, 0.00, 0.01], "C": [0.05, 0.04, 0.03]}
    )
    m = equal_weight_market(rets)
    np.testing.assert_allclose(m["A"], [0.04, 0.02, 0.02])  # A가 없는 날은 나머지 전부의 평균
    np.testing.assert_allclose(m["B"], [0.03, 0.03, 0.03])


def test_cap_weighted_index_biases_megacap_pairs():
    """지수 비중이 큰 두 종목은 그 지수로 잔차를 내면 서로 음(-)의 상관처럼 보인다."""
    rng = np.random.default_rng(0)
    n_days, n_small = 500, 30
    common = rng.normal(0, 0.01, n_days)
    tech = rng.normal(0, 0.01, n_days)  # 두 대형주만 공유하는 요인
    rets = {"MEGA1": common + tech + rng.normal(0, 0.005, n_days)}
    rets["MEGA2"] = common + tech + rng.normal(0, 0.005, n_days)
    for i in range(n_small):
        rets[f"S{i}"] = common + rng.normal(0, 0.01, n_days)
    frame = pd.DataFrame(rets, index=pd.bdate_range("2025-01-01", periods=n_days))
    small_avg = frame[[f"S{i}" for i in range(n_small)]].mean(axis=1)
    frame["CAPW"] = 0.45 * frame["MEGA1"] + 0.45 * frame["MEGA2"] + 0.10 * small_avg
    prices = 100 * (1 + frame).cumprod()
    tickers = ["MEGA1", "MEGA2", *[f"S{i}" for i in range(n_small)]]
    _, resid_capw, _ = correlation_matrices(prices, tickers, "CAPW", 100)
    _, resid_ew, _ = correlation_matrices(prices, tickers, EQUAL_WEIGHT, 100)
    assert resid_capw[0, 1] < 0  # 실제로는 같은 요인을 공유하는데도
    assert resid_ew[0, 1] > 0.5

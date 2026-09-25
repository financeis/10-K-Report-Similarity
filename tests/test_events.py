"""실적 발표 이벤트 스터디 (earnings.py, relations/events.py): 0일, 창, 모멘트 회귀, 보고서."""

import json
import sys
import types

import numpy as np
import pandas as pd
from test_research import sym, sym_bool

from tenksim import earnings
from tenksim.relations import events, research
from tenksim.returns import residual_returns

# ---------------------------------------------------------------- 0일과 8-K 목록


def test_day_zero_uses_new_york_close_early_close_and_report_date():
    # 2025-01-03(금), 01-06(월), 07-01(화), 07-02(수), 07-03(목, 13:00 조기 폐장), 07-07(월)이 거래일
    calendar = pd.DatetimeIndex(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-07-01", "2025-07-02", "2025-07-03",
         "2025-07-07"]
    )  # fmt: skip
    accepted = pd.Series(
        pd.to_datetime(
            [
                "2025-01-03 11:30",  # 06:30 뉴욕, 장 전 → 그날
                "2025-01-03 20:30",  # 15:30 뉴욕, 장중 → 그날
                "2025-01-03 21:30",  # 16:30 뉴욕, 장 마감 뒤 → 다음 거래일(월)
                "2025-01-04 15:00",  # 토요일 → 월요일
                "2025-07-01 20:30",  # 16:30 뉴욕(서머타임), 장 마감 뒤 → 수요일
                "2025-01-02 12:00",  # 첫 거래일: 전 거래일 종가가 없어 수익률이 없음
                "2025-07-07 21:00",  # 마지막 거래일 장 마감 뒤: 다음 거래일이 기간 밖
                "2025-07-03 17:07",  # 13:07 뉴욕, 조기 폐장일 장 마감 뒤 → 월요일
            ]
        ).tz_localize("UTC")
    )
    pos, early = earnings.day_zero(accepted, calendar)
    assert pos.tolist() == [1, 1, 2, 2, 4, -1, -1, 6] and not early.any()
    assert earnings.session(accepted).tolist() == [
        "pre", "intraday", "after", "intraday", "after", "pre", "after", "after"
    ]  # fmt: skip

    # 보고일이 휴장일이고 접수일보다 1~4일 앞서면 보고일 뒤 첫 거래일 (토요일 발표를 월요일 저녁에 낸 8-K)
    calendar = calendar.insert(3, pd.Timestamp("2025-01-07"))
    accepted = pd.Series(
        pd.to_datetime(
            ["2025-01-06 22:00", "2025-07-07 13:00", "2025-01-06 13:00", "2025-01-07 11:00"]
        ).tz_localize("UTC")
    )
    reports = ["2025-01-04", "2025-06-01", "2025-01-03", "2025-01-04"]
    pos, early = earnings.day_zero(accepted, calendar, reports)
    # 두 번째는 너무 앞선 보고일, 세 번째는 거래일인 보고일(전날 장 마감 뒤 보도자료일 수 있음), 네 번째는
    # 장 전(06:00 뉴욕)에 낸 8-K(그날 아침 보도자료)라 보고일을 쓰지 않는다
    assert pos.tolist() == [2, 7, 2, 3] and early.tolist() == [True, False, False, False]


def fake_edgar(monkeypatch, tables: dict[int, pd.DataFrame | Exception]):
    """edgar.Company(cik).get_filings(...).data.to_pandas()만 흉내 낸다."""

    class Filings:
        def __init__(self, df):
            self.data = types.SimpleNamespace(to_pandas=lambda: df)

    class Company:
        def __init__(self, cik):
            self.cik = cik

        def get_filings(self, **kwargs):
            value = tables[self.cik]
            if isinstance(value, Exception):
                raise value
            return Filings(value)

    monkeypatch.setitem(sys.modules, "edgar", types.SimpleNamespace(Company=Company))


def test_fetch_company_keeps_only_item_202(monkeypatch):
    raw = pd.DataFrame(
        {"accession_number": ["a", "b", "c", "d"], "filing_date": pd.to_datetime(["2025-01-30"] * 4),
         "reportDate": ["2025-01-30"] * 4,
         "acceptanceDateTime": pd.to_datetime(["2025-01-30 21:30"] * 4).tz_localize("UTC"),
         "items": ["2.02,9.01", " 2.02", "7.01", None]}
    )  # fmt: skip
    fake_edgar(monkeypatch, {1: raw, 2: raw.iloc[2:], 3: RuntimeError("timeout")})
    start, end = pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-12-31").date()
    out = earnings.fetch_company(1, start, end)
    assert out["accession_number"].tolist() == ["a", "b"] and set(out["status"]) == {"event"}
    assert out["accepted"].dt.tz is None  # UTC를 시간대 표시 없이 저장
    assert out["accepted"].iloc[0] == pd.Timestamp("2025-01-30 21:30")
    assert earnings.fetch_company(2, start, end)["status"].tolist() == ["none"]
    assert earnings.fetch_company(3, start, end)["status"].tolist() == ["error"]
    # 바뀐 CIK까지 조회해 합치고, 같은 8-K는 한 번만 센다 (cik는 기업 목록의 것)
    both = earnings.fetch_firm(2, start, end, [1])
    assert both["accession_number"].tolist() == ["a", "b"] and set(both["cik"]) == {2}
    assert set(both["filer_cik"]) == {1}
    assert earnings.fetch_firm(1, start, end, [3])["status"].tolist() == ["error"]


def test_load_earnings_filings_caches_and_retries_failures(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(cik, start, end, others):
        calls.append((cik, others))
        if cik == 3:
            return earnings._rows(3, filer_cik=3, status="error", error="timeout")
        if cik == 2:
            return earnings._rows(2, filer_cik=2, status="none")
        return pd.DataFrame(
            {"cik": 1, "filer_cik": 1, "status": "event", "accession_number": ["a1", "a2"],
             "filing_date": ["2025-01-30", "2025-05-01"], "report_date": ["2025-01-30", "2025-05-01"],
             "accepted": pd.to_datetime(["2025-01-30 21:30", "2025-05-01 20:30"]),
             "items": ["2.02,9.01", "2.02"]}
        ).reindex(columns=earnings.COLUMNS)  # fmt: skip

    monkeypatch.setattr(earnings, "fetch_firm", fake_fetch)
    monkeypatch.setattr(earnings, "current_ciks", lambda firms: {1: [9]})
    monkeypatch.setattr(earnings, "require_identity", lambda: None)
    firms = pd.DataFrame({"cik": [1, 2, 3], "ticker": ["A", "B", "C"]})
    path = tmp_path / "e.parquet"
    start, end = pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-12-31").date()
    first = earnings.load_earnings_filings(firms, start, end, path)
    assert sorted(calls) == [(1, [9]), (2, []), (3, [])] and len(first) == 4
    assert str(first["accepted"].dt.tz) == "UTC"
    calls.clear()
    again = earnings.load_earnings_filings(firms, start, end, path)
    assert calls == [(3, [])]  # 받지 못한 회사만 다시 받는다
    assert set(again["status"]) == {"event", "none", "error"}
    assert (again["accepted"].dropna() == first["accepted"].dropna()).all()

    # 예전 형식의 캐시(보고일 없음)는 모두 다시 받는다
    pd.read_parquet(path).drop(columns=["report_date"]).to_parquet(path)
    calls.clear()
    earnings.load_earnings_filings(firms, start, end, path)
    assert sorted(c[0] for c in calls) == [1, 2, 3]


def test_build_event_data_dedupes_and_counts():
    n = 3
    firms = pd.DataFrame({"cik": [10, 20, 30], "ticker": ["A", "B", "C"]})
    pair = research.PairData(firms, np.zeros((n, n)), np.zeros((n, n)), ~np.eye(n, dtype=bool),
                             {}, {"config": "t"})  # fmt: skip
    days = pd.bdate_range("2025-01-02", periods=6)
    resid = pd.DataFrame(np.zeros((5, n)), index=days[1:], columns=["A", "B", "C"])
    resid["C"] = np.nan  # 주가 부족
    utc = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    filings = pd.DataFrame(
        {"cik": [10, 10, 10, 20, 30, 99, 20], "status": ["event"] * 6 + ["none"],
         "accepted": [utc("2025-01-03 14:00"), utc("2025-01-03 15:00"), utc("2025-01-06 22:00"),
                      utc("2025-01-02 12:00"), utc("2025-01-06 14:00"), utc("2025-01-06 14:00"), pd.NaT],
         "report_date": ["2025-01-03", "2025-01-03", "2025-01-06", "2025-01-02", "2025-01-06",
                         "2025-01-06", None]}
    ).reindex(columns=earnings.COLUMNS)  # fmt: skip
    data = events.build_event_data(pair, resid, days, filings)
    info = data.info["events"]
    assert data.events[["firm", "day"]].values.tolist() == [[0, 0], [0, 2]]
    assert info["filings"] == 5  # 분석 대상 밖 회사(99)는 세지 않는다
    assert (info["outside_period"], info["duplicates"], info["no_returns"]) == (1, 1, 1)
    assert info["events"] == 2 and info["announcers"] == 1
    assert info["per_firm"] == {"0": 2, "2": 1}
    assert info["without_events"] == ["B", "C"] and info["few_events"] == ["A"]


# ---------------------------------------------------------------- 창과 모멘트


def test_window_car_and_own_event_overlap():
    resid = np.arange(20, dtype=float).reshape(10, 2)  # 10일 × 2개사
    resid[4, 1] = np.nan
    car = events.window_car(resid, np.array([2, 9, 0]), 0, 1)
    assert car[0].tolist() == [4 + 6, 5 + 7]
    assert np.isnan(car[1]).all()  # 창이 기간 밖
    assert car[2].tolist() == [0 + 2, 1 + 3]
    assert np.isnan(events.window_car(resid, np.array([3]), 0, 1)[0, 1])  # 빠진 날

    own = np.zeros((2, 10), dtype=bool)
    own[1, [3, 7]] = True  # 회사 1이 3일, 7일에 발표 (자기 반응 창 [0, +1])
    hit = events.own_event_overlap(own, np.array([4, 5, 1]), 0, 1)
    # 4일 이벤트 [4,5]: 3일 발표의 [3,4]와 겹침 / 5일 [5,6]: 겹치지 않음 / 1일 [1,2]: [3,4]와 안 겹침
    assert hit[:, 1].tolist() == [True, False, False] and not hit[:, 0].any()
    assert events.own_event_overlap(own, np.array([5]), 0, 2)[0, 1]  # [5,7]은 7일 발표와 겹침


def rows_regression(x, firm, y, valid, marks, names, w):
    """비교용: 이벤트-쌍 목록을 만들고 w_i·w_j만큼 복제해 보통 OLS로 푼다."""
    rows, ys = [], []
    for e, j in zip(*np.nonzero(valid), strict=True):
        d = [marks[k][firm[e], j] for k in names]
        for _ in range(int(w[firm[e]] * w[j])):
            rows.append([1.0, *d, x[e], *(x[e] * v for v in d)])
            ys.append(y[e, j])
    coef = np.linalg.lstsq(np.array(rows, dtype=float), np.array(ys), rcond=None)[0]
    k = len(names) + 1
    return {"x": coef[k], **{name: coef[k + 1 + i] for i, name in enumerate(names)}}


def toy_events(rng, n=12, n_events=40):
    firm = rng.integers(0, n, size=n_events)
    x = rng.normal(size=n_events)
    a = sym_bool(0.3, n, rng)
    b = sym_bool(0.3, n, rng) & ~a
    y = 0.2 * x[:, None] + 0.5 * x[:, None] * a[firm] - 0.3 * x[:, None] * b[firm]
    y = y + 0.1 * rng.normal(size=(n_events, n))
    valid = rng.random((n_events, n)) < 0.8
    valid[np.arange(n_events), firm] = False
    return x, firm, y, valid, {"a": a, "b": b}


def test_spill_ols_matches_event_pair_regression():
    rng = np.random.default_rng(3)
    x, firm, y, valid, marks = toy_events(rng)
    n = len(marks["a"])
    moments = events.event_moments(x, firm, y, valid, n)
    cells = events.make_cells(marks, ~np.eye(n, dtype=bool))
    values = np.stack([moments[m][cells.ii, cells.jj] for m in events.MOMENTS])
    for w in (np.ones(n), rng.integers(0, 3, size=n).astype(float)):
        sums = events.cell_sums(cells, values, w)
        ours = events.spill_ols(sums, cells.patterns, cells.names)
        ref = rows_regression(x, firm, y, valid, marks, ["a", "b"], w)
        for k in ref:
            assert abs(ours[k] - ref[k]) < 1e-8, k
    point = events.spill_ols(events.cell_sums(cells, values), cells.patterns, cells.names)
    assert abs(point["x"] - 0.2) < 0.05 and abs(point["a"] - 0.5) < 0.05
    assert abs(point["b"] + 0.3) < 0.05

    # 묶음 기울기 = 그 묶음 이벤트-쌍만의 단순 회귀 (조건: 있음/없음)
    defs = {"a": [("a", True)], "not_a": [("a", False)], "all": []}
    member = events.group_membership(cells, defs)
    g = events.cell_sums(cells, values) @ member
    xs = np.broadcast_to(x[:, None], y.shape)
    for k, mask in enumerate((marks["a"][firm], ~marks["a"][firm], np.ones_like(valid))):
        m = valid & mask
        assert abs(events.slope(g[:, k]) - np.polyfit(xs[m], y[m], 1)[0]) < 1e-10

    # 쓰이는 쌍이 없는 표시는 '정확히 0'이 아니라 추정할 수 없음(None)
    empty = events.make_cells(
        {**marks, "none": np.zeros((n, n), dtype=bool)}, ~np.eye(n, dtype=bool)
    )
    values = np.stack([moments[m][empty.ii, empty.jj] for m in events.MOMENTS])
    out = events.spill_ols(events.cell_sums(empty, values), empty.patterns, empty.names)
    assert out["none"] is None and abs(out["a"] - point["a"]) < 1e-10


def test_transfer_removes_ordinary_comovement():
    """평소에도 같이 움직이는 쌍: 발표일 기울기에는 그 몫이 섞이고, 늘어난 공분산 ÷ 늘어난 분산은 뉴스 반응만 남긴다."""
    rng = np.random.default_rng(1)
    k = 20000
    common = rng.normal(0, 0.02, k)
    news = rng.normal(0, 0.05, k)
    x_normal, y_normal = common + rng.normal(0, 0.01, k), 0.8 * common + rng.normal(0, 0.01, k)
    x_event, y_event = x_normal + news, y_normal + 0.2 * news

    def sums(x, y):
        return np.array([len(x), x.sum(), (x * x).sum(), y.sum(), (x * y).sum()])

    event, normal = sums(x_event, y_event), sums(x_normal, y_normal)
    assert events.slope(event) > 0.25  # 뉴스 반응 0.2에 평소 동조성이 더해짐
    assert abs(events.transfer(event, normal) - 0.2) < 0.02


def test_announcer_out_removes_mechanical_market_effect():
    """상대의 시장 평균에 발표 회사가 들어 있으면, 관계가 없어도 기울기가 −β/K쯤 나온다."""
    rng = np.random.default_rng(0)
    t, n = 200, 25
    rets = rng.normal(0, 0.01, size=(t, n)) + rng.normal(0, 0.01, size=(t, 1))
    days = rng.integers(5, t - 5, size=60)
    firm = rng.integers(0, n, size=60)
    rets[days, firm] += rng.normal(0, 0.2, size=60)  # 발표 회사만 크게 움직임
    prices = pd.DataFrame(100 * np.cumprod(1 + np.vstack([np.zeros(n), rets]), axis=0))
    prices.columns = [f"T{i}" for i in range(n)]
    stock, resid, enough = residual_returns(prices, list(prices.columns), "equal_weight", 100)
    market = events.market_parts(stock, resid, enough)
    r = resid.to_numpy()
    x = r[days, firm]
    raw = events.window_car(r, days, 0, 0)
    fixed = raw + events.announcer_out(market, firm, days, 0, 0)
    valid = np.ones_like(raw, dtype=bool)
    valid[np.arange(len(days)), firm] = False
    xs = np.broadcast_to(x[:, None], raw.shape)[valid]
    before = np.polyfit(xs, raw[valid], 1)[0]
    after = np.polyfit(xs, fixed[valid], 1)[0]
    assert before < -0.5 / (n - 1)  # 기계적으로 음수 (β가 1에 가깝고 K = n−1)
    assert abs(after) < 0.25 * abs(before)

    # 정확히: 발표 회사를 뺀 시장으로 다시 낸 잔차 (α, β는 그대로)
    ret, mk, e = stock.to_numpy(), market.market, r
    for k in range(10):
        i, d = firm[k], days[k]
        for j in range(n):
            if j == i:
                continue
            alpha = np.nanmean(ret[:, j] - market.beta[j] * mk[:, j])
            others = market.others[d, j]
            m_wo = (mk[d, j] * others - ret[d, i]) / (others - 1)
            expected = ret[d, j] - alpha - market.beta[j] * m_wo
            assert abs(fixed[k, j] - expected) < 1e-12
        assert abs(e[d, 0] - (ret[d, 0] - np.nanmean(ret[:, 0] - market.beta[0] * mk[:, 0])
                              - market.beta[0] * mk[d, 0])) < 1e-12  # fmt: skip


def test_winsorize_only_inside_mask():
    v = np.array([1.0, 2.0, 3.0, 100.0, -50.0])
    mask = np.array([True, True, True, True, False])
    out = events.winsorize(v, mask, 0.25)
    assert out[4] == -50.0 and out[3] < 100.0 and out[0] > 1.0


# ---------------------------------------------------------------- 분석과 보고서


def synthetic(n=36, t=220, seed=7, spill=0.3) -> events.EventData:
    """가격에서 시작한 합성 자료. 경쟁 쌍은 발표 회사 소식에 spill만큼, 같은 섹터는 0.1만큼 같은 방향으로
    반응한다. 같은 섹터는 평소에도 섹터 요인으로 같이 움직인다(평소 동조성)."""
    rng = np.random.default_rng(seed)
    sectors = np.array(["A", "B", "C"])[np.arange(n) % 3]
    subs = np.array([f"{s}{(i // 3) % 2}" for i, s in enumerate(sectors)])
    same = (sectors[:, None] == sectors[None, :]) & ~np.eye(n, dtype=bool)
    same_sub = subs[:, None] == subs[None, :]
    sim = sym(rng.random((n, n)) * 100)
    comp = sym_bool(0.08, n, rng)
    biz = sym_bool(0.08, n, rng) & ~comp
    z = np.zeros((n, n), dtype=bool)
    firms = pd.DataFrame(
        {"cik": np.arange(n) + 1, "node_id": [f"cik:{i + 1}" for i in range(n)],
         "ticker": [f"T{i}" for i in range(n)], "name": [f"Firm {i}" for i in range(n)],
         "gics_sector": sectors, "gics_sub_industry": subs}
    )  # fmt: skip
    pairs = {
        "competitor": comp, "business": biz, "equity": z, "uncertain": z,
        "mentioned": comp | biz | (other := sym_bool(0.05, n, rng)),
        "entity_confirmed": comp | biz | other, "top20": sim > 90,
        "similar_only": (sim > 90) & ~comp & ~biz, "same_sub": same_sub,
        "same_sector": same | np.eye(n, dtype=bool), "gics_known": np.ones((n, n), dtype=bool),
    }  # fmt: skip
    info = {
        "config": "t", "filings_year": 2024, "start": "2025-01-01", "end": "2025-12-31",
        "similarity": "s", "top_k": 20, "graph_created": "x", "judge_model": "m",
        "question_version": "v",
    }  # fmt: skip
    pair = research.PairData(
        firms, sym(rng.random((n, n))), sim, ~np.eye(n, dtype=bool), pairs, info
    )
    sector_factor = rng.normal(0, 0.008, size=(t, 3))[:, np.arange(n) % 3]
    rets = rng.normal(0, 0.01, size=(t, 1)) + sector_factor + rng.normal(0, 0.01, size=(t, n))
    firm = np.repeat(np.arange(n), 4)
    day = np.concatenate(
        [rng.choice(np.arange(25, t - 25, 8), size=4, replace=False) for _ in range(n)]
    )
    for f, d, s in zip(firm, day, rng.normal(0, 0.05, size=len(firm)), strict=True):
        rets[d, f] += s
        rets[d] += s * (spill * comp[f] + 0.1 * same[f])
    days_index = pd.bdate_range("2025-01-02", periods=t + 1)
    prices = pd.DataFrame(100 * np.cumprod(1 + np.vstack([np.zeros(n), rets]), axis=0),
                          index=days_index, columns=firms["ticker"])  # fmt: skip
    stock, resid, enough = residual_returns(prices, list(firms["ticker"]), "equal_weight", 150)
    filings = pd.DataFrame(
        {"cik": firms["cik"].to_numpy()[firm], "status": "event",
         "accepted": [pd.Timestamp(days_index[d + 1]).tz_localize("UTC") + pd.Timedelta(hours=12)
                      for d in day]}
    ).reindex(columns=earnings.COLUMNS)  # fmt: skip
    data = events.build_event_data(pair, resid, days_index, filings)
    data.market = events.market_parts(stock, resid, enough)
    assert data.events["day"].sort_values().tolist() == sorted(day.tolist())
    return data


def test_analyze_and_report():
    data = synthetic()
    result = events.analyze(data, n_boot=30)
    json.dumps(result)  # 결과 파일로 저장할 수 있어야 한다
    main = result["specs"][events.MAIN]
    g = main["groups"]
    assert abs(g["competitor"]["slope"] - 0.3) < 0.12
    assert g["competitor"]["lo"] < g["competitor"]["slope"] < g["competitor"]["hi"]
    assert abs(g["all"]["slope"]) < 0.02  # 시장에서 발표 회사를 빼면 계산상 0 근처
    coef = main["regression"]["text"]["coefs"]
    assert abs(coef["competitor"]["coef"] - 0.3) < 0.12
    assert coef["equity"]["coef"] is None  # 쌍이 없는 묶음
    # 평소 날: 같은 섹터는 섹터 요인으로 같이 움직이고, 경쟁 효과는 발표일에만 있다
    placebo = result["specs"]["placebo"]["groups"]
    assert (
        placebo["same_sub"]["slope"] > 0.2
        and abs(placebo["competitor"]["slope"]) < placebo["same_sub"]["slope"]
    )
    assert abs(result["transfer"]["competitor"]["diff"] - 0.3) < 0.12
    assert result["transfer"]["same_sub"]["diff"] < g["same_sub"]["slope"]
    # GICS 위치별 묶음과 짝지은 차이
    assert (
        g["competitor@same_sub"]["n_event_pairs"]
        + g["competitor@same_sector"]["n_event_pairs"]
        + g["competitor@other_sector"]["n_event_pairs"]
        == g["competitor"]["n_event_pairs"]
    )
    assert main["contrasts"]["competitor_vs_unrelated@other_sector"]["diff"] > 0.15
    assert result["exclusion"]["share"]["all"] > 0
    assert (
        main["n_events"] > 0
        and result["specs"]["drift"]["groups"]["competitor"]["slope"] is not None
    )
    prof = result["profile"]["by_day"]
    assert prof["0"] > 2 * prof["1"]  # 0일에 뉴스를 심었다
    assert result["examples"]["competitor"]

    text = events.format_report(result)
    for heading in ("## 1. 요약", "## 3. 관계 유형별 반응 계수", "## 4. GICS 위치별", "## 7. 통제 회귀",
                    "## 10. 해석할 때"):  # fmt: skip
        assert heading in text
    assert "**결론:**" in text and "Firm" in text and "-0.000" not in text
    assert (
        "GICS·텍스트 유사도를 통제해도 실적 발표 반응을 더 설명합니다" in text
    )  # 심은 효과를 찾음

    null = events.format_report(events.analyze(synthetic(spill=0.0), n_boot=30))
    assert "이미 설명합니다" in null


def test_spec_filters():
    data = synthetic()
    _, firm, _, valid = events.event_arrays(data, events.SPECS["d01"])
    _, _, _, solo = events.event_arrays(data, events.SPECS["solo"])
    assert solo.sum() < valid.sum() and (solo <= valid).all()
    ev = data.events
    same = data.pair.pairs["same_sub"] & ~np.eye(len(data.pair.firms), dtype=bool)
    for e in np.flatnonzero(solo.any(axis=1)):
        others = ev[(ev["firm"] != firm[e]) & same[firm[e], ev["firm"]]]
        assert not (abs(others["day"] - ev["day"].iloc[e]) <= 1).any()

    x, _, _, big = events.event_arrays(data, events.SPECS["big"])
    assert (np.abs(x[big.any(axis=1)]) >= 0.05).all()
    # 가짜 발표일: 발표 회사 자신의 발표 창과 겹치지 않는다
    spec = events.SPECS["placebo"]
    _, pfirm, _, pvalid = events.event_arrays(data, spec)
    pdays = np.concatenate([ev["day"].to_numpy() + s for s in spec.placebo])
    pdays = pdays[(pdays >= 0) & (pdays < data.resid.shape[0])]
    for e in np.flatnonzero(pvalid.any(axis=1)):
        own = ev.loc[ev["firm"] == pfirm[e], "day"].to_numpy()
        assert not (abs(own - pdays[e]) <= 1).any()

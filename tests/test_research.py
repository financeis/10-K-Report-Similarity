"""관계 유형별 주가 동조성 (relations/research.py): 기업쌍 회귀, 쌍 평균, 쌍 묶음, 보고서."""

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
from test_export import export  # tests/는 rootdir 기준으로 import

from tenksim.relations import research, reviews
from tenksim.relations.stages import edge_evidence_texts


def sym(a: np.ndarray) -> np.ndarray:
    return np.triu(a, 1) + np.triu(a, 1).T


def sym_bool(p: float, n: int, rng) -> np.ndarray:
    a = np.triu(rng.random((n, n)) < p, 1)
    return a | a.T


def pair_list(y, covariates, mask, weights, firm_effects):
    """비교용: 쌍 목록을 만들고 (i<j) 가중치만큼 복제해 보통 OLS로 푼다."""
    n = len(y)
    rows, ys = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if not mask[i, j]:
                continue
            x = [c[i, j] for c in covariates.values()]
            fe = np.zeros(n)
            fe[[i, j]] = 1
            design = [*x, *fe] if firm_effects else [1.0, *x]
            for _ in range(int(weights[i] * weights[j])):
                rows.append(design)
                ys.append(y[i, j])
    coef = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    names = list(covariates) if firm_effects else ["const", *covariates]
    return dict(zip(names, coef[: len(names)], strict=True))


def toy(rng):
    """회사 효과(α_i + α_j)가 있는 쌍 자료: y = 0.5·x1 + 0.3·x2 + α_i + α_j + 잡음."""
    n = 25
    alpha = rng.normal(size=n)
    x1, x2 = sym(rng.normal(size=(n, n))), sym_bool(0.2, n, rng).astype(float)
    y = 0.5 * x1 + 0.3 * x2 + alpha[:, None] + alpha[None, :] + sym(0.1 * rng.normal(size=(n, n)))
    mask = ~np.eye(n, dtype=bool)
    mask[0, 1] = mask[1, 0] = False
    return y, {"x1": x1, "x2": x2}, mask


def test_dyadic_ols_matches_pair_list_regression():
    rng = np.random.default_rng(1)
    y, cov, mask = toy(rng)
    n = len(y)
    ones = np.ones(n)
    drawn = rng.integers(0, 3, size=n).astype(float)  # 재표집: 회사마다 뽑힌 횟수 (0이면 빠짐)
    for weights in (None, drawn):
        w = ones if weights is None else weights
        for fe in (False, True):
            ours = research.dyadic_ols(y, cov, mask, weights=weights, firm_effects=fe)
            ref = pair_list(y, cov, mask, w, fe)
            assert set(ours) == set(ref)
            for k in ref:
                assert abs(ours[k] - ref[k]) < 1e-8, (weights is None, fe, k)
    fe = research.dyadic_ols(y, cov, mask, firm_effects=True)
    assert abs(fe["x1"] - 0.5) < 0.05 and abs(fe["x2"] - 0.3) < 0.05  # 회사 효과를 흡수한다

    # 쌍이 하나도 없는 묶음은 '정확히 0'이 아니라 추정할 수 없음(None)
    empty = {**cov, "none": np.zeros((n, n))}
    for fe_on in (False, True):
        out = research.dyadic_ols(y, empty, mask, firm_effects=fe_on)
        full = research.dyadic_ols(y, cov, mask, firm_effects=fe_on)
        assert out["none"] is None and abs(out["x1"] - full["x1"]) < 1e-10
    reg = research.pair_regression(y, empty, mask, research.boot_weights(n, 5), firm_effects=True)
    assert reg["coefs"]["none"] == {"coef": None, "lo": None, "hi": None, "n_boot": 0}


def test_group_means_are_pair_means_with_company_bootstrap():
    rng = np.random.default_rng(2)
    n = 30
    y = sym(rng.normal(size=(n, n)))
    np.fill_diagonal(y, 1.0)
    g = sym_bool(0.3, n, rng)
    empty = np.zeros((n, n), dtype=bool)
    out = research.group_means(y, {"g": g, "none": empty}, research.boot_weights(n, 300))
    upper = np.triu(g, 1)
    assert out["g"]["n_pairs"] == upper.sum()
    assert abs(out["g"]["mean"] - y[upper].mean()) < 1e-12  # 대각(자기 자신)은 들어가지 않는다
    assert out["g"]["lo"] < out["g"]["mean"] < out["g"]["hi"]
    assert out["none"] == {"n_pairs": 0, "n_firms": 0, "mean": None, "lo": None, "hi": None}
    w = research.boot_weights(n, 5, seed=3)
    assert w.shape == (5, n) and (w.sum(1) == n).all()

    other = sym_bool(0.3, n, rng) & ~g
    weights = research.boot_weights(n, 300)
    c = research.contrast(y, g, other, weights)
    assert abs(c["diff"] - (y[upper].mean() - y[np.triu(other, 1)].mean())) < 1e-12
    assert c["lo"] < c["diff"] < c["hi"]
    same = research.contrast(y, g, g, weights)
    assert same["diff"] == 0 and same["lo"] == same["hi"] == 0  # 같은 재표집으로 짝지어 잰다
    assert research.contrast(y, g, empty, weights)["diff"] is None


def test_pair_groups_keep_judged_out_mentions_separate():
    n = 4
    z = np.zeros((n, n), dtype=bool)

    def pair(i, j):
        m = z.copy()
        m[i, j] = m[j, i] = True
        return m

    pairs = {
        "competitor": pair(0, 1), "business": pair(0, 1) | pair(1, 2), "equity": z,
        "uncertain": pair(1, 2), "mentioned": pair(0, 1) | pair(1, 2) | pair(2, 3) | pair(0, 3) | pair(1, 3),
        "entity_confirmed": pair(0, 1) | pair(1, 2) | pair(2, 3), "similar_only": pair(0, 2),
        "same_sub": pair(0, 1),
    }  # fmt: skip
    usable = ~np.eye(n, dtype=bool)
    usable[0, 3] = usable[3, 0] = False  # 주가가 없는 쌍
    groups = research.pair_groups(pairs, usable)
    assert groups["competitor"][0, 1] and groups["business"][0, 1]  # 관계 유형은 겹칠 수 있다
    # 이미 보이는 관계가 있는 쌍은 불확실로 세지 않는다
    assert not groups["uncertain"].any()
    # 관계로 채택되지 않은 언급: 판정 모델이 그 회사로 확인했으면 '회사 언급', 아니면 '이름만 겹침'
    assert groups["mention_only"][2, 3] and groups["mention_only"].sum() == 2
    assert groups["name_only"][1, 3] and groups["name_only"].sum() == 2  # (0,3)은 분석 대상 밖
    assert groups["all"].sum() == usable.sum()


def test_pair_percentile_and_similarity_bins():
    rng = np.random.default_rng(4)
    n = 40
    sim = sym(rng.random((n, n)))
    usable = ~np.eye(n, dtype=bool)
    pct = research.pair_percentile(sim, usable)
    assert np.allclose(pct, pct.T, equal_nan=True) and np.isnan(np.diag(pct)).all()
    upper = np.triu(usable, 1)
    assert pct[upper].min() == 0 and pct[upper].max() == 100
    bins = research.similarity_bins(pct)
    assert list(bins) == [f"sim_{b:g}" for b in research.SIM_BINS]
    stacked = np.stack(list(bins.values())).sum(0)
    assert (stacked[upper] == (pct[upper] >= 50)).all()  # 50 이상은 정확히 한 구간, 50 미만은 기준


def synthetic(n=40, seed=5) -> research.PairData:
    """경쟁 쌍은 유사도가 높고, 유사도가 높을수록 주가가 같이 움직인다 (관계 자체의 효과는 없음)."""
    rng = np.random.default_rng(seed)
    sectors = np.array(["A", "B", "C", "D"])[np.arange(n) % 4]
    sim = sym(rng.random((n, n)) * 100)
    comp = sym_bool(0.05, n, rng) & (sim > 80)
    biz = sym_bool(0.05, n, rng)
    corr = sym(
        0.004 * sim + 0.1 * (sectors[:, None] == sectors[None, :]) + 0.05 * rng.normal(size=(n, n))
    )
    np.fill_diagonal(corr, 1.0)
    same = sectors[:, None] == sectors[None, :]
    z = np.zeros((n, n), dtype=bool)
    firms = pd.DataFrame(
        {"node_id": [f"cik:{i}" for i in range(n)], "ticker": [f"T{i}" for i in range(n)],
         "name": [f"Firm {i}" for i in range(n)], "gics_sector": sectors, "gics_sub_industry": sectors}
    )  # fmt: skip
    pairs = {
        "competitor": comp, "business": biz, "equity": z, "uncertain": z,
        "mentioned": comp | biz | (other := sym_bool(0.1, n, rng)),
        "entity_confirmed": comp | biz | (other & (sim > 50)),
        "top20": sim > 90, "similar_only": (sim > 90) & ~comp,
        "same_sub": same, "same_sector": same, "gics_known": np.ones((n, n), dtype=bool),
    }  # fmt: skip
    info = {
        "config": "t", "filings_year": 2024, "start": "2025-01-01", "end": "2025-12-31",
        "market": "equal_weight", "min_obs": 150, "similarity": "s", "top_k": 20, "n_firms": n,
        "n_firms_similarity": n, "n_pairs": n * (n - 1) // 2, "graph_created": "x", "judge_model": "m",
        "question_version": "v", "validated": ["competitor", "business"],
        "coverage": {r: {"shown": 3, "used": 1, "external": 1, "outside_universe": 1, "no_returns": 0}
                     for r in research.RELATIONS},
        "outside_companies": ["INTC"], "reviews": {},
    }  # fmt: skip
    return research.PairData(firms, corr, sim, ~np.eye(n, dtype=bool), pairs, info)


def test_analyze_and_report():
    data = synthetic()
    result = research.analyze(data, n_boot=40, n_boot_regression=10)
    json.dumps(result)  # 결과 파일로 저장할 수 있어야 한다
    g = result["groups"]
    assert g["competitor"]["mean"] > g["all"]["mean"]  # 유사도가 높은 쌍이라 같이 움직인다
    assert g["competitor"]["median_similarity_pct"] > 60
    reg = result["regression"]
    assert set(reg) == {"gics", "text", "firm", "linear", "linear_firm"}
    assert "const" not in reg["firm"]["coefs"] and "sim_90" in reg["text"]["coefs"]
    # 관계 자체의 효과는 없으므로, 유사도를 통제하면 경쟁 계수가 작아진다
    assert (
        abs(reg["text"]["coefs"]["competitor"]["coef"]) < reg["gics"]["coefs"]["competitor"]["coef"]
    )
    ex = result["examples"]
    assert ex["negative_competitors"][0]["corr"] <= ex["negative_competitors"][-1]["corr"]

    assert g["name_only"]["n_pairs"] > 0 and g["mention_only"]["n_pairs"] > 0
    # 쌍이 없는 묶음(지분·불확실)은 회귀표에 '정확히 0'이 아니라 '–'로 나온다
    assert reg["firm"]["coefs"]["equity"]["coef"] is None

    text = research.format_report(result)
    equity_row = next(line for line in text.splitlines() if line.startswith("| 지분 (검증 전) | –"))
    assert "+0.000" not in equity_row
    for heading in (
        "## 1. 요약",
        "## 3. GICS 위치별",
        "## 5. 기업쌍 회귀",
        "## 7. 해석할 때 주의할 점",
    ):
        assert heading in text
    assert "유사도를 직선 하나로만 넣으면" in text and "Firm" in text
    assert "2/3 (외부 기업 1, 분석 대상 밖 S&P 500 1)" in text and "(이번에는 INTC)" in text


def test_coverage_counts_each_reason_for_dropping_an_edge():
    pos = {"cik:1": 0, "cik:2": 1, "cik:3": 2}
    kind = {**dict.fromkeys(pos, "company"), "cik:9": "company", "ext:x": "external"}
    usable = ~np.eye(3, dtype=bool)
    usable[0, 2] = usable[2, 0] = False  # 주가 관측치 부족
    edges = pd.DataFrame(
        {"src": ["cik:1", "cik:1", "cik:1", "cik:1"], "dst": ["cik:2", "cik:3", "cik:9", "ext:x"]}
    )
    out = research._coverage(edges, kind, pos, usable)
    assert out == {"shown": 4, "used": 1, "external": 1, "outside_universe": 1, "no_returns": 1}


def test_verdict_reads_all_three_models():
    def c(lo, hi):
        return {"coef": (lo + hi) / 2, "lo": lo, "hi": hi}

    def reg(gics, text, firm):
        return {
            m: {"coefs": {"x": e}} for m, e in zip(research.MODELS, (gics, text, firm), strict=True)
        }

    assert "사라집니다" in research._verdict(reg(c(0.1, 0.2), c(-0.1, 0.1), c(-0.1, 0.1)), "x")
    assert "GICS만 통제해도" in research._verdict(
        reg(c(-0.1, 0.2), c(-0.1, 0.1), c(-0.1, 0.1)), "x"
    )
    assert "회사 고정효과를 더하면" in research._verdict(
        reg(c(0.1, 0.2), c(0.1, 0.2), c(-0.1, 0.1)), "x"
    )
    assert "뚜렷하게 높습니다" in research._verdict(reg(c(0.1, 0.2), c(0.1, 0.2), c(0.1, 0.2)), "x")
    none = {"coef": None, "lo": None, "hi": None}
    assert "추정할 수 없는" in research._verdict(reg(none, none, none), "x")


def test_shown_edges_apply_latest_reviews(tmp_path):
    # 웹앱 테스트의 관계(UPS–Amazon 채택, UPS–Samsung 불확실)를 쓴다. fastapi가 없으면 건너뛴다
    from test_relations_app import BUSINESS, UNCERTAIN, relations

    path = export(tmp_path / "graph.db", relations=relations)
    con = sqlite3.connect(path)
    cfg = SimpleNamespace(relations_dir=tmp_path)
    before = research.shown_edges(cfg, con).set_index("edge_id")["state"]
    assert before[BUSINESS] == "accepted" and before[UNCERTAIN] == "uncertain"

    evidence = pd.read_sql("SELECT edge_id, span_id FROM edge_evidence", con)
    texts = edge_evidence_texts(con, evidence)
    edges = pd.read_sql("SELECT * FROM edges", con).set_index("edge_id")
    rev = reviews.connect(tmp_path / "reviews.sqlite")
    for edge_id, verdict in ((BUSINESS, "reject"), (UNCERTAIN, "accept")):
        reviews.record_edge_review(
            rev, edge={**edges.loc[edge_id].to_dict(), "edge_id": edge_id},
            evidence_hash=reviews.evidence_hash(texts[edge_id]), verdict=verdict,
        )  # fmt: skip
    rev.close()
    after = research.shown_edges(cfg, con).set_index("edge_id")["state"]
    assert after[BUSINESS] == "rejected" and after[UNCERTAIN] == "confirmed"  # export 없이 반영

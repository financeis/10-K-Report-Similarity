"""네트워크 없이 도는 end-to-end 테스트: 가짜 10-K 텍스트와 가짜 임베딩 모델을 쓴다."""

import json

import numpy as np
import pandas as pd
import pytest

from tenksim import pipeline
from tenksim.config import Config
from tenksim.embed import DenseEmbedder
from tenksim.report import write_report

TOPICS = {
    "semis": "semiconductor chips wafer foundry gpu datacenter accelerator",
    "banks": "deposits loans mortgage branch banking credit interest",
    "oil": "crude oil gas drilling refinery upstream pipeline barrels",
}


class FakeEmbedder(DenseEmbedder):
    """단어마다 고정 난수 벡터를 두고 평균내는 결정적 모델."""

    model_id = "fake:bow"
    max_tokens = 30
    batch_size = 8

    def __init__(self):
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _embed(self, texts):
        self.calls += len(texts)
        out = []
        for t in texts:
            vecs = [np.random.default_rng(abs(hash(w)) % 2**32).normal(size=16) for w in t.split()]
            out.append(np.mean(vecs, axis=0))
        return np.array(out)


def make_inputs():
    rows, recs = [], []
    rng = np.random.default_rng(0)
    for i in range(9):
        topic = list(TOPICS)[i % 3]
        cik = 1000 + i
        rows.append(
            {
                "cik": cik,
                "ticker": f"T{i}",
                "name": f"Co {i}",
                "gics_sector": topic,
                "gics_sub_industry": f"{topic}-sub",
            }
        )
        words = TOPICS[topic].split() + ["company", "products", "customers", "employees"]
        paragraphs = [" ".join(rng.choice(words, size=25)) + "." for _ in range(12)]
        text = "Item 1. Business\n" + "\n".join(paragraphs) + "\n12\n"
        recs.append(
            {
                "cik": cik,
                "year": 2024,
                "section": "business",
                "status": "fetched",
                "error": None,
                "company": f"Co {i}",
                "sic": str(3000 + 100 * (i % 3)),
                "text_raw": text,
            }
        )
    # 추출 실패 사례 하나 (GE처럼 주석을 잘못 가져온 경우)
    rows.append(
        {
            "cik": 2000,
            "ticker": "BAD",
            "name": "Bad Co",
            "gics_sector": "oil",
            "gics_sub_industry": "oil-sub",
        }
    )
    recs.append(
        {
            "cik": 2000,
            "year": 2024,
            "section": "business",
            "status": "fetched",
            "error": None,
            "company": "Bad Co",
            "sic": "2911",
            "text_raw": "NOTE 2. BUSINESSES HELD FOR SALE\n" + "numbers " * 500,
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(recs)


@pytest.fixture
def cfg(tmp_path):
    return Config.model_validate(
        {
            "name": "t",
            "data_dir": str(tmp_path / "data"),
            "reports_dir": str(tmp_path / "reports"),
            "filings": {"year": 2024},
            "text": {"min_chars": 200},
            "methods": [
                {"name": "tfidf", "kind": "tfidf", "max_df": 0.9, "min_df": 1},
                {"name": "fake", "kind": "sbert", "model": "fake"},
            ],
            "ensembles": [{"name": "ens", "members": ["tfidf", "fake+center"]}],
            "evaluation": {"k": [1, 2], "labels": ["gics_sector", "sic2"], "n_boot": 100},
        }
    )


def test_end_to_end_offline(cfg, monkeypatch):
    fake = FakeEmbedder()
    real_build = pipeline.build_method
    monkeypatch.setattr(
        pipeline, "build_method", lambda m: fake if m.kind == "sbert" else real_build(m)
    )
    universe, records = make_inputs()
    cfg.run_dir.mkdir(parents=True)
    universe.to_parquet(cfg.run_dir / "universe.parquet", index=False)

    docs = pipeline.stage_documents(cfg, universe, records)
    assert docs.set_index("ticker").loc["BAD", "status"] == "suspect"
    assert not docs["text"].str.contains("Item 1. Business").any()

    results = pipeline.stage_methods(cfg, universe, docs)
    assert 2000 not in set(results["fake"].companies["cik"])
    assert results["fake"].chunks["n_tokens"].max() <= FakeEmbedder.max_tokens
    first_calls = fake.calls

    metrics = pipeline.stage_evaluate(cfg, universe, docs, pipeline.load_methods(cfg))
    assert metrics["n_companies"] == 9
    assert set(metrics["labels"]) == {"tfidf", "fake", "fake+center", "ens"}
    ci = metrics["labels_ci"]["p@2"]
    assert ci["tfidf"]["lo"] <= ci["tfidf"]["mean"] <= ci["tfidf"]["hi"]
    assert "diff" in ci["ens"] and "diff" not in ci["tfidf"]  # 기준 방법 자신은 차이가 없다
    for variant in metrics["labels"].values():
        assert variant["gics_sector"]["auc"] > 0.9  # 주제 단어가 분명해서 쉽게 맞혀야 한다
    assert (
        json.loads((cfg.run_dir / "metrics.json").read_text(encoding="utf-8"))["n_companies"] == 9
    )

    report = write_report(cfg, metrics, docs, universe).read_text(encoding="utf-8")
    for heading in (
        "## 1. 데이터 품질",
        "## 4. 산업분류 재현",
        "## 5. 방법 간 일치도",
        "## 6. 예시",
    ):
        assert heading in report
    assert "BAD" in report and "suspect" in report

    # 다시 돌리면 캐시에서 읽어 모델을 부르지 않는다
    pipeline.stage_methods(cfg, universe, docs, only=["fake"])
    assert fake.calls == first_calls


def test_too_few_companies_is_an_error(cfg):
    universe, records = make_inputs()
    cfg.run_dir.mkdir(parents=True)
    docs = pipeline.stage_documents(cfg, universe.head(2), records.head(2))
    with pytest.raises(RuntimeError, match="분석 가능한 회사"):
        pipeline.stage_methods(cfg, universe.head(2), docs)


def test_leading_chunks_budget():
    chunks = [("a", 10), ("b", 10), ("c", 10)]
    assert pipeline._leading_chunks(chunks, 25) == [("a", 10), ("b", 10)]
    assert pipeline._leading_chunks(chunks, 5) == [("a", 10)]  # 최소 1개는 남긴다


def test_doc_tokens_limits_chunks_per_company(cfg, monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(pipeline, "build_method", lambda m: fake)
    cfg.methods[1].doc_tokens = 60
    cfg.methods = [cfg.methods[1]]
    universe, records = make_inputs()
    cfg.run_dir.mkdir(parents=True)
    docs = pipeline.stage_documents(cfg, universe, records)
    result = pipeline.stage_methods(cfg, universe, docs)["fake"]
    per_company = result.chunks.groupby("cik")["n_tokens"].sum()
    assert (per_company <= 60).all()


def test_overlapping_sections_are_both_excluded(cfg):
    # Devon Energy처럼 Item 1 자리에 Item 1A의 일부가 들어온 경우
    risk = "Item 1A. Risk Factors\n" + "\n".join(
        f"Risk {i}: our results could be adversely affected by events we cannot control."
        for i in range(40)
    )
    business = "Item 1. Business\n" + "\n".join(risk.split("\n")[1:30])
    universe = pd.DataFrame(
        [
            {
                "cik": 1,
                "ticker": "DVN",
                "name": "Devon",
                "gics_sector": "Energy",
                "gics_sub_industry": "E&P",
            }
        ]
    )
    base = {
        "cik": 1,
        "year": 2024,
        "status": "fetched",
        "error": None,
        "company": "Devon",
        "sic": "1311",
    }
    records = pd.DataFrame(
        [
            {**base, "section": "business", "text_raw": business},
            {**base, "section": "risk_factors", "text_raw": risk},
        ]
    )
    cfg.filings.sections = ["business", "risk_factors"]
    cfg.run_dir.mkdir(parents=True)
    docs = pipeline.stage_documents(cfg, universe, records).set_index("section")
    assert docs.loc["business", "note"] == "duplicate:risk_factors"
    assert docs.loc["risk_factors", "note"] == "duplicate:business"
    assert (docs["status"] == "suspect").all()


def fake_prices(tickers, market, start, end, cache_path):
    """같은 주제(산업) 회사끼리 공통 요인을 갖는 가짜 주가."""
    rng = np.random.default_rng(0)
    n = 250
    market_ret = rng.normal(0, 0.01, n)
    factors = {t: rng.normal(0, 0.01, n) for t in TOPICS}
    data = {market: market_ret}
    for i, t in enumerate(tickers):
        data[t] = market_ret + factors[list(TOPICS)[i % 3]] + rng.normal(0, 0.01, n)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in data.items()}, index=index)


def test_returns_and_beyond_gics_sections(cfg, monkeypatch):
    from datetime import date

    from tenksim.config import ReturnsConfig

    fake = FakeEmbedder()
    real_build = pipeline.build_method
    monkeypatch.setattr(
        pipeline, "build_method", lambda m: fake if m.kind == "sbert" else real_build(m)
    )
    monkeypatch.setattr(pipeline, "load_prices", fake_prices)
    cfg.evaluation.returns = ReturnsConfig(
        start=date(2025, 1, 1), end=date(2025, 12, 31), min_obs=50
    )
    universe, records = make_inputs()
    cfg.run_dir.mkdir(parents=True)
    universe.to_parquet(cfg.run_dir / "universe.parquet", index=False)
    docs = pipeline.stage_documents(cfg, universe, records)
    pipeline.stage_methods(cfg, universe, docs)

    metrics = pipeline.stage_evaluate(cfg, universe, docs, pipeline.load_methods(cfg))
    rm = metrics["returns"]
    assert rm["n"] == 9 and rm["best"] in metrics["labels"]
    entry = rm["ci"]["resid@2"]["fake+center"]
    assert entry["lo"] <= entry["mean"] <= entry["hi"]
    # 텍스트 이웃(같은 주제)은 무작위보다 훨씬 같이 움직여야 한다
    assert rm["methods"]["tfidf"]["resid@1"] > rm["baselines"]["random"]["resid"] + 0.2
    beyond = metrics["beyond"]
    assert set(beyond["regression"]) == set(metrics["labels"])
    assert all(np.isfinite(r["beta"]) for r in beyond["regression"].values())
    assert beyond["examples"]

    report = write_report(cfg, metrics, docs, universe).read_text(encoding="utf-8")
    for heading in ("## 2. 주가 동조성", "## 3. GICS가 묶지 않는 연결", "### 3.3 예시"):
        assert heading in report


def test_similarity_for_ensemble(cfg, monkeypatch):
    fake = FakeEmbedder()
    real_build = pipeline.build_method
    monkeypatch.setattr(
        pipeline, "build_method", lambda m: fake if m.kind == "sbert" else real_build(m)
    )
    universe, records = make_inputs()
    cfg.run_dir.mkdir(parents=True)
    docs = pipeline.stage_documents(cfg, universe, records)
    pipeline.stage_methods(cfg, universe, docs)
    companies, sim = pipeline.similarity_for(cfg, "ens")
    assert len(companies) == 9 and sim.shape == (9, 9)
    assert np.allclose(sim, sim.T) and (np.diag(sim) == 100).all()
    with pytest.raises(KeyError):
        pipeline.similarity_for(cfg, "tfidf+center")  # TF-IDF에는 +center 변형이 없다

"""표본 뽑기와 구간 라벨 (reviews.sqlite)."""

import sqlite3
from urllib.parse import quote

import pandas as pd
import pytest

from tenksim.relations import reviews
from tenksim.relations.candidates import build_candidates
from tenksim.relations.export import export_graph
from tenksim.relations.mentions import find_mentions
from tenksim.relations.names import AliasFile, build_dictionary

SECTORS = ["Energy", "Energy", "Energy", "Financials", "Financials", "Utilities"]
UNIVERSE = pd.DataFrame(
    {
        "cik": [1, 2, 3, 4, 5, 6],
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "name": [
            "Alpha Oil",
            "Beta Gas",
            "Gamma Fuel",
            "Delta Bank",
            "Echo Capital",
            "Foxtrot Power",
        ],
        "gics_sector": SECTORS,
        "gics_sub_industry": SECTORS,
    }
)
NAMES = UNIVERSE.set_index("cik")["name"].to_dict()


def text_for(cik: int) -> str:
    # 회사마다 다른 모든 회사를 한 문장씩 언급한다 → 회사당 단위 10개(자기 10-K 5 + 남의 10-K 5)
    others = [NAMES[c] for c in NAMES if c != cik]
    return " ".join(f"We compete with {n} in region {i}." for i, n in enumerate(others))


DOCS = pd.DataFrame(
    [
        {"cik": c, "section": "business", "status": "ok", "accession_number": f"acc-{c}",
         "form": "10-K", "filing_date": "2024-02-01", "period_of_report": "2023-12-31",
         "filing_url": f"https://example/{c}", "text": text_for(c)}
        for c in NAMES
    ]
)  # fmt: skip


@pytest.fixture()
def dbs(tmp_path):
    tables = find_mentions(DOCS, build_dictionary(UNIVERSE, AliasFile()))
    cands = build_candidates(tables, None, None, top_k=1, max_per_side=6)
    path = export_graph(
        tmp_path / "graph.db", tables=tables, candidates=cands, universe=UNIVERSE,
        documents=DOCS, meta={},
    )  # fmt: skip
    graph = sqlite3.connect(path)
    graph.row_factory = sqlite3.Row
    return reviews.connect(tmp_path / "reviews.sqlite"), graph, path


def units_of(rev, sample_id):
    return pd.read_sql(
        "SELECT * FROM sample_units WHERE sample_id = ? ORDER BY ord", rev, params=(sample_id,)
    )


def test_sample_picks_companies_round_robin_by_sector_and_caps_units(dbs):
    rev, graph, _ = dbs
    res = reviews.create_sample(
        rev, graph, sample_id="dev1", purpose="dev", n_companies=3, per_company_cap=4, seed=1
    )
    sectors = {SECTORS[int(c["node_id"][4:]) - 1] for c in res.companies}
    assert len(res.companies) == 3 and len(sectors) == 3  # 세 섹터에서 하나씩
    assert all(c["n_units_total"] == 10 and c["n_units"] == 4 for c in res.companies)
    u = units_of(rev, "dev1")
    assert u["unit_id"].is_unique and list(u["ord"]) == list(range(len(u)))
    assert len(u) == res.n_units <= 12


def test_samples_do_not_share_companies_units_or_filings(dbs):
    rev, graph, _ = dbs
    first = reviews.create_sample(
        rev, graph, sample_id="dev1", purpose="dev", n_companies=2, seed=0
    )
    second = reviews.create_sample(
        rev, graph, sample_id="conf1", purpose="confirm", n_companies=2, seed=0
    )
    a = {c["node_id"] for c in first.companies}
    b = {c["node_id"] for c in second.companies}
    assert not a & b
    u1, u2 = units_of(rev, "dev1"), units_of(rev, "conf1")
    assert not set(u1["unit_id"]) & set(u2["unit_id"])
    assert not set(u2["doc_node"]) & a  # 개발 표본 회사의 10-K 문장은 확인 표본에 없다
    with pytest.raises(ValueError, match="이미 있는"):
        reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)


def test_labels_are_validated_and_latest_wins(dbs):
    rev, graph, _ = dbs
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)
    unit = units_of(rev, "dev1")["unit_id"].iloc[0]
    kw = dict(unit=unit, sample_id="dev1")
    with pytest.raises(reviews.LabelError, match="같은 회사가 아니면"):
        reviews.record_label(rev, graph, is_entity="no", relations=["competitor"], **kw)
    with pytest.raises(reviews.LabelError, match="모르는 관계"):
        reviews.record_label(rev, graph, is_entity="yes", relations=["partner"], **kw)  # v1 코드

    reviews.record_label(rev, graph, is_entity="yes", relations=[], **kw)  # 관계 아님
    reviews.record_label(rev, graph, is_entity="yes", relations=["equity", "competitor"], **kw)
    label = reviews.latest_label(rev, unit)
    assert label["relations"] == ["competitor", "equity"]  # 정해진 순서로 저장
    assert label["status"] is None and label["partner_type"] is None and label["blind"] == 1
    assert label["schema_version"] == reviews.REVIEWS_SCHEMA_VERSION == 2
    assert "We compete with" in label["span_text"]
    assert rev.execute("SELECT COUNT(*) FROM span_labels").fetchone()[0] == 2  # 덮어쓰지 않음
    (progress,) = reviews.sample_progress(rev)
    assert progress["n_labeled"] == 1


def test_labels_follow_a_span_whose_id_changed(dbs):
    """근거 구간 규칙이 바뀌어 구간이 넓어지면(span_id가 바뀜) 표본 단위와 라벨이 새 구간을 따라간다."""
    rev, graph, _ = dbs
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)
    unit = units_of(rev, "dev1")["unit_id"].iloc[0]
    reviews.record_label(rev, graph, unit=unit, sample_id="dev1", is_entity="yes",
                         relations=["competitor"])  # fmt: skip
    old_spans = pd.read_sql(
        "SELECT span_id, accession, section, char_start, char_end FROM spans", graph
    )
    span_id, target = unit.rsplit("|", 1)
    prefix, rng = span_id.rsplit(":", 1)
    start, end = (int(x) for x in rng.split("-"))
    wider = f"{prefix}:{start}-{end + 3}"
    with graph:
        graph.execute(
            "UPDATE spans SET span_id = ?, char_end = ? WHERE span_id = ?",
            (wider, end + 3, span_id),
        )
        graph.execute("UPDATE mentions SET span_id = ? WHERE span_id = ?", (wider, span_id))
    mapping = reviews.moved_units(rev, old_spans, graph)
    assert mapping == {unit: f"{wider}|{target}"}
    assert reviews.remap_units(rev, mapping) == 1
    assert unit not in set(units_of(rev, "dev1")["unit_id"])
    label = reviews.latest_label(rev, f"{wider}|{target}")
    assert label["relations"] == ["competitor"] and label["span_id"] == wider
    assert reviews.latest_label(rev, unit) is None


def test_review_api_is_blind_and_saves(dbs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from tenksim.app.server import create_app

    rev, graph, path = dbs
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)
    client = TestClient(create_app(path))
    unit = client.get("/api/review/samples/dev1/units/0").json()
    for leaked in ("cues", "source", "rank_ab", "similarity_pct", "input_order"):
        assert leaked not in unit
    h = unit["highlights"][0]
    assert unit["text"][h["start"] : h["end"]] == unit["target"]["name"]

    body = {"unit_id": unit["unit_id"], "sample_id": "dev1", "is_entity": "yes",
            "relations": ["competitor", "business"]}  # fmt: skip
    assert client.post("/api/review/labels", json=body).status_code == 200
    assert client.get(f"/api/review/samples/{quote('dev1')}").json()["units"][0]["labeled"] == 1
    bad = {**body, "relations": ["doc_supplies_target"]}  # v1 코드는 더 받지 않는다
    assert client.post("/api/review/labels", json=bad).status_code == 422
    other = {**body, "sample_id": "nope"}
    assert client.post("/api/review/labels", json=other).status_code == 404


def test_confirm_sample_freezes_judge_settings(dbs):
    rev, graph, _ = dbs
    settings = {"model": "jev-1", "question_version": "v3.1", "context": "nearby",
                "thresholds": {"competitor": [0.7, 0.5]}}  # fmt: skip
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)
    reviews.create_sample(
        rev, graph, sample_id="conf1", purpose="confirm", n_companies=1, settings=settings
    )
    assert reviews.frozen_settings(rev, "dev1") is None
    frozen = reviews.frozen_settings(rev, "conf1")
    assert frozen == settings and reviews.settings_changes(frozen, settings) == []
    changed = {**settings, "context": "span",
               "thresholds": {"competitor": (0.6, 0.5), "business": [0.8, 0.5]}}  # fmt: skip
    assert reviews.settings_changes(frozen, changed) == [
        "context", "thresholds.business", "thresholds.competitor",
    ]  # fmt: skip


def test_second_sample_skips_every_10k_already_used(tmp_path):
    # 회사마다 다음 두 회사를 한 문장에 언급한다: 1→2·3, 2→3·4, …, 6→1·2
    ring = UNIVERSE.assign(gics_sector=[f"S{c}" for c in UNIVERSE["cik"]])
    docs = DOCS.assign(
        text=[
            f"We compete with {NAMES[c % 6 + 1]} and {NAMES[(c + 1) % 6 + 1]}." for c in DOCS["cik"]
        ]  # fmt: skip
    )
    tables = find_mentions(docs, build_dictionary(ring, AliasFile()))
    cands = build_candidates(tables, None, None, top_k=1, max_per_side=6)
    path = export_graph(
        tmp_path / "graph.db", tables=tables, candidates=cands, universe=ring, documents=docs,
        meta={},
    )  # fmt: skip
    graph = sqlite3.connect(path)
    graph.row_factory = sqlite3.Row
    rev = reviews.connect(tmp_path / "reviews.sqlite")
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1, seed=3)
    second = reviews.create_sample(
        rev, graph, sample_id="conf1", purpose="confirm", n_companies=3, seed=3
    )
    filings = {s: set(units_of(rev, s)["span_id"].str.split(":").str[1]) for s in ("dev1", "conf1")}
    assert second.n_units > 0 and len(filings["dev1"]) == 3  # 자기 10-K + 자기를 언급한 10-K 둘
    assert not filings["dev1"] & filings["conf1"]

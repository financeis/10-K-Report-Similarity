"""관계 화면 API: 회사의 관계, 관계 하나의 근거, 관계 검수 대기열과 검수 저장."""

from urllib.parse import quote

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_export import export  # noqa: E402  (tests/는 rootdir 기준으로 import)

from tenksim.app.server import create_app  # noqa: E402
from tenksim.relations import reviews  # noqa: E402
from tenksim.relations.judge import Judgement  # noqa: E402
from tenksim.relations.merge import build_relations  # noqa: E402

UPS, AMZN, SAMSUNG = "cik:2", "cik:1", "ext:samsung"
BUSINESS = f"business|{AMZN}|{UPS}"
UNCERTAIN = f"business|{UPS}|{SAMSUNG}"


def scores(business: float) -> dict:
    return {"is_entity": {"score": 0.95}, "competitor": {"score": 0.05},
            "business": {"score": business}, "equity": {"score": 0.05},
            "status": {"choice": "current"}}  # fmt: skip


def relations(con):
    """UPS 10-K: Amazon은 공급·협력 채택, Samsung은 불확실 (Customer A는 관계 없음)."""
    units = pd.read_sql(
        "SELECT DISTINCT span_id, doc_node, target_node, accession FROM mentions "
        "WHERE excluded IS NULL", con,
    )  # fmt: skip
    units["unit_id"] = units["span_id"] + "|" + units["target_node"]
    a, b = units["doc_node"], units["target_node"]
    units["pair_key"] = a.where(a < b, b) + "|" + b.where(a < b, a)
    score = {AMZN: 0.95, SAMSUNG: 0.6}
    judgements = {
        u: Judgement(u, "fake", "v", scores(score.get(t, 0.05)))
        for u, t in zip(units["unit_id"], units["target_node"], strict=True)
    }
    for u, t in zip(units["unit_id"], units["target_node"], strict=True):
        if t == AMZN:  # 같은 쌍에 경쟁 관계도 있다고 두어, 경쟁 관계 화면의 매출 비중을 확인한다
            judgements[u].answers["competitor"] = {"score": 0.95}
    cands = pd.read_sql("SELECT pair_key, similarity_pct FROM candidates", con)
    tables = build_relations(units, judgements, lambda q: (0.8, 0.3), cands, {"business"})
    return tables, {}


@pytest.fixture()
def client(tmp_path):
    path = export(tmp_path / "graph.db", relations=relations)
    return TestClient(create_app(path, tmp_path / "reviews.sqlite"))


def enc(s: str) -> str:
    return quote(s, safe="")


def review(client, edge_id: str, verdict: str) -> dict:
    """화면처럼: 관계를 불러와 받은 근거 해시를 검수와 함께 보낸다."""
    h = client.get(f"/api/edges/{enc(edge_id)}").json()["evidence_hash"]
    return {"edge_id": edge_id, "verdict": verdict, "evidence_hash": h}


def test_node_relations_and_edge_detail(client):
    rels = client.get(f"/api/nodes/{enc(UPS)}/relations").json()
    by_id = {r["edge_id"]: r for r in rels}
    assert set(by_id) == {BUSINESS, UNCERTAIN, f"competitor|{AMZN}|{UPS}"}
    assert by_id[BUSINESS]["other_id"] == AMZN and by_id[BUSINESS]["state"] == "accepted"
    assert by_id[UNCERTAIN]["state"] == "uncertain"

    e = client.get(f"/api/edges/{enc(BUSINESS)}").json()
    assert (e["a"]["node_id"], e["b"]["node_id"], e["validated"]) == (AMZN, UPS, 1)
    (ev,) = e["evidence"]
    h = ev["highlights"][0]
    assert ev["text"][h["start"] : h["end"]] == "Amazon" and ev["doc_node"] == UPS
    assert e["figures"][0]["value"] == 11.8 and e["figures"][0]["doc_node"] == UPS
    assert e["thresholds"] == {} and len(e["evidence_hash"]) == 16
    comp = client.get(f"/api/edges/{enc(f'competitor|{AMZN}|{UPS}')}").json()
    assert comp["figures"] == []  # 매출 비중은 공급·협력 관계에서만
    assert client.get(f"/api/edges/{enc('business|x|y')}").status_code == 404
    home = client.get("/api/nodes").json()
    assert {n["node_id"]: n["n_relations"] for n in home}[UPS] == 2  # 채택만 센다 (불확실 빼고)


def test_edge_review_queue_and_saving(client):
    queue = client.get("/api/review/edges").json()
    assert [q["edge_id"] for q in queue] == [UNCERTAIN]
    assert queue[0]["b_name"] == "Samsung"

    saved = client.post("/api/review/edges", json=review(client, UNCERTAIN, "accept"))
    assert saved.status_code == 200 and saved.json()["edge"]["state"] == "confirmed"
    assert client.get("/api/review/edges").json() == []  # 검수한 것은 빠진다
    assert len(client.get("/api/review/edges?include_done=true").json()) == 1

    client.post("/api/review/edges", json={**review(client, BUSINESS, "reject"), "note": "x"})
    rels = {r["edge_id"]: r for r in client.get(f"/api/nodes/{enc(UPS)}/relations").json()}
    assert rels[BUSINESS]["state"] == "rejected" and rels[BUSINESS]["review"]["note"] == "x"
    assert rels[UNCERTAIN]["state"] == "confirmed"

    bad = client.post("/api/review/edges", json=review(client, UNCERTAIN, "maybe"))
    assert bad.status_code == 422
    no_hash = client.post("/api/review/edges", json={"edge_id": UNCERTAIN, "verdict": "accept"})
    assert no_hash.status_code == 422  # 무엇을 보고 검수했는지 모르면 저장하지 않는다
    stale = client.post(
        "/api/review/edges",
        json={"edge_id": UNCERTAIN, "verdict": "reject", "evidence_hash": "not-what-i-saw"},
    )
    assert stale.status_code == 409  # 검수자가 본 근거와 지금 근거가 다르면 저장하지 않는다
    missing = client.post(
        "/api/review/edges", json={"edge_id": "nope", "verdict": "accept", "evidence_hash": "x"}
    )
    assert missing.status_code == 404


def test_review_state_needs_recheck_when_evidence_changes(tmp_path):
    rev = reviews.connect(tmp_path / "reviews.sqlite")
    edge = {"edge_id": "business|a|b", "src": "a", "dst": "b", "relation": "business",
            "decision": "uncertain", "score": 0.6}  # fmt: skip
    evidence = [("s1", "We buy from B."), ("s2", "B supplies us.")]
    h = reviews.evidence_hash(evidence)
    reviews.record_edge_review(rev, edge=edge, evidence_hash=h, verdict="accept")
    latest = reviews.latest_edge_reviews(rev)["business|a|b"]
    assert reviews.review_state(latest, reviews.evidence_hash(evidence[::-1])) == "accepted"
    changed = reviews.evidence_hash([*evidence, ("s3", "New text.")])
    assert reviews.review_state(latest, changed) == "needs_recheck"
    assert reviews.review_state(None, changed) is None
    with pytest.raises(reviews.LabelError):
        reviews.record_edge_review(rev, edge=edge, evidence_hash=h, verdict="maybe")


def test_counts_and_candidate_status_follow_human_reviews(client):
    home = {n["node_id"]: n["n_relations"] for n in client.get("/api/nodes").json()}
    assert (home[AMZN], home[SAMSUNG]) == (2, 0)
    ok = client.post("/api/review/edges", json=review(client, UNCERTAIN, "accept"))
    assert ok.status_code == 200
    client.post("/api/review/edges", json=review(client, BUSINESS, "reject"))

    home = {n["node_id"]: n["n_relations"] for n in client.get("/api/nodes").json()}
    assert (home[AMZN], home[SAMSUNG]) == (1, 1)  # 거절은 빼고 확인은 센다

    pair = client.get(f"/api/candidates/{enc(f'{AMZN}|{UPS}')}").json()
    assert pair["status"] == "accepted"  # 경쟁 관계는 남아 있다
    assert {e["edge_id"]: e["state"] for e in pair["edges"]} == {
        BUSINESS: "rejected", f"competitor|{AMZN}|{UPS}": "accepted",
    }  # fmt: skip
    client.post("/api/review/edges", json=review(client, f"competitor|{AMZN}|{UPS}", "reject"))
    pair = client.get(f"/api/candidates/{enc(f'{AMZN}|{UPS}')}").json()
    assert pair["status"] == "rejected_by_review"
    status = {
        c["pair_key"]: c["status"] for c in client.get(f"/api/nodes/{enc(UPS)}/candidates").json()
    }
    assert status[f"{UPS}|{SAMSUNG}"] == "accepted"

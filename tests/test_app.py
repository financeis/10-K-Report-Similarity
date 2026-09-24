"""웹앱 API: 작은 graph.db를 만들어 FastAPI TestClient로 확인한다."""

from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_export import export  # noqa: E402  (tests/는 rootdir 기준으로 import)

from tenksim.app.server import create_app  # noqa: E402

UPS, AMZN = "cik:2", "cik:1"


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(export(tmp_path / "graph.db")))


def enc(s: str) -> str:
    return quote(s, safe="")


def test_search_and_node(client):
    hits = client.get("/api/nodes", params={"q": "ups"}).json()
    assert hits[0]["node_id"] == UPS
    node = client.get(f"/api/nodes/{enc(UPS)}").json()
    assert node["filings"][0]["accession"] == "acc-2"
    assert client.get(f"/api/nodes/{enc('cik:999')}").status_code == 404


def test_candidates_are_seen_from_the_viewer(client):
    rows = client.get(f"/api/nodes/{enc(AMZN)}/candidates").json()
    (row,) = rows
    assert row["other_id"] == UPS
    assert (row["mentions_out"], row["mentions_in"]) == (0, 1)  # UPS가 Amazon을 언급


def test_candidate_detail_and_context(client):
    key = "cik:1|cik:2"
    d = client.get(f"/api/candidates/{enc(key)}").json()
    (span,) = d["spans"]
    (h,) = span["highlights"]
    assert span["text"][h["start"] : h["end"]] == "Amazon"
    assert d["figures"][0]["value"] == 11.8

    ctx = client.get(f"/api/spans/{enc(span['span_id'])}/context").json()
    marks = {m["kind"]: m for m in ctx["marks"]}
    assert ctx["excerpt"][marks["name"]["start"] : marks["name"]["end"]] == "Amazon"
    assert ctx["excerpt"][marks["span"]["start"] : marks["span"]["end"]] == span["text"]


def test_missing_graph_db_is_a_clear_error(tmp_path):
    client = TestClient(create_app(tmp_path / "none.db"))
    r = client.get("/api/meta")
    assert r.status_code == 503 and "tenksim export" in r.json()["detail"]

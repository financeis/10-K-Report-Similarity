"""graph.db 내보내기: 작은 가짜 10-K로 언급 → 후보 → SQLite까지."""

import sqlite3

import pandas as pd
import pytest

from tenksim.relations.candidates import build_candidates
from tenksim.relations.export import SCHEMA_VERSION, export_graph
from tenksim.relations.mentions import find_mentions
from tenksim.relations.names import AliasFile, build_dictionary

UNIVERSE = pd.DataFrame(
    {
        "cik": [1, 2],
        "ticker": ["AMZN", "UPS"],
        "name": ["Amazon", "United Parcel Service"],
        "gics_sector": ["Consumer Discretionary", "Industrials"],
        "gics_sub_industry": ["Broadline Retail", "Air Freight & Logistics"],
    }
)
ALIASES = AliasFile.model_validate(
    {"external": [{"id": "ext:samsung", "name": "Samsung", "names": ["Samsung"]}]}
)
TEXT = (
    "One customer, Amazon, represented approximately 11.8% of our consolidated revenues "
    "in 2023. Sales to Customer A were 5% of revenues. We buy scanners from Samsung."
)
DOCS = pd.DataFrame(
    [
        {
            "cik": 2,
            "section": "business",
            "status": "ok",
            "accession_number": "acc-2",
            "form": "10-K",
            "filing_date": "2024-02-20",
            "period_of_report": "2023-12-31",
            "filing_url": "https://example/ups",
            "text": TEXT,
        }
    ]
)


def export(path):
    tables = find_mentions(DOCS, build_dictionary(UNIVERSE, ALIASES))
    cands = build_candidates(tables, None, None, top_k=1, max_per_side=6)
    return export_graph(
        path, tables=tables, candidates=cands, universe=UNIVERSE, documents=DOCS,
        meta={"config": "t"},
    )  # fmt: skip


def test_export_writes_consistent_graph(tmp_path):
    path = export(tmp_path / "graph.db")
    con = sqlite3.connect(path)
    nodes = pd.read_sql("SELECT * FROM nodes", con).set_index("node_id")
    assert nodes.at["cik:2", "analyzed"] == 1 and nodes.at["cik:1", "analyzed"] == 0
    assert nodes.at["cik:1", "gics_sector"] == "Consumer Discretionary"
    assert nodes.at["anon:2:acc-2:customer-a", "parent_node"] == "cik:2"
    assert nodes.at["ext:samsung", "kind"] == "external"

    pairs = set(pd.read_sql("SELECT pair_key FROM candidates", con)["pair_key"])
    assert pairs == {"cik:1|cik:2", "anon:2:acc-2:customer-a|cik:2", "cik:2|ext:samsung"}

    # 근거 위치는 documents의 정제 텍스트 기준으로 정확히 맞아야 한다
    for text, start, end, doc in con.execute(
        "SELECT s.text, s.char_start, s.char_end, d.text FROM spans s "
        "JOIN documents d USING (accession, section)"
    ):
        assert doc[start:end] == text
    assert dict(con.execute("SELECT key, value FROM meta"))["schema_version"] == str(SCHEMA_VERSION)
    assert con.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0  # 1단계에서 채운다


def test_export_replaces_existing_file(tmp_path):
    path = tmp_path / "graph.db"
    path.write_text("old")
    export(path)
    assert (
        sqlite3.connect(path).execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 4
    )  # 회사 2 + 외부 1 + 익명 1
    assert not path.with_suffix(".tmp").exists()


def test_export_rejects_dangling_references(tmp_path):
    tables = find_mentions(DOCS, build_dictionary(UNIVERSE, ALIASES))
    tables.nodes = tables.nodes[tables.nodes["node_id"] != "ext:samsung"]
    cands = build_candidates(tables, None, None, top_k=1, max_per_side=6)
    with pytest.raises(RuntimeError, match="참조 오류"):
        export_graph(
            tmp_path / "graph.db", tables=tables, candidates=cands, universe=UNIVERSE,
            documents=DOCS, meta={},
        )  # fmt: skip

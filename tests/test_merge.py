"""관계 만들기: 단위별 판정 → 두 회사 관계와 근거 문장, 후보 상태."""

import sqlite3

import pandas as pd
from test_export import export  # noqa: E402  (tests/는 rootdir 기준으로 import)

from tenksim.relations.judge import Judgement
from tenksim.relations.merge import build_relations, edge_status

A, B, C, D = "cik:1", "cik:2", "cik:3", "cik:4"


def unit(n, doc, target):
    a, b = sorted((doc, target))
    return {"unit_id": f"s{n}|{target}", "span_id": f"s{n}", "doc_node": doc,
            "target_node": target, "pair_key": f"{a}|{b}", "accession": f"acc-{doc}"}  # fmt: skip


def judged(u, status="current", **scores):
    answers = {q: {"score": scores.get(q, 0.05)} for q in ("is_entity", "competitor", "business",
                                                           "equity")}  # fmt: skip
    answers["is_entity"]["score"] = scores.get("is_entity", 0.95)
    answers["status"] = {"choice": status}
    return Judgement(u["unit_id"], "fake", "v", answers)


def threshold(q):
    return {"competitor": (0.7, 0.5)}.get(q, (0.8, 0.3))


UNITS = [
    unit(1, A, B),  # 경쟁 채택
    unit(2, B, A),  # 같은 쌍, 다른 쪽 10-K: 경쟁 채택 + 공급·협력 불확실
    unit(3, A, C),  # 다른 회사 이름: 관계 점수가 높아도 식별에서 기각
    unit(4, C, D),  # 지분 채택 (과거), 검증 전 유형
    unit(5, A, D),  # 아직 판정 안 함
]
JUDGED = {
    "s1|cik:2": judged(UNITS[0], competitor=0.75),
    "s2|cik:1": judged(UNITS[1], competitor=0.9, business=0.6, status="historical"),
    "s3|cik:3": judged(UNITS[2], is_entity=0.1, business=0.95),
    "s4|cik:4": judged(UNITS[3], equity=0.9, status="historical"),
}
CANDIDATES = pd.DataFrame(
    {
        "pair_key": [f"{A}|{B}", f"{A}|{C}", f"{C}|{D}", f"{A}|{D}", f"{B}|{D}"],
        "similarity_pct": [99.0, None, None, 80.0, 95.0],
    }  # fmt: skip
)


def test_edges_keep_only_accepted_evidence_and_gate_on_entity():
    r = build_relations(pd.DataFrame(UNITS), JUDGED, threshold, CANDIDATES, {"competitor"})
    e = r.edges.set_index("edge_id")
    comp = e.loc[f"competitor|{A}|{B}"]
    assert (comp["decision"], comp["n_evidence"], comp["validated"]) == ("accepted", 2, 1)
    assert comp["score"] == 0.9 and comp["similarity_pct"] == 99.0 and comp["status"] == "current"
    assert e.loc[f"business|{A}|{B}", "decision"] == "uncertain"  # 0.6: 채택도 기각도 아님
    assert f"business|{A}|{C}" not in e.index  # 식별에서 기각
    eq = e.loc[f"equity|{C}|{D}"]
    assert (eq["decision"], eq["status"], eq["validated"]) == ("accepted", "historical", 0)

    ev = r.edge_evidence[r.edge_evidence["edge_id"] == f"competitor|{A}|{B}"]
    assert list(ev["span_id"]) == ["s2", "s1"]  # 점수 높은 순

    status = r.candidate_status.to_dict()
    assert status == {f"{A}|{B}": "accepted", f"{A}|{C}": "rejected", f"{C}|{D}": "accepted",
                      f"{A}|{D}": "pending", f"{B}|{D}": "similar"}  # fmt: skip
    assert len(r.unit_judgements) == 4


def test_edge_status_prefers_current():
    assert edge_status(["historical", "current"]) == "current"
    assert edge_status(["historical", "historical"]) == "historical"
    assert edge_status(["planned", "unclear"]) == "planned"
    assert edge_status([None, "unclear"]) == "unclear"


def test_export_writes_relations_and_checks_references(tmp_path):
    def relations(con):
        units = pd.read_sql(
            "SELECT DISTINCT span_id, doc_node, target_node, accession FROM mentions "
            "WHERE excluded IS NULL AND target_node = 'cik:1'", con,
        )  # fmt: skip
        units["unit_id"] = units["span_id"] + "|" + units["target_node"]
        units["pair_key"] = "cik:1|cik:2"
        judgements = {u: judged({"unit_id": u}, business=0.95) for u in units["unit_id"]}
        cands = pd.read_sql("SELECT pair_key, similarity_pct FROM candidates", con)
        return build_relations(units, judgements, threshold, cands, set()), {"edges": "test"}

    path = export(tmp_path / "graph.db", relations=relations)
    con = sqlite3.connect(path)
    (edge,) = con.execute("SELECT edge_id, relation, decision, validated FROM edges").fetchall()
    assert edge == ("business|cik:1|cik:2", "business", "accepted", 0)
    assert con.execute(
        "SELECT status FROM candidates WHERE pair_key = 'cik:1|cik:2'"
    ).fetchone() == ("accepted",)
    assert con.execute("SELECT COUNT(*) FROM unit_judgements").fetchone()[0] == 1
    assert dict(con.execute("SELECT key, value FROM meta"))["edges"] == "test"


def test_repeated_or_overlapping_sentences_become_one_evidence():
    units = [
        {"unit_id": f"{sid}|{B}", "span_id": sid, "doc_node": A, "target_node": B,
         "pair_key": f"{A}|{B}", "accession": "acc"}
        for sid in ("s:acc:business:0-50", "s:acc:business:0-80", "s:acc:risk_factors:10-60",
                    "s:acc:risk_factors:500-560")
    ]  # fmt: skip
    text = {
        "s:acc:business:0-50": "Our largest customer, B, was 21% of sales.",
        "s:acc:business:0-80": "Our largest customer, B, was 21% of sales. It buys widgets.",  # 겹침
        "s:acc:risk_factors:10-60": "Our largest customer, B, was 21%  of sales.",  # 1A에 반복
        "s:acc:risk_factors:500-560": "We also license patents to B.",
    }
    judgements = {u["unit_id"]: judged(u, business=0.9 + i / 100) for i, u in enumerate(units)}
    r = build_relations(
        pd.DataFrame(units), judgements, threshold, CANDIDATES, set(), span_text=text
    )
    ev = r.edge_evidence[r.edge_evidence["edge_id"] == f"business|{A}|{B}"]
    assert list(ev["span_id"]) == ["s:acc:risk_factors:500-560", "s:acc:risk_factors:10-60",
                                   "s:acc:business:0-80"]  # fmt: skip
    assert r.edges.set_index("edge_id").at[f"business|{A}|{B}", "n_evidence"] == 3

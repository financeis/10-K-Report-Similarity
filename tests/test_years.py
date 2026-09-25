"""연도별 관계도 비교 (relations/years.py): 회사 맞추기, 유지·새로 보임·보이지 않음과 그 이유."""

import pandas as pd

from tenksim.relations import years


def year(y, edges, mentioned, analyzed, evidence=None) -> years.YearGraph:
    rows = [
        {"key": years.pair_key(a, b), "relation": r, "state": s, "edge_id": f"{y}:{r}|{a}|{b}",
         "src": a, "dst": b, "status": "current"}
        for a, b, r, s in edges
    ]  # fmt: skip
    table = pd.DataFrame(
        rows, columns=["key", "relation", "state", "edge_id", "src", "dst", "status"]
    )
    names = {k: k for k in analyzed}
    return years.YearGraph(
        y, f"c{y}", table, {years.pair_key(a, b) for a, b in mentioned}, analyzed, names,
        evidence or {},
    )  # fmt: skip


def test_node_keys_match_companies_by_ticker():
    nodes = pd.DataFrame(
        {"node_id": ["cik:1364742", "cik:2012383", "ext:samsung"], "kind": ["company", "company", "external"],
         "ticker": ["BLK", "BLK", None]}
    )  # fmt: skip
    keys = years.node_keys(nodes)
    assert keys == {"cik:1364742": "BLK", "cik:2012383": "BLK", "ext:samsung": "ext:samsung"}


def test_compare_statuses_and_reasons():
    analyzed = {"A": True, "B": True, "C": True, "D": True, "E": True}
    base = year(
        2024,
        [("A", "B", "competitor", "accepted"), ("A", "C", "business", "accepted"),
         ("B", "C", "competitor", "uncertain"), ("A", "D", "competitor", "confirmed"),
         ("B", "E", "business", "rejected")],
        mentioned=[("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("B", "E"), ("A", "E")],
        analyzed=analyzed,
        evidence={"2024:competitor|A|B": ["We compete with B."]},
    )  # fmt: skip
    cur = year(
        2025,
        [("A", "B", "competitor", "accepted"), ("B", "C", "competitor", "accepted"),
         ("A", "E", "business", "accepted"), ("B", "E", "business", "accepted")],
        mentioned=[("A", "B"), ("B", "C"), ("A", "E"), ("B", "E"), ("A", "D")],
        analyzed={**analyzed, "D": False},
        evidence={"2025:competitor|A|B": ["We compete with B.", "B is a rival."]},
    )  # fmt: skip
    t = years.compare(base, cur).set_index(["key", "relation"])
    assert t.loc[("A|B", "competitor"), "status"] == "kept"
    assert bool(t.loc[("A|B", "competitor"), "same_text"])  # 같은 문장이 되풀이됨
    assert t.loc[("A|C", "business"), ["status", "reason"]].tolist() == ["gone", "not_mentioned"]
    assert t.loc[("B|C", "competitor"), ["status", "reason"]].tolist() == ["new", "uncertain"]
    assert t.loc[("A|D", "competitor"), ["status", "reason"]].tolist() == ["gone", "not_analyzed"]
    assert t.loc[("A|E", "business"), ["status", "reason"]].tolist() == ["new", "not_related"]
    # 앞 해에 검수로 거절한 관계가 이듬해 채택되면 '새로 보임'이고, 이유는 검수에서 거절
    assert t.loc[("B|E", "business"), ["status", "reason"]].tolist() == [
        "new",
        "rejected_by_review",
    ]
    assert len(t) == 6


def test_same_company_under_two_ciks_counts_once():
    """CIK가 둘인 회사가 같은 키로 두 줄을 가지면 보이는 줄을 쓴다."""
    base = year(2024, [("A", "B", "competitor", "uncertain"), ("A", "B", "competitor", "accepted")],
                mentioned=[("A", "B")], analyzed={"A": True, "B": True})  # fmt: skip
    cur = year(2025, [("A", "B", "competitor", "accepted")], mentioned=[("A", "B")],
               analyzed={"A": True, "B": True})  # fmt: skip
    t = years.compare(base, cur)
    assert t["status"].tolist() == ["kept"]


def test_shrunken_section_counts_as_not_analyzed():
    """근거가 있던 섹션이 이듬해 요약만 추출되어(4분의 1 아래로 줄어) 이름이 빠지면 비교 불가로 센다."""
    analyzed = {"A": True, "B": True}
    base = year(2024, [("A", "B", "business", "accepted")], mentioned=[("A", "B")],
                analyzed=analyzed)  # fmt: skip
    base.evidence_sections = {"2024:business|A|B": {("A", "risk_factors")}}
    base.section_chars = {("A", "risk_factors"): 100_000}
    cur = year(2025, [], mentioned=[], analyzed=analyzed)
    cur.section_chars = {("A", "risk_factors"): 8_000}
    t = years.compare(base, cur)
    assert t[["status", "reason"]].values.tolist() == [["gone", "not_analyzed"]]
    cur.section_chars = {("A", "risk_factors"): 90_000}  # 제대로 추출됐는데 이름이 없으면 공시 변화
    assert years.compare(base, cur)["reason"].tolist() == ["not_mentioned"]

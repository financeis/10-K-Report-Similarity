"""관계 만들기 (docs/relation-map-plan.md 7.4): 단위별 판정 → 두 회사 관계(선)와 근거 문장.

- 단위(근거 구간 × 언급된 회사)마다 회사 식별과 관계 질문을 합친다 (judge.combine).
- 쌍 × 관계마다 채택된 단위가 하나라도 있으면 관계를 채택하고, 채택된 단위만 근거로 잇는다.
  채택은 없고 불확실만 있으면 '불확실' 관계로 남겨 검수 대기열에 올린다.
- 관계에는 방향이 없다(5장). 시점은 채택된 근거의 Jev 답으로 정하되 조정·검수하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from .judge import Judgement, combine, decide
from .judge.questions import ENTITY_Q, RELATION_QUESTIONS, STATUS_Q

EDGE_DECISIONS = ("accepted", "uncertain")


@dataclass
class RelationTables:
    unit_judgements: pd.DataFrame
    """단위마다 질문별 점수·결정 (판정하지 못한 단위는 빠진다)."""
    edges: pd.DataFrame
    edge_evidence: pd.DataFrame
    candidate_status: pd.Series
    """pair_key → accepted / uncertain / rejected / pending / similar."""


def unit_frame(
    units: pd.DataFrame,
    judgements: dict[str, Judgement],
    threshold: Callable[[str], tuple[float, float]],
) -> pd.DataFrame:
    """units: unit_id, span_id, doc_node, target_node, pair_key, accession."""
    rows = []
    for u in units.to_dict("records"):
        j = judgements.get(u["unit_id"])
        if j is None:
            continue
        a = j.answers
        row = {**u, "model_id": j.model_id, "question_version": j.question_version}
        d_entity = decide(a.get(ENTITY_Q), *threshold(ENTITY_Q))
        row[f"s_{ENTITY_Q}"] = (a.get(ENTITY_Q) or {}).get("score")
        row[f"d_{ENTITY_Q}"] = d_entity
        for r in RELATION_QUESTIONS:
            row[f"s_{r}"] = (a.get(r) or {}).get("score")
            row[f"d_{r}"] = combine(d_entity, decide(a.get(r), *threshold(r)))
        row[STATUS_Q] = (a.get(STATUS_Q) or {}).get("choice")
        rows.append(row)
    return pd.DataFrame(rows)


def edge_status(statuses: list[str | None]) -> str:
    """근거 문장들의 시점 → 관계의 시점. 하나라도 현재면 현재(지금도 이어지는 관계),
    전부 과거면 과거, 계획만 있으면 계획, 나머지는 알 수 없음."""
    s = {x for x in statuses if x}
    if "current" in s:
        return "current"
    if s == {"historical"}:
        return "historical"
    if "planned" in s:
        return "planned"
    return "unclear"


def build_relations(
    units: pd.DataFrame,
    judgements: dict[str, Judgement],
    threshold: Callable[[str], tuple[float, float]],
    candidates: pd.DataFrame,
    validated: set[str],
    span_text: dict[str, str] | None = None,
) -> RelationTables:
    """candidates: pair_key, similarity_pct (graph.db의 candidates).
    span_text: span_id → 원문. 주면 같은 문장의 반복(Item 1과 1A)과 겹치는 구간을 근거에서 하나로 합친다."""
    uj = unit_frame(units, judgements, threshold)
    edges, evidence = [], []
    sim = candidates.set_index("pair_key")["similarity_pct"].to_dict()
    if not uj.empty:
        for r in RELATION_QUESTIONS:
            for pair_key, g in uj.groupby("pair_key"):
                yes = g[g[f"d_{r}"] == "yes"]
                chosen, decision = (yes, "accepted") if len(yes) else (
                    g[g[f"d_{r}"] == "abstain"], "uncertain"
                )  # fmt: skip
                if chosen.empty:
                    continue
                chosen = _dedupe(chosen.sort_values(f"s_{r}", ascending=False), span_text)
                src, dst = pair_key.split("|")
                edge_id = f"{r}|{pair_key}"
                top = chosen.iloc[0]
                edges.append(
                    {"edge_id": edge_id, "src": src, "dst": dst, "relation": r, "subtype": None,
                     "directed": 0, "status": edge_status(list(chosen[STATUS_Q])),
                     "basis": "disclosed", "decision": decision,
                     "validated": int(r in validated), "score": float(top[f"s_{r}"]),
                     "n_evidence": len(chosen), "model_id": top["model_id"],
                     "question_version": top["question_version"],
                     "similarity_pct": sim.get(pair_key), "resid_corr": None,
                     "as_of": top["accession"], "review_state": None}
                )  # fmt: skip
                evidence += [
                    {"edge_id": edge_id, "span_id": e["span_id"], "target_node": e["target_node"],
                     "question": r, "score": e[f"s_{r}"], "decision": e[f"d_{r}"], "ord": i}
                    for i, e in enumerate(chosen.to_dict("records"))
                ]  # fmt: skip
    edges_df = pd.DataFrame(edges, columns=EDGE_COLUMNS)
    evidence_df = pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS)
    return RelationTables(
        uj, edges_df, evidence_df, candidate_status(candidates, units, uj, edges_df)
    )


def _span_range(span_id: str) -> tuple[str, str, int, int] | None:
    """s:<accession>:<section>:<start>-<end> → (accession, section, start, end)."""
    try:
        _, accession, section, span = span_id.split(":")
        start, end = span.split("-")
        return accession, section, int(start), int(end)
    except ValueError:
        return None


def _dedupe(chosen: pd.DataFrame, span_text: dict[str, str] | None) -> pd.DataFrame:
    """점수 높은 순으로 보며, 같은 10-K에서 이미 고른 근거와 문장이 같거나 구간이 겹치면 뺀다.
    다른 회사 10-K의 같은 문장은 서로 다른 진술이므로 남긴다."""
    if not span_text:
        return chosen
    keep, texts, ranges = [], set(), []
    for i, row in enumerate(chosen.itertuples()):
        text = " ".join((span_text.get(row.span_id) or "").split())
        rng = _span_range(row.span_id)
        if text and (row.doc_node, text) in texts:
            continue
        if rng and any(r[:2] == rng[:2] and r[2] < rng[3] and rng[2] < r[3] for r in ranges):
            continue
        keep.append(i)
        texts.add((row.doc_node, text))
        if rng:
            ranges.append(rng)
    return chosen.iloc[keep]


EDGE_COLUMNS = [
    "edge_id", "src", "dst", "relation", "subtype", "directed", "status", "basis", "decision",
    "validated", "score", "n_evidence", "model_id", "question_version", "similarity_pct",
    "resid_corr", "as_of", "review_state",
]  # fmt: skip
EVIDENCE_COLUMNS = ["edge_id", "span_id", "target_node", "question", "score", "decision", "ord"]


def candidate_status(
    candidates: pd.DataFrame, units: pd.DataFrame, uj: pd.DataFrame, edges: pd.DataFrame
) -> pd.Series:
    """후보 쌍의 판정 결과. 관계의 검수 상태(review_state)가 있으면 모델 판정보다 우선한다."""
    has_units = set(units["pair_key"])
    all_judged = {
        k for k, g in units.groupby("pair_key") if set(g["unit_id"]) <= set(uj.get("unit_id", []))
    }
    pair = edges["src"] + "|" + edges["dst"]
    review = edges["review_state"].fillna("")
    shown = ((edges["decision"] == "accepted") & (review != "rejected")) | (review == "accepted")
    waiting = (edges["decision"] == "uncertain") & ~review.isin(["accepted", "rejected"])
    accepted = set(pair[shown])
    uncertain = set(pair[waiting])
    reviewed_out = set(pair) - accepted - uncertain  # 관계는 있었지만 사람이 모두 거절

    def status(k: str) -> str:
        if k in accepted:
            return "accepted"
        if k in uncertain:
            return "uncertain"
        if k in reviewed_out:
            return "rejected_by_review"
        if k not in has_units:
            return "similar"  # 이름 언급이 없어 판정할 근거가 없는 유사도 후보
        if k in all_judged:
            return "rejected"  # 모든 근거를 판정했지만 관계가 없음
        return "pending"  # 아직 판정하지 않은 근거가 있음

    return pd.Series({k: status(k) for k in candidates["pair_key"]}, name="status")

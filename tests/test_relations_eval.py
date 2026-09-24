"""판정 채점: 라벨과 점수를 맞대어 질문별·단위별·쌍별 지표를 만든다."""

import sqlite3

import pandas as pd
import pytest
from test_reviews import DOCS, UNIVERSE  # noqa: E402  (tests/는 rootdir 기준으로 import)

from tenksim.relations import reviews
from tenksim.relations.candidates import build_candidates
from tenksim.relations.evaluate import (
    LABEL_GROUPS,
    disagreements,
    evaluate,
    format_report,
    load_labels,
    pair_table,
    unit_table,
    wilson,
)
from tenksim.relations.export import export_graph
from tenksim.relations.judge import Judgement
from tenksim.relations.judge.cache import connect, judge_cached
from tenksim.relations.judge.inputs import requests_for_units
from tenksim.relations.judge.questions import (
    RELATION_QUESTIONS,
    YES_NO_QUESTIONS,
    questions_for,
    state_for,
)
from tenksim.relations.mentions import find_mentions
from tenksim.relations.names import AliasFile, build_dictionary
from tenksim.relations.reviews import LEGACY_RELATIONS, RELATIONS


def labels_frame(rows):
    base = {"ord": 0, "span_id": "s", "labeled": True, "skipped": False, "status": "current",
            "partner_type": None, "note": None, "span_text": "t"}  # fmt: skip
    return pd.DataFrame([{**base, "ord": i, **r} for i, r in enumerate(rows)])


def judgement(unit, **scores):
    answers = {q: {"score": scores.get(q, 0.05)} for q in YES_NO_QUESTIONS}
    answers["status"] = {"choice": "current"}
    return Judgement(unit, "fake", "v", answers)


A, B = "cik:1", "cik:2"
ROWS = [
    # 맞게 채택된 경쟁
    {"unit_id": "u1", "doc_node": A, "target_node": B, "pair_key": f"{A}|{B}",
     "is_entity": "yes", "relations": ["competitor"]},
    # 관계 없음을 맞게 기각
    {"unit_id": "u2", "doc_node": A, "target_node": B, "pair_key": f"{A}|{B}",
     "is_entity": "yes", "relations": [], "status": None},
    # 다른 회사: 관계 점수가 높아도 식별에서 걸러야 한다
    {"unit_id": "u3", "doc_node": A, "target_node": B, "pair_key": f"{A}|{B}",
     "is_entity": "no", "relations": [], "status": None},
    # 공급(A→B) 라벨, 점수가 애매해 불확실
    {"unit_id": "u4", "doc_node": A, "target_node": B, "pair_key": f"{A}|{B}",
     "is_entity": "yes", "relations": ["doc_supplies_target"]},
    # B의 10-K가 말한 협력: 방향·세부 유형 없이 같은 '공급·협력'으로 묶인다
    {"unit_id": "u5", "doc_node": B, "target_node": A, "pair_key": f"{A}|{B}",
     "is_entity": "yes", "relations": ["partner"], "partner_type": "licensing"},
]  # fmt: skip
JUDGED = {
    "u1": judgement("u1", is_entity=0.95, competitor=0.9),
    "u2": judgement("u2", is_entity=0.95),
    "u3": judgement("u3", is_entity=0.1, competitor=0.9),
    "u4": judgement("u4", is_entity=0.95, business=0.5),
    "u5": judgement("u5", is_entity=0.95, business=0.95),
}


def test_label_groups_cover_every_review_code_once():
    codes = [c for group in LABEL_GROUPS.values() for c in group]
    assert sorted(codes) == sorted(set(RELATIONS) | set(LEGACY_RELATIONS))  # v2 + v1 코드
    assert len(codes) == len(set(codes)) and set(LABEL_GROUPS) == set(RELATION_QUESTIONS)
    assert set(RELATIONS) == set(RELATION_QUESTIONS)  # v2 라벨은 판정 관계와 같다


def test_metrics_combine_entity_gate_and_merge_pairs():
    t = unit_table(labels_frame(ROWS), JUDGED, accept=0.8, reject=0.3)
    m = evaluate(t)

    comp = m["questions"]["competitor"]  # 회사가 맞는 단위(u1, u2, u4, u5)에서만
    assert (comp["n"], comp["n_pos"], comp["accepted"], comp["accepted_precision"]) == (
        4, 1, 1, 1.0,
    )  # fmt: skip
    assert comp["auc"] == 1.0
    ent = m["questions"]["is_entity"]
    assert (ent["n"], ent["rejected"], ent["rejected_pos"]) == (5, 1, 0)

    comb = m["units_combined"]["competitor"]  # u3는 관계 점수가 높지만 식별에서 기각
    assert (comb["n"], comb["accepted"], comb["rejected"]) == (5, 1, 4)
    biz = m["units_combined"]["business"]
    assert (biz["n_pos"], biz["accepted"], biz["uncertain"], biz["uncertain_pos"]) == (2, 1, 1, 1)

    pairs = pair_table(t)
    business = pairs[pairs["relation"] == "business"]
    assert len(business) == 1 and business["label"].iloc[0]  # u4와 u5가 한 줄로 합쳐진다
    assert business["pred"].iloc[0] == "yes"  # u5가 채택했으므로
    assert m["pairs"]["business"]["accepted_recall"] == 1.0
    assert m["pairs"]["n_pairs"] == 1

    wrong = disagreements(t)
    assert list(wrong["unit_id"]) == ["u4"]  # 불확실은 채택이 아니므로 놓친 것으로 본다
    assert "business:abstain" in wrong["wrong"].iloc[0]


def test_wilson_interval_matches_plan_example():
    lo, hi = wilson(54, 60)  # 계획서 7.6: 60건 중 54건이면 약 80~95%
    assert 0.79 < lo < 0.81 and 0.94 < hi < 0.96
    assert wilson(0, 0) is None


class ScoreByText:
    """본문에 'compete'가 있으면 경쟁 점수를 높게 주는 가짜 판정 모델."""

    model_id = "fake"

    def payload(self, request):
        return {"state": state_for(request), "questions": questions_for(request)}

    def judge(self, requests):
        return [
            judgement(r.unit_id, is_entity=0.9,
                      competitor=0.9 if "compete" in r.passage else 0.1)
            for r in requests
        ]  # fmt: skip


def test_sample_to_report_end_to_end(tmp_path):
    tables = find_mentions(DOCS, build_dictionary(UNIVERSE, AliasFile()))
    cands = build_candidates(tables, None, None, top_k=1, max_per_side=6)
    path = export_graph(
        tmp_path / "graph.db", tables=tables, candidates=cands, universe=UNIVERSE,
        documents=DOCS, meta={},
    )  # fmt: skip
    graph = sqlite3.connect(path)
    graph.row_factory = sqlite3.Row
    rev = reviews.connect(tmp_path / "reviews.sqlite")
    reviews.create_sample(rev, graph, sample_id="dev1", purpose="dev", n_companies=1)
    labels = load_labels(rev, "dev1")
    assert not labels["labeled"].any()
    for unit in labels["unit_id"][:3]:
        reviews.record_label(rev, graph, unit=unit, sample_id="dev1", is_entity="yes",
                             relations=["competitor"])  # fmt: skip
    labels = load_labels(rev, "dev1")
    assert labels["labeled"].sum() == 3 and labels["relations"].iloc[0] == ["competitor"]

    run = judge_cached(
        ScoreByText(), requests_for_units(graph, labels["unit_id"]), connect(tmp_path / "j.sqlite")
    )
    judged = {j.unit_id: j for j in run.judgements if j}
    m = evaluate(unit_table(labels, judged, accept=0.8, reject=0.3))
    assert m["units"]["scored"] == 3
    assert m["questions"]["competitor"]["accepted_precision"] == 1.0
    report = format_report(
        m, {"sample": "dev1", "model": "fake", "question_version": "v", "labels": "test",
            "accept": 0.8, "reject": 0.3},
    )  # fmt: skip
    assert "## 질문별" in report and "경쟁" in report


def test_load_labels_rejects_unknown_sample(tmp_path):
    rev = reviews.connect(tmp_path / "reviews.sqlite")
    with pytest.raises(ValueError, match="표본"):
        load_labels(rev, "nope")


def test_per_question_thresholds_and_sweep_suggestion():
    t = unit_table(labels_frame(ROWS), JUDGED, 0.8, 0.3, per_question={"business": (0.4, 0.3)})
    assert t.set_index("unit_id").at["u4", "p_business"] == "yes"  # 0.5 ≥ 0.4
    sweep = evaluate(t)["sweep"]["business"]
    at = {x["at"]: x for x in sweep["combined"]}
    assert at[0.5]["pair_recall"] == 1.0 and at[0.5]["unit_recall"] == 1.0
    assert at[0.6]["unit_recall"] == 0.5  # u4(0.5)가 빠진다
    assert sweep["suggest_accept"] == 0.5  # 쌍·문장 정밀도를 지키면서 재현율이 가장 높은 값


def test_judge_config_thresholds_fall_back_and_validate():
    from typing import get_args

    from pydantic import ValidationError

    from tenksim.config import JudgeConfig, JudgeQuestion

    j = JudgeConfig(thresholds={"competitor": {"accept": 0.7}})
    assert j.threshold("competitor") == (0.7, 0.3) and j.threshold("business") == (0.8, 0.3)
    assert set(get_args(JudgeQuestion)) == set(YES_NO_QUESTIONS)
    with pytest.raises(ValidationError, match="business"):
        JudgeConfig(thresholds={"business": {"reject": 0.9}})


def test_confirm_report_hides_threshold_suggestions():
    m = evaluate(unit_table(labels_frame(ROWS), JUDGED, 0.8, 0.3))
    meta = {"sample": "c", "model": "fake", "question_version": "v", "labels": "t",
            "accept": 0.8, "reject": 0.3}  # fmt: skip
    assert "## 임계값별" in format_report(m, {**meta, "purpose": "dev"})
    assert "## 임계값별" not in format_report(m, {**meta, "purpose": "confirm"})

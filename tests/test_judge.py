"""판정 입력 만들기, 캐시, Jev 어댑터 (네트워크 없이)."""

import json
import sqlite3

import pytest
from test_export import export  # noqa: E402  (tests/는 rootdir 기준으로 import)

from tenksim.relations.judge import Judgement, decide
from tenksim.relations.judge.cache import connect, judge_cached
from tenksim.relations.judge.inputs import (
    display_name,
    mark_names,
    nearby_text,
    requests_for_units,
)
from tenksim.relations.judge.questions import QUESTION_NAMES, questions_for, state_for
from tenksim.relations.reviews import STATUS


def test_display_name_reads_like_a_sentence():
    assert display_name("Coca-Cola Company (The)") == "The Coca-Cola Company"
    assert display_name("Lilly (Eli)") == "Eli Lilly"
    assert display_name("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert (
        display_name("Hon Hai Precision (Foxconn)", kind="external")
        == "Hon Hai Precision (Foxconn)"
    )
    assert mark_names("We sell to Amazon and Amazon Web", [(11, 17), (22, 28)]) == (
        "We sell to [[Amazon]] and [[Amazon]] Web"
    )


@pytest.fixture()
def graph(tmp_path):
    con = sqlite3.connect(export(tmp_path / "graph.db"))
    con.row_factory = sqlite3.Row
    return con


def units(graph) -> dict[str, str]:
    return {
        r["target_node"]: f"{r['span_id']}|{r['target_node']}"
        for r in graph.execute("SELECT DISTINCT span_id, target_node FROM mentions")
    }


def test_requests_mark_the_named_company_and_describe_both_sides(graph):
    u = units(graph)
    amzn, anon, samsung = requests_for_units(
        graph, [u["cik:1"], u["anon:2:acc-2:customer-a"], u["ext:samsung"]]
    )
    assert amzn.passage.startswith("One customer, [[Amazon]], represented")
    assert amzn.passage.count("[[") == 1
    assert amzn.filer.name == "United Parcel Service"
    assert amzn.filer.description == (
        "United Parcel Service (ticker UPS; industry: Air Freight & Logistics)"
    )
    assert (
        amzn.source
        == "United Parcel Service's 10-K annual report filed 2024-02-20, Item 1. Business"
    )
    assert anon.named.description == "Customer A (a party the filer does not name)"
    assert samsung.named.description == "Samsung"
    assert requests_for_units(graph, ["s:gone:business:0-1|cik:1"]) == []

    state, q = state_for(amzn), questions_for(amzn)
    assert state["passage"] == amzn.passage and "refer to United Parcel Service" in state["source"]
    assert tuple(q) == QUESTION_NAMES
    assert set(q["status"]["criteria"]) == set(STATUS)
    assert "United Parcel Service and Amazon do business" in q["business"]["instructions"]
    assert "context_before" not in state and "other_passages" not in state


def test_context_modes_add_nearby_text_or_other_passages(graph):
    u = units(graph)
    (near,) = requests_for_units(graph, [u["ext:samsung"]], context="nearby")
    assert near.context_before.startswith("One customer, Amazon") and near.context_after is None
    assert "[[Samsung]]" in near.passage and "[[" not in near.context_before
    assert "context_before" in state_for(near) and "use them" in state_for(near)["source"]

    (pair,) = requests_for_units(graph, [u["cik:1"]], context="pair")
    assert pair.related == ()  # 이 쌍의 근거 구간은 자기 하나뿐
    with pytest.raises(ValueError, match="context"):
        requests_for_units(graph, [u["cik:1"]], context="everything")


def test_nearby_text_takes_neighbouring_paragraphs_and_cuts_at_sentences():
    text = "Title\nIntro one. Intro two.\nA. B target C. D.\nNext one. Next two.\nFar away."
    start = text.index("target")
    before, after = nearby_text(text, start, start + 6, max_chars=1000)
    assert before == "Intro one. Intro two.\nA. B"  # 앞 문단 하나 + 같은 문단
    assert after == "C. D.\nNext one. Next two."
    before, after = nearby_text(text, start, start + 6, max_chars=12)
    assert before == "A. B" and after == "C. D."  # 잘린 앞 문장 조각("wo.")은 버린다


class FakeJudge:
    model_id = "fake-1"

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def payload(self, request):
        return {"state": state_for(request), "questions": questions_for(request)}

    def judge(self, requests):
        self.calls += [r.unit_id for r in requests]
        return [
            None if r.unit_id in self.fail
            else Judgement(r.unit_id, self.model_id, "v", {"is_entity": {"score": 0.9}},
                           "fake-1.0", 100)
            for r in requests
        ]  # fmt: skip


def test_cache_asks_each_input_once(graph, tmp_path):
    u = units(graph)
    reqs = requests_for_units(graph, [u["cik:1"], u["ext:samsung"], u["cik:1"]])
    cache = connect(tmp_path / "judgements.sqlite")

    judge = FakeJudge(fail={u["ext:samsung"]})
    run = judge_cached(judge, reqs, cache)
    assert judge.calls == [u["cik:1"], u["ext:samsung"]]  # 같은 입력은 한 번만
    assert (run.n_new, run.n_failed, run.n_cached, run.input_tokens) == (1, 1, 0, 100)
    assert run.judgements[1] is None and run.judgements[2].unit_id == u["cik:1"]

    judge = FakeJudge()
    run = judge_cached(judge, reqs, cache)
    assert judge.calls == [u["ext:samsung"]]  # 실패했던 것만 다시
    assert run.n_cached == 2 and run.judgements[0].cached
    assert run.judgements[0].answers == {"is_entity": {"score": 0.9}}


def test_decide_uses_thresholds_or_llm_decision():
    assert decide({"score": 0.85}, 0.8, 0.3) == "yes"
    assert decide({"score": 0.5}, 0.8, 0.3) == "abstain"
    assert decide({"score": 0.1}, 0.8, 0.3) == "no"
    assert decide({"score": None, "decision": "no"}, 0.8, 0.3) == "no"
    assert decide(None, 0.8, 0.3) == "abstain"


def test_jev_adapter_parses_answers_and_stops_on_auth_error(graph):
    pytest.importorskip("typesafe_sdk")
    import httpx2

    from tenksim.relations.judge.jev import JevJudge, JudgeError

    seen = []

    def ok(request):
        body = json.loads(request.content)
        seen.append(body)
        answers = {
            name: {"type": "noul", "noul": 0.9} if q["type"] == "noul"
            else {"type": "choice", "choice": "current", "confidence": 0.8,
                  "probabilities": {k: 0.7 if k == "current" else 0.1 for k in q["criteria"]}}
            for name, q in body["questions"].items()
        }  # fmt: skip
        return httpx2.Response(
            200,
            json={"model": "jev-1.13.0", "answers": answers,
                  "usage": {"input_tokens": 1234, "output_tokens": 10}},
        )  # fmt: skip

    reqs = requests_for_units(graph, [units(graph)["cik:1"]])
    judge = JevJudge("jev-1.13.0", api_key="test-key", transport=httpx2.MockTransport(ok))
    (j,) = judge.judge(reqs)
    assert seen[0]["model"] == "jev-1.13.0" and seen[0]["state"] == state_for(reqs[0])
    assert j.answers["competitor"] == {"score": 0.9}
    assert j.answers["status"]["choice"] == "current"
    assert (j.served_model, j.input_tokens) == ("jev-1.13.0", 1234)

    denied = JevJudge(
        "jev-1.13.0", api_key="bad-key",
        transport=httpx2.MockTransport(lambda r: httpx2.Response(401, json={"detail": "no"})),
    )  # fmt: skip
    with pytest.raises(JudgeError, match="인증"):
        denied.judge(reqs)

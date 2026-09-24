"""후보 쌍: 유사도 상위 K ∪ 이름 언급, 쌍별 근거 구간 선택."""

import numpy as np
import pandas as pd

from tenksim.relations.candidates import build_candidates, pair_key, similarity_ranks
from tenksim.relations.mentions import MentionTables

COMPANIES = pd.DataFrame({"cik": [1, 2, 3, 4]})
# 1-2가 가장 비슷하고, 3-4가 가장 비슷하다
SIM = np.array(
    [
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.8],
        [0.1, 0.2, 0.8, 1.0],
    ]
)


def mention(doc, target, span, excluded=None):
    return {"doc_node": doc, "target_node": target, "span_id": span, "excluded": excluded}


def span(span_id, text, start=0, lead=None):
    return {"span_id": span_id, "section": "business", "char_start": start, "text": text,
            "lead_text": lead}  # fmt: skip


def tables(mentions, spans):
    return MentionTables(
        pd.DataFrame(mentions), pd.DataFrame(spans), pd.DataFrame(), pd.DataFrame()
    )


def test_similarity_ranks():
    r = similarity_ranks(SIM)
    assert r[0, 1] == 1 and r[0, 2] == 2 and r[0, 3] == 3 and r[0, 0] == 0


def test_candidates_union_of_top_k_and_mentions():
    t = tables(
        [
            mention("cik:1", "cik:4", "s1"),  # 유사도 밖이지만 언급
            mention("cik:4", "cik:1", "s2"),
            mention("cik:1", "ext:tsmc", "s3"),  # 외부 기업
            mention("cik:2", "cik:3", "s4", excluded="exec_bio"),  # 제외된 언급은 후보가 아님
        ],
        [
            span("s1", "We compete with Four."),
            span("s2", "One is our customer."),
            span("s3", "TSMC makes our wafers."),
            span("s4", "He worked at Three."),
        ],  # fmt: skip
    )
    c = build_candidates(t, COMPANIES, SIM, top_k=1, max_per_side=6).candidates.set_index(
        "pair_key"
    )
    assert set(c.index) == {
        pair_key("cik:1", "cik:2"),
        pair_key("cik:3", "cik:4"),
        pair_key("cik:1", "cik:4"),
        pair_key("cik:1", "ext:tsmc"),
    }
    assert c.at["cik:1|cik:2", "source"] == "similarity"
    row = c.loc["cik:1|cik:4"]
    assert row["source"] == "mention"
    assert (row["mentions_ab"], row["mentions_ba"]) == (1, 1)
    assert (row["rank_ab"], row["rank_ba"]) == (3, 3)  # 서로에게 가장 먼 회사
    assert row["n_spans"] == 2
    ext = c.loc["cik:1|ext:tsmc"]
    assert pd.isna(ext["rank_ab"]) and pd.isna(ext["similarity_pct"])
    assert (c["status"] == "pending").all()


def test_span_selection_alternates_cue_types():
    competition = [span(f"c{i}", f"We compete with Two in market {i}.", start=i) for i in range(5)]
    supply = [span("p1", "Two is a supplier of our chips.", start=10)]
    t = tables(
        [mention("cik:1", "cik:2", s["span_id"]) for s in competition + supply],
        competition + supply,
    )
    spans = build_candidates(t, None, None, top_k=1, max_per_side=3).candidate_spans
    picked = spans.sort_values("order")["span_id"].tolist()
    # 경쟁 문장이 5개여도 공급 문장이 함께 들어가고, 문서 순서를 유지한다
    assert picked == ["c0", "c1", "p1"]

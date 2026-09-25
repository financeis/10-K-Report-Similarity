"""이름 사전과 언급 찾기. 네트워크 없이 작은 가짜 10-K로 확인한다."""

import pandas as pd
import pytest

from tenksim.relations.mentions import (
    exec_bio_ranges,
    find_mentions,
    parse_figures,
    sentence_bounds,
)
from tenksim.relations.names import AliasFile, build_dictionary, strip_name

UNIVERSE = pd.DataFrame(
    {
        "cik": [1, 2, 3, 4, 5, 6, 7],
        "ticker": ["AMZN", "UPS", "TGT", "STZ", "CEG", "NDAQ", "GEV"],
        "name": [
            "Amazon",
            "United Parcel Service",
            "Target Corporation",
            "Constellation Brands",
            "Constellation Energy",
            "Nasdaq, Inc.",
            "GE Vernova",
        ],
    }
)
ALIASES = AliasFile.model_validate(
    {
        "companies": {
            "AMZN": {"names": ["Amazon", "Amazon.com"], "products": [{"name": "AWS"}]},
            "TGT": {"names": ["Target Corporation"]},
            "NDAQ": {"names": ["Nasdaq"]},
            "GEV": {"since": 2024},
        },
        "external": [{"id": "cik:1046179", "name": "TSMC", "names": ["TSMC"]}],
        "generic_words": ["Target", "Constellation"],
        "exclude_contexts": [
            {"reason": "listing", "tickers": ["NDAQ"], "pattern": "listed on (?:the )?Nasdaq"}
        ],
    }
)


@pytest.fixture(scope="module")
def dictionary():
    return build_dictionary(UNIVERSE, ALIASES)


def doc(cik, text, section="business", period="2023-12-31"):
    return {
        "cik": cik,
        "section": section,
        "accession_number": f"acc-{cik}",
        "period_of_report": period,
        "text": text,
    }


def run(dictionary, *docs):
    return find_mentions(pd.DataFrame(docs), dictionary)


def test_strip_name():
    assert strip_name("Apple Inc.") == "Apple"
    assert strip_name("Coca-Cola Company (The)") == "Coca-Cola"
    assert strip_name("Nasdaq, Inc.") == "Nasdaq"


def test_dictionary_skips_generic_and_ambiguous_names(dictionary):
    assert dictionary.lookup("Target") is None  # 일반 단어
    assert dictionary.lookup("Target Corporation").node_id == "cik:3"
    assert dictionary.lookup("Constellation") is None  # 두 회사가 공유
    assert dictionary.lookup("Constellation Energy").node_id == "cik:5"
    assert dictionary.lookup("AWS").kind == "product"
    assert dictionary.lookup("TSMC").node_id == "cik:1046179"


def test_company_since_hides_names_before_spin_off(dictionary):
    assert dictionary.lookup("GE Vernova", 2023) is None
    assert dictionary.lookup("GE Vernova", 2024).node_id == "cik:7"


def test_sentence_bounds_keep_abbreviations_together():
    text = "We compete with Huawei Technologies Co. Ltd. and Intel. Sales rose."
    sents = [text[s:e] for s, e in sentence_bounds(text)]
    assert sents == ["We compete with Huawei Technologies Co. Ltd. and Intel.", "Sales rose."]


def test_sentence_bounds_join_hard_line_breaks():
    # 원문의 줄 맞춤으로 문장 중간에 들어간 줄바꿈은 잇고, 문장이 끝난 줄바꿈과 제목 줄은 나눈다
    text = (
        "Our largest customers, which together accounted for 40% of\n"
        "sales, are McKesson Corporation and Cardinal Health.\n"
        "Competition\n"
        "We face strong competition."
    )
    sents = [text[s:e] for s, e in sentence_bounds(text)]
    assert sents == [
        "Our largest customers, which together accounted for 40% of\n"
        "sales, are McKesson Corporation and Cardinal Health.",
        "Competition",
        "We face strong competition.",
    ]


def test_sentence_bounds_join_long_open_lines_but_not_rows_or_bullets():
    text = (
        "We purchase most of our chips from one predominant merchant silicon vendor,\n"
        "Broadcom, for our switching chips.\n"
        "Total revenue,\n"
        "Net income\n"
        "a) Describe the climate-related risks the organization has identified over the short,\n"
        "b) Describe the impact of climate-related risks.\n"
        "Representative customers include:\n"
        "lPuget Sound Energy, Inc.\n"
        "lSouthern Company"
    )
    sents = [text[s:e] for s, e in sentence_bounds(text)]
    assert sents == [
        "We purchase most of our chips from one predominant merchant silicon vendor,\n"
        "Broadcom, for our switching chips.",
        "Total revenue,",  # 짧은 줄(표 행)은 잇지 않는다
        "Net income",
        "a) Describe the climate-related risks the organization has identified over the short,",
        "b) Describe the impact of climate-related risks.",  # 항목 표시 앞은 잇지 않는다
        "Representative customers include:",
        "lPuget Sound Energy, Inc.",  # Wingdings 'l' 글머리표
        "lSouthern Company",
    ]


def test_mentions_spans_and_self_mentions(dictionary):
    text = (
        "Amazon is our largest customer. "
        "United Parcel Service delivers packages for Amazon.com and others."
    )
    t = run(dictionary, doc(2, text))
    m = t.mentions
    assert list(m["target_node"]) == ["cik:1", "cik:1"]  # 자기 자신(UPS)은 뺀다
    assert m["excluded"].isna().all()
    span = t.spans.set_index("span_id").loc[m["span_id"].iloc[1]]
    assert span["text"] == "United Parcel Service delivers packages for Amazon.com and others."
    assert text[span["char_start"] : span["char_end"]] == span["text"]
    assert text[m["char_start"].iloc[1] : m["char_end"].iloc[1]] == "Amazon.com"


def test_list_items_get_their_lead_in(dictionary):
    text = (
        "Our current competitors include:\n"
        "•cloud companies with internal teams, such as Amazon and others;\n"
        "•foundries such as TSMC."
    )
    t = run(dictionary, doc(2, text))
    spans = t.spans.set_index("span_id")
    for span_id in t.mentions["span_id"]:
        assert spans.at[span_id, "lead_text"] == "Our current competitors include:"


def test_prose_after_a_bulleted_list_gets_no_lead_in(dictionary):
    text = (
        "Our current competitors include:\n"
        "•cloud companies with internal teams;\n"
        "•foundries.\n"
        "In addition, we sell through Amazon and other online retailers under long-term agreements "
        "that renew each year and that cover most of our consumer products in North America.\n"
        "Key competitors:\n"
        "(1) Includes TSMC and other foundries in Asia."
    )
    t = run(dictionary, doc(2, text))
    spans = t.spans.set_index("span_id")
    lead = {row.target_node: spans.at[row.span_id, "lead_text"] for row in t.mentions.itertuples()}
    assert pd.isna(lead["cik:1"])  # 목록이 끝난 뒤의 일반 문장 (Amazon)
    assert lead["cik:1046179"] == "Key competitors:"  # 각주는 도입문을 붙인다 (TSMC)


def test_second_sentence_inside_a_bullet_keeps_the_lead_in(dictionary):
    text = (
        "In our business, we compete with:\n"
        "•Cloud vendors. These include Amazon and others with internal teams.\n"
        "•Foundries."
    )
    t = run(dictionary, doc(2, text))
    spans = t.spans.set_index("span_id")
    (span_id,) = t.mentions.loc[t.mentions["target_node"] == "cik:1", "span_id"]
    assert spans.at[span_id, "lead_text"] == "In our business, we compete with:"


def test_exec_bio_and_context_exclusions(dictionary):
    text = (
        "Our common stock is listed on the Nasdaq Global Select Market.\n"
        "We have a data agreement with Nasdaq.\n"
        "Information About Our Executive Officers\n"
        "Jane Doe (52) Chief Financial Officer (2020). Previously served as Vice President at "
        "Amazon.\n"
        "Competition\n"
        "We compete with Amazon in retail.\n"
    )
    assert len(exec_bio_ranges(text)) == 1
    m = run(dictionary, doc(3, text)).mentions
    assert m["excluded"].fillna("-").tolist() == ["context:listing", "-", "exec_bio", "-"]


def test_anonymous_customers_are_scoped_to_the_filing(dictionary):
    text = "Sales to one customer, Customer A, represented 13% of total revenue for fiscal 2024."
    t = run(dictionary, doc(2, text))
    assert list(t.mentions["target_node"]) == ["anon:2:acc-2:customer-a"]
    assert "anon:2:acc-2:customer-a" in set(t.nodes["node_id"])
    fig = t.figures.iloc[0]
    assert (fig["value"], fig["operator"], fig["period"]) == (13.0, "=", "2024")


def test_parse_figures_operator_and_unclear_attribution():
    (f,) = parse_figures(
        "Revenues from Apple, Samsung and Xiaomi each comprised 10% or more of our "
        "consolidated revenues in fiscal 2024.",
        n_targets=3,
    )
    assert (f["subject"], f["value"], f["operator"]) == ("each", 10.0, ">=")
    (f,) = parse_figures(
        "Our two largest customers together represented 34% of net sales.", n_targets=2
    )
    assert f["subject"] == "unclear" and f["value"] is None  # 합계는 회사별로 나누지 않는다
    figs = parse_figures(
        "Amazon represented 11.8%, 13.3% and 11.3% of revenues in 2023, 2022 and 2021.",
        n_targets=1,
    )
    assert all(f["value"] is None for f in figs)  # 여러 연도가 섞이면 비운다


def test_rating_tables_are_excluded_by_window_and_lead_in():
    aliases = AliasFile.model_validate(
        {
            "companies": {"NDAQ": {"names": ["Nasdaq"]}},
            "exclude_contexts": [
                {
                    "reason": "credit_rating",
                    "tickers": ["NDAQ"],
                    "window": 40,
                    "pattern": r"\bratings?\b|A\.M\. Best",
                },  # fmt: skip
                {"reason": "credit_rating", "tickers": ["NDAQ"], "pattern": r"\boutlook\b"},
            ],
        }
    )
    d = build_dictionary(UNIVERSE, aliases)
    text = (
        "Financial Strength Ratings\n"
        "A.M. Best (1)\n"
        "Nasdaq (3)\n"  # 표가 풀려 한 줄로 잘린 등급 표: 문장만 보면 문맥이 없다
        "Our outlook ratings are summarized as follows:\n"
        "•Nasdaq\n"  # 문맥은 도입문에만 있다
        "Other matters are described in the notes.\n"
        "In the exchange business we compete with many firms, and also with Nasdaq in data.\n"
    )
    m = run(d, doc(2, text)).mentions
    assert m["excluded"].fillna("-").tolist() == ["context:credit_rating"] * 2 + ["-"]

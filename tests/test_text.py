import time

import pytest

from tenksim.text import (
    assess,
    chunk_text,
    clean_section,
    duplicate_share,
    is_page_marker,
    is_table_row,
    item_token,
)


def word_count(text: str) -> int:
    return len(text.split())


# 아래 예시 줄들은 edgartools로 받은 실제 2024년 10-K에서 가져왔다.
@pytest.mark.parametrize(
    "line",
    [
        "12",
        "K-3",
        "- 4 -",
        "Page 7",
        "Apple Inc. | 2024 Form 10-K | 5",
        "2023 FORM 10-K 50",
        "Table of Contents",
        "Part I",
        "PART II",
    ],
)
def test_page_markers(line):
    assert is_page_marker(line)


@pytest.mark.parametrize(
    "line",
    ["Products", "iPhone", "Item 1. Business", "The Company sells iPhone in 2024."],
)
def test_not_page_markers(line):
    assert not is_page_marker(line)


@pytest.mark.parametrize(
    "line",
    [
        "North America186,751CCB141,640",
        "Cash and cash equivalents$609 $35",
        "Current receivables, inventories and contract assets551 495",
        "Grocery11.4%10.0%10.2%9.8%7.9%",
        "202320222021",
        "December 3120232022",
    ],
)
def test_table_rows(line):
    assert is_table_row(line)


@pytest.mark.parametrize(
    "line",
    [
        "Americans with Disabilities Act of 1990",
        "The Company had approximately 164,000 full-time equivalent employees.",
        "In 2023, revenue grew 12% to $4.2 billion, driven by services.",
        "Competition",
    ],
)
def test_not_table_rows(line):
    assert not is_table_row(line)


APPLE_LIKE = """Item 1.\xa0\xa0\xa0\xa0Business

Company Background

The Company designs, manufactures and markets smartphones, personal computers, tablets.

Apple Inc. | 2024 Form 10-K | 1

Net sales by region$12,345 $56,789

The Company&#8217;s fiscal year ends in September.
"""


def test_clean_section_removes_heading_footer_and_tables():
    c = clean_section(APPLE_LIKE, "business")
    assert c.heading_ok and not c.starts_with_note
    assert c.text.splitlines() == [
        "Company Background",
        "The Company designs, manufactures and markets smartphones, personal computers, tablets.",
        "The Company’s fiscal year ends in September.",
    ]
    assert c.n_page_lines == 1 and c.n_table_lines == 1
    assert 0 < c.table_char_share < 0.2


def test_clean_section_can_keep_tables():
    c = clean_section(APPLE_LIKE, "business", drop_table_rows=False, drop_page_markers=False)
    assert "Net sales by region$12,345 $56,789" in c.text
    assert "Apple Inc. | 2024 Form 10-K | 1" in c.text
    assert c.n_table_lines == 1  # 지우지 않아도 품질 판정용 통계는 센다


def test_ge_like_extraction_is_suspect():
    # GE 2024 10-K에서 edgartools가 Item 1 대신 재무제표 주석을 돌려준 실제 사례
    raw = (
        "NOTE 2. BUSINESSES HELD FOR SALE AND DISCONTINUED OPERATIONS. In the fourth quarter...\n"
        * 30
    )
    c = clean_section(raw, "business")
    assert not c.heading_ok
    assert c.starts_with_note
    assert assess("fetched", c, min_chars=100) == ("suspect", "note")


BODY = "\n" + "We design, manufacture and sell products to customers worldwide. " * 60


# S&P 500 2024년 10-K에서 실제로 나온 제목 형태들
@pytest.mark.parametrize(
    "head, section, heading_ok, wrong_item",
    [
        ("ITEM 1. BUSINESS", "business", True, None),
        ("Item 1: Business", "business", True, None),
        ("Items 1 and 2. Business and Properties", "business", True, None),
        ("Part I Item 1", "business", True, None),
        ("ITEM I. BUSINESS", "business", True, None),
        ("OUR BUSINESS", "business", True, None),
        ("(Dollars in millions except per share data)\nItem 1. Business", "business", True, None),
        ("Forward-Looking Statements", "business", False, None),
        ("Item 1C. Cybersecurity", "business", False, "1c"),
        ("Item 1A. Risk Factors", "business", False, "1a"),
        ("Item 1A. Risk Factors", "risk_factors", True, None),
        ("ITEM 1(A). RISK FACTORS", "risk_factors", True, None),
        ("Item 1(a) | Risk Factors", "risk_factors", True, None),
        ("PART IItem 1A", "risk_factors", True, None),
        ("RISK FACTORS. The following discussion", "risk_factors", True, None),
        ("Item 1. Business", "risk_factors", False, "1"),
    ],
)
def test_heading_detection(head, section, heading_ok, wrong_item):
    c = clean_section(head + BODY, section)
    assert c.heading_ok is heading_ok
    assert c.wrong_item == wrong_item
    if heading_ok:
        assert c.text.startswith("We design")  # 제목 줄(과 그 앞 머리글)은 지운다


def test_financial_statements_and_toc_are_suspect():
    glw = clean_section(
        "Consolidated Statements of Income\nNet sales\nCost of sales" + BODY, "business"
    )
    assert assess("fetched", glw, 100) == ("suspect", "financial_statement")
    # Intel처럼 'Form 10-K 상호참조 색인' 표를 Item 1로 돌려준 경우
    index = "\n".join(
        f"Item {n}. Something Pages {p}-{p + 5}"
        for n, p in [("1A", 48), ("2", 14), ("3", 108), ("7", 21), ("8", 70)]
    )
    intc = clean_section("Item Number Item\nPart I\nItem 1. Business:\n" + index + BODY, "business")
    assert intc.toc_like
    assert assess("fetched", intc, 100) == ("suspect", "toc")


@pytest.mark.parametrize(
    "line, token",
    [
        ("Item 1.A. Risk Factors", "1a"),
        ("ITEM 1A — RISK FACTORS", "1a"),
        ("Item 1 (a) Risk Factors", "1a"),
        ("Item 1. A Letter to Shareholders", "1"),
        ("Item 10. Directors", "10"),
        ("Items in this report are unaudited", None),
    ],
)
def test_item_token(line, token):
    assert item_token(line) == token


def test_business_is_cut_at_next_item_heading():
    # Bloom Energy처럼 Item 1 뒤에 Item 1A 전체가 붙어 온 경우
    raw = "Item 1. Business" + BODY + "\nITEM 1A — RISK FACTORS\nOur stock price may be volatile."
    c = clean_section(raw, "business")
    assert c.cut_at == "1a"
    assert "volatile" not in c.text
    assert assess("fetched", c, 100) == ("ok", "cut:1a")


def test_sentence_fragments_do_not_cut():
    # General Mills처럼 문장이 줄바꿈으로 잘게 끊겨 'Item 8'만 한 줄에 오는 경우
    raw = (
        "Item 1. Business\nSee the notes to the financial statements in\nItem 8\nof this report."
        + BODY
    )
    c = clean_section(raw, "business")
    assert c.cut_at is None
    assert "We design" in c.text


def test_repeated_short_lines_are_dropped_as_headers():
    pages = [f"CSX CORPORATION\nWe run trains across {n} states in the east." for n in range(5)]
    c = clean_section("Item 1. Business\n" + "\n".join(pages), "business")
    assert "CSX CORPORATION" not in c.text
    assert c.text.count("We run trains") == 5


def test_duplicate_share():
    risk = "\n".join(
        f"Risk paragraph number {i} about shareholder activism and proxy contests."
        for i in range(10)
    )
    business = "\n".join(
        risk.split("\n")[:4]
    )  # Devon Energy처럼 Item 1A의 일부가 Item 1로 들어온 경우
    assert duplicate_share(business, risk) == 1.0
    assert duplicate_share(risk, business) == 0.4
    assert duplicate_share("short", risk) == 0.0


def test_assess_statuses():
    good = clean_section("Item 1. Business\n" + "We sell things. " * 200, "business")
    no_head = clean_section("Unless the context otherwise requires..." + BODY, "business")
    short = clean_section("Item 1. Business\nWe sell things.", "business")
    assert assess("fetched", good, min_chars=1000) == ("ok", None)
    assert assess("fetched", no_head, min_chars=1000) == ("ok", "no_heading")
    assert assess("fetched", short, min_chars=1000) == ("too_short", "chars:15")
    assert assess("fetched", good, 1000, duplicate_of="risk_factors") == (
        "suspect",
        "duplicate:risk_factors",
    )
    assert assess("missing", None, min_chars=1000) == ("missing", None)
    assert assess("error", None, min_chars=1000) == ("error", None)


def test_chunk_text_respects_limit_and_keeps_all_words():
    text = "\n".join(f"Paragraph {i}. " + "word " * (i * 7 % 40 + 1) for i in range(60))
    chunks = chunk_text(text, max_tokens=50, count_tokens=word_count)
    assert all(0 < n <= 50 for _, n in chunks)
    assert all(word_count(c) <= 50 for c, _ in chunks)
    assert " ".join(c for c, _ in chunks).split() == text.split()


def test_chunk_text_splits_long_sentence_without_periods():
    # 구 버전(v1) 청커는 마침표 없는 긴 구간(표)을 만나면 무한 루프에 빠졌다
    prose = "We design and sell products. " * 300
    table = " ".join(f"{i:,} {i * 1.5:.1f}" for i in range(2000))
    text = prose + table + " End of table. More prose here."
    start = time.perf_counter()
    chunks = chunk_text(text, max_tokens=200, count_tokens=word_count)
    assert time.perf_counter() - start < 5
    assert all(word_count(c) <= 200 for c, _ in chunks)
    assert " ".join(c for c, _ in chunks).split() == text.split()


def test_chunk_text_hard_splits_giant_token():
    blob = "x" * 5000
    chunks = chunk_text(blob, max_tokens=100, count_tokens=len)
    assert "".join(c for c, _ in chunks) == blob
    assert all(len(c) <= 100 for c, _ in chunks)


def test_chunk_text_empty():
    assert chunk_text("", 10, word_count) == []
    assert chunk_text("\n\n  \n", 10, word_count) == []

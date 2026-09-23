import time

import pytest

from tenksim.text import assess, chunk_text, clean_section, is_page_marker, is_table_row


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
    assert assess("fetched", c, min_chars=100) == "suspect"


@pytest.mark.parametrize(
    "first, section, ok",
    [
        ("ITEM 1. BUSINESS", "business", True),
        ("Item 1: Business", "business", True),
        ("Items 1 and 2. Business and Properties", "business", True),
        ("Item 1A. Risk Factors", "business", False),
        ("Item 1A. Risk Factors", "risk_factors", True),
        ("RISK FACTORS. The following discussion", "risk_factors", True),
        ("Item 7. Management's Discussion", "risk_factors", False),
    ],
)
def test_heading_detection(first, section, ok):
    c = clean_section(first + "\n" + "Some text here. " * 100, section)
    assert c.heading_ok is ok


def test_assess_statuses():
    good = clean_section("Item 1. Business\n" + "We sell things. " * 200, "business")
    short = clean_section("Item 1. Business\nWe sell things.", "business")
    assert assess("fetched", good, min_chars=1000) == "ok"
    assert assess("fetched", short, min_chars=1000) == "too_short"
    assert assess("missing", None, min_chars=1000) == "missing"
    assert assess("error", None, min_chars=1000) == "error"


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

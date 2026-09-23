"""실제 EDGAR에 요청을 보내는 테스트. 기본 실행에서는 빠진다.

EDGAR_IDENTITY="이름 이메일" uv run pytest -m live
"""

import os

import pytest

from tenksim.filings import fetch_company
from tenksim.text import assess, clean_section

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("EDGAR_IDENTITY"), reason="EDGAR_IDENTITY가 필요합니다"),
]


def test_apple_2024_business_section():
    df = fetch_company(320193, 2024, ["business", "risk_factors"]).set_index("section")
    assert (df["status"] == "fetched").all()
    assert df.loc["business", "accession_number"] == "0000320193-24-000123"
    cleaned = clean_section(df.loc["business", "text_raw"], "business")
    assert assess("fetched", cleaned, min_chars=1000) == "ok"
    assert "iPhone" in cleaned.text
    assert "Form 10-K |" not in cleaned.text

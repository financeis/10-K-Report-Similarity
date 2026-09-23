from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from tenksim import filings


class FakeTenK:
    business = "Item 1. Business\nWe make widgets."
    risk_factors = None


def fake_filing(filed: date, accession: str):
    return SimpleNamespace(
        filing_date=filed,
        accession_number=accession,
        form="10-K",
        period_of_report="2023-12-31",
        filing_url=f"https://www.sec.gov/{accession}.htm",
        obj=lambda: FakeTenK(),
    )


class FakeCompany:
    calls: list = []
    filings_by_cik = {
        1: [fake_filing(date(2024, 2, 1), "a-early"), fake_filing(date(2024, 11, 1), "a-late")],
        2: [],
    }

    def __init__(self, cik):
        if cik == 3:
            raise ConnectionError("network down")
        FakeCompany.calls.append(cik)
        self.cik, self.name, self.sic, self.industry = cik, f"Co{cik}", "3571", "Computers"

    def get_filings(self, **kwargs):
        assert kwargs == {
            "form": "10-K",
            "amendments": False,
            "filing_date": "2024-01-01:2024-12-31",
        }
        return self.filings_by_cik[self.cik]


@pytest.fixture
def fake_edgar(monkeypatch):
    import edgar

    FakeCompany.calls = []
    monkeypatch.setattr(edgar, "Company", FakeCompany)
    monkeypatch.setenv("EDGAR_IDENTITY", "Test User test@example.com")


def test_fetch_company_picks_latest_filing_and_marks_missing_sections(fake_edgar):
    df = filings.fetch_company(1, 2024, ["business", "risk_factors"])
    assert df["accession_number"].unique().tolist() == ["a-late"]
    assert df.set_index("section")["status"].to_dict() == {
        "business": "fetched",
        "risk_factors": "missing",
    }
    assert df["sic"].iloc[0] == "3571"
    assert list(df.columns) == filings.RECORD_COLUMNS


def test_fetch_company_without_filing(fake_edgar):
    df = filings.fetch_company(2, 2024, ["business"])
    assert df["status"].tolist() == ["no_filing"]


def test_fetch_company_records_errors(fake_edgar):
    df = filings.fetch_company(3, 2024, ["business"])
    assert df["status"].tolist() == ["error"]
    assert "network down" in df["error"].iloc[0]


def test_ingest_caches_results_but_retries_errors(fake_edgar, tmp_path):
    out = filings.ingest([1, 2, 3], 2024, ["business"], tmp_path, workers=1)
    assert out.groupby("cik")["status"].first().to_dict() == {
        1: "fetched",
        2: "no_filing",
        3: "error",
    }
    assert sorted(p.stem for p in tmp_path.glob("*.parquet")) == ["1", "2"]

    FakeCompany.calls = []
    again = filings.ingest([1, 2, 3], 2024, ["business"], tmp_path, workers=1)
    assert FakeCompany.calls == []  # 1, 2는 캐시에서 읽고, 3은 생성자에서 다시 실패
    pd.testing.assert_frame_equal(
        out.drop(columns="error"), again.drop(columns="error"), check_dtype=False
    )


def test_ingest_refetches_when_new_section_requested(fake_edgar, tmp_path):
    filings.ingest([1], 2024, ["business"], tmp_path, workers=1)
    FakeCompany.calls = []
    filings.ingest([1], 2024, ["business", "risk_factors"], tmp_path, workers=1)
    assert FakeCompany.calls == [1]


def test_require_identity(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    with pytest.raises(RuntimeError, match="EDGAR_IDENTITY"):
        filings.require_identity()

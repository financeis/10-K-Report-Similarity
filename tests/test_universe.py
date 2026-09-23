import pandas as pd
import pytest

from tenksim import universe as universe_mod
from tenksim.config import UniverseConfig


def fake_sp500() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": [320193, 789019, 1067983],
            "ticker": ["AAPL", "MSFT", "BRK.B"],
            "name": ["Apple Inc.", "Microsoft", "Berkshire Hathaway"],
            "gics_sector": ["Information Technology"] * 2 + ["Financials"],
            "gics_sub_industry": ["Hardware", "Systems Software", "Multi-Sector Holdings"],
        }
    )


def test_ticker_filter(monkeypatch):
    monkeypatch.setattr(universe_mod, "load_sp500_wikipedia", fake_sp500)
    df = universe_mod.build_universe(UniverseConfig(tickers=["brk.b", "AAPL"]))
    assert df["ticker"].tolist() == ["AAPL", "BRK.B"]


def test_cik_overrides(monkeypatch):
    # 예: ExxonMobil은 2026년 지주회사 전환으로 CIK가 바뀌어, 2024년 10-K는 옛 CIK에 있다
    monkeypatch.setattr(universe_mod, "load_sp500_wikipedia", fake_sp500)
    df = universe_mod.build_universe(UniverseConfig(cik_overrides={"aapl": 1}))
    assert df.set_index("ticker")["cik"].to_dict() == {"AAPL": 1, "MSFT": 789019, "BRK.B": 1067983}


def test_unknown_ticker_is_an_error(monkeypatch):
    monkeypatch.setattr(universe_mod, "load_sp500_wikipedia", fake_sp500)
    with pytest.raises(ValueError, match="ZZZZ"):
        universe_mod.build_universe(UniverseConfig(tickers=["AAPL", "ZZZZ"]))


def test_load_csv_with_cik_and_bom(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("﻿Ticker,CIK,GICS_Sector\naapl,320193,IT\nmsft,789019,IT\n", encoding="utf-8")
    df = universe_mod.load_csv(path)
    assert df["ticker"].tolist() == ["AAPL", "MSFT"]
    assert df["cik"].tolist() == [320193, 789019]
    assert df["gics_sector"].tolist() == ["IT", "IT"]
    assert df["gics_sub_industry"].isna().all()


def test_load_csv_resolves_missing_cik(tmp_path, monkeypatch):
    import edgar

    monkeypatch.setattr(
        edgar, "get_ticker_to_cik_lookup", lambda: {"AAPL": 320193, "BRK-B": 1067983}
    )
    path = tmp_path / "u.csv"
    path.write_text("ticker\nAAPL\nBRK.B\n", encoding="utf-8")
    assert universe_mod.load_csv(path)["cik"].tolist() == [320193, 1067983]


def test_load_csv_requires_identifier(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("name\nApple\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ticker"):
        universe_mod.load_csv(path)

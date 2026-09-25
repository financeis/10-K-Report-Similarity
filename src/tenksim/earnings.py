"""실적 발표 날짜: EDGAR 8-K Item 2.02 (Results of Operations and Financial Condition).

회사는 실적 보도자료를 낼 때 8-K Item 2.02로 SEC에 함께 제출한다. 접수 시각(acceptance time,
UTC)을 뉴욕 시각으로 바꿔 시장이 반응할 수 있는 첫 거래일(0일)을 정한다.
- 장 마감(16:00, 조기 폐장일은 13:00) 이후 접수: 다음 거래일
- 그 전(장 전·장중) 접수: 그날. 휴장일이면 다음 거래일
- 8-K의 보고일(report date)이 휴장일이고 접수일보다 며칠 앞서며 8-K를 장 전이 아닌 때 냈으면(토요일에
  실적을 내고 월·화요일 오후에 8-K를 내는 버크셔 해서웨이 등) 보고일 뒤 첫 거래일을 0일로 둔다.
  보고일이 거래일이면 쓰지 않는다: 장 마감 뒤 보도자료를 내고 다음 날 아침 8-K를 내는 회사가 많아,
  보고일을 쓰면 오히려 0일이 이르게 잡힌다. 장 전에 낸 8-K는 그날 아침 보도자료와 함께 낸 것으로 본다.

Item 2.02에는 분기 실적 외에 잠정 실적·가이던스·운영 지표 공시도 들어간다. 모두 회사 고유 뉴스라
그대로 두고, 보고서에서 회사당 건수를 밝힌다.

회사별 목록을 data/runs/{name}/earnings_8k_{start}_{end}.parquet 에 저장하고 다시 받지 않는다.
지주회사 전환 등으로 CIK가 바뀐 회사는 기업 목록의 CIK와 지금 티커의 CIK를 모두 조회해 합친다.
SEC 요청 속도 제한과 HTTP 캐시(~/.edgar)는 edgartools가 처리한다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .filings import require_identity

log = logging.getLogger(__name__)

EARNINGS_ITEM = "2.02"
MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16
EARLY_CLOSE = {
    date(2024, 7, 3): 13, date(2024, 11, 29): 13, date(2024, 12, 24): 13,
    date(2025, 7, 3): 13, date(2025, 11, 28): 13, date(2025, 12, 24): 13,
}  # fmt: skip
"""NYSE 조기 폐장일(13:00 마감). 목록에 없는 해는 모두 16:00으로 본다."""
FILING_LAG_DAYS = 7
"""17:30 이후 접수는 다음 영업일 제출일로 잡히므로, 기간 끝 뒤 며칠치 제출도 받아 둔다."""
REPORT_DATE_MAX_LAG = 4
"""휴장일인 보고일이 접수일보다 이 날수(달력일) 이내로 앞설 때만 보고일을 쓴다."""

COLUMNS = ["cik", "filer_cik", "status", "error", "accession_number", "filing_date",
           "report_date", "accepted", "items"]  # fmt: skip
"""cik: 기업 목록의 CIK, filer_cik: 8-K를 낸 CIK. status: event(실적 8-K 한 건) / none(기간 안에 없음) /
error(받지 못함, 저장하지 않음). accepted는 UTC 시각을 시간대 표시 없이 저장한다."""


def _rows(cik: int, **fields) -> pd.DataFrame:
    return pd.DataFrame([{"cik": cik, **fields}]).reindex(columns=COLUMNS)


def fetch_company(cik: int, start: date, end: date, filer_cik: int | None = None) -> pd.DataFrame:
    """한 CIK(filer_cik, 없으면 cik)가 기간 안에 낸 8-K 가운데 Item 2.02가 든 것. 실패도 status로 기록한다."""
    from edgar import Company

    filer = filer_cik or cik
    try:
        span = f"{start.isoformat()}:{(end + timedelta(days=FILING_LAG_DAYS)).isoformat()}"
        filings = Company(filer).get_filings(form="8-K", amendments=False, filing_date=span)
        data = filings.data.to_pandas() if filings is not None else pd.DataFrame()
    except Exception as exc:  # 한 회사의 실패로 전체가 멈추지 않도록 기록만 한다
        log.warning("CIK %s: %s: %s", filer, type(exc).__name__, exc)
        return _rows(cik, filer_cik=filer, status="error", error=f"{type(exc).__name__}: {exc}")
    if data.empty:
        return _rows(cik, filer_cik=filer, status="none")
    items = data["items"].fillna("").astype(str)
    hit = data[items.str.split(",").apply(lambda xs: EARNINGS_ITEM in [x.strip() for x in xs])]
    if hit.empty:
        return _rows(cik, filer_cik=filer, status="none")
    accepted = pd.to_datetime(hit["acceptanceDateTime"], utc=True).dt.tz_localize(None)
    return pd.DataFrame(
        {
            "cik": cik,
            "filer_cik": filer,
            "status": "event",
            "error": None,
            "accession_number": hit["accession_number"].to_numpy(),
            "filing_date": pd.to_datetime(hit["filing_date"]).dt.date.astype(str).to_numpy(),
            "report_date": hit["reportDate"].astype(str).to_numpy(),
            "accepted": accepted.to_numpy(),
            "items": hit["items"].to_numpy(),
        }
    ).reindex(columns=COLUMNS)


def fetch_firm(cik: int, start: date, end: date, other_ciks: list[int]) -> pd.DataFrame:
    """회사 하나: 기업 목록의 CIK와 다른 CIK(바뀐 지주회사 등)를 모두 조회해 합친다."""
    parts = [fetch_company(cik, start, end, filer) for filer in [cik, *other_ciks]]
    rows = pd.concat(parts, ignore_index=True)
    if (rows["status"] == "error").any():
        return rows[rows["status"] == "error"].head(1)
    found = rows[rows["status"] == "event"].drop_duplicates("accession_number")
    return found if len(found) else _rows(cik, filer_cik=cik, status="none")


def current_ciks(firms: pd.DataFrame) -> dict[int, list[int]]:
    """기업 목록 CIK → 지금 티커로 찾은 다른 CIK (SEC company_tickers.json). 찾지 못하면 빈 목록."""
    try:
        from edgar import get_ticker_to_cik_lookup

        lookup = get_ticker_to_cik_lookup()
    except Exception as exc:
        log.warning("티커 → CIK 목록을 받지 못해 기업 목록의 CIK만 씁니다: %s", exc)
        return {}
    out = {}
    for cik, ticker in zip(firms["cik"], firms["ticker"], strict=True):
        now = lookup.get(ticker) or lookup.get(str(ticker).replace(".", "-"))
        if now is not None and int(now) != int(cik):
            out[int(cik)] = [int(now)]
    return out


def load_earnings_filings(
    firms: pd.DataFrame, start: date, end: date, cache_path: Path, *, refresh: bool = False
) -> pd.DataFrame:
    """회사들(cik, ticker)의 실적 8-K 목록 (COLUMNS). 캐시에 없는 회사만 받고, 받지 못한 회사는
    저장하지 않는다(다음 실행에서 다시 받는다)."""
    ciks = [int(c) for c in firms["cik"]]
    cached = pd.DataFrame(columns=COLUMNS)
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        if set(COLUMNS) - set(cached.columns):  # 예전 형식이면 모두 다시 받는다
            log.info("8-K 목록 캐시의 형식이 예전 것이라 다시 받습니다: %s", cache_path)
            cached = pd.DataFrame(columns=COLUMNS)
    todo = sorted(set(ciks) - set(cached["cik"].astype(int)))
    if todo:
        require_identity()
        log.info("8-K(Item %s) 목록 받기: %d개사 (%s ~ %s)", EARNINGS_ITEM, len(todo), start, end)
        logging.getLogger("edgar").setLevel(logging.ERROR)
        others = current_ciks(firms[firms["cik"].isin(todo)])
        if others:
            log.info("CIK가 바뀐 회사 %d곳은 두 CIK를 모두 조회합니다: %s", len(others), others)
        parts = [
            fetch_firm(cik, start, end, others.get(cik, []))
            for cik in tqdm(todo, desc="8-K", unit="사")
        ]
        fresh = pd.concat(parts, ignore_index=True).reindex(columns=COLUMNS)
        failed = fresh[fresh["status"] == "error"]
        if len(failed):
            log.warning("8-K 목록을 받지 못한 회사 %d곳 (다시 실행하면 재시도): %s",
                        len(failed), failed["cik"].tolist())  # fmt: skip
        keep = fresh[fresh["status"] != "error"]
        frames = [f for f in (cached, keep) if len(f)]
        cached = pd.concat(frames, ignore_index=True) if frames else keep
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)
        cached = pd.concat([cached, failed], ignore_index=True)
    cached = cached.copy()
    cached["cik"] = cached["cik"].astype(int)
    cached["accepted"] = pd.to_datetime(cached["accepted"], utc=True)
    return cached[cached["cik"].isin(set(ciks))].reset_index(drop=True)


def _local(accepted) -> pd.Series:
    return pd.to_datetime(pd.Series(accepted), utc=True).dt.tz_convert(MARKET_TZ)


def after_close(accepted) -> np.ndarray:
    """접수 시각이 그날 장 마감(조기 폐장일 반영) 이후인가."""
    local = _local(accepted)
    close = np.array([EARLY_CLOSE.get(d, MARKET_CLOSE_HOUR) for d in local.dt.date])
    return (local.dt.hour.to_numpy() >= close) & local.notna().to_numpy()


def day_zero(
    accepted, calendar: pd.DatetimeIndex, report_date=None
) -> tuple[np.ndarray, np.ndarray]:
    """접수 시각 → (반응 첫 거래일의 calendar 위치, 보고일로 앞당겼는가). calendar 밖이면 위치 -1.

    calendar는 가격이 있는 거래일(오름차순)이다. 0일의 수익률은 전 거래일 종가에서 재므로, 0일이
    calendar의 첫날이면(그 전 종가가 없음) 쓸 수 없어 -1로 둔다. report_date가 휴장일이고 접수일보다
    1~4일 앞서며 8-K를 장 전이 아닌 때 냈으면 보고일 뒤 첫 거래일을 0일로 둔다. 둘째 값은 그래서 0일이
    실제로 바뀐 이벤트다.
    """
    local = _local(accepted)
    day = local.dt.tz_localize(None).dt.normalize().dt.as_unit("ns")
    cal = pd.DatetimeIndex(calendar).as_unit("ns")
    pos = np.where(
        after_close(accepted),
        cal.searchsorted(day.to_numpy(), side="right"),
        cal.searchsorted(day.to_numpy(), side="left"),
    )
    early = np.zeros(len(pos), dtype=bool)
    if report_date is not None:
        report = pd.to_datetime(pd.Series(report_date).reset_index(drop=True), errors="coerce")
        report = report.dt.as_unit("ns")
        lag = (day.reset_index(drop=True) - report).dt.days
        closed = ~report.isin(cal) & report.notna()
        pre = (local.dt.hour * 60 + local.dt.minute).to_numpy() < 9 * 60 + 30
        use = ((lag >= 1) & (lag <= REPORT_DATE_MAX_LAG) & closed).to_numpy() & ~pre
        moved = np.where(use, cal.searchsorted(report.to_numpy(), side="left"), pos)
        early = moved != pos
        pos = moved
    return np.where((pos >= 1) & (pos < len(calendar)), pos, -1), early


def session(accepted: pd.Series) -> pd.Series:
    """접수 시각이 장 전(<9:30) / 장중 / 장 마감 뒤(조기 폐장일 반영) 중 어디인지 (뉴욕 시각)."""
    local = _local(accepted)
    pre = (local.dt.hour * 60 + local.dt.minute).to_numpy() < 9 * 60 + 30
    labels = np.select([after_close(accepted), pre], ["after", "pre"], "intraday")
    return pd.Series(labels, index=accepted.index)

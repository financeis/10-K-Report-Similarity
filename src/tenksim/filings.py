"""edgartools로 EDGAR에서 10-K 섹션 원문을 가져온다.

회사별 결과를 data/sections/{year}/{cik}.parquet 로 저장하므로, 중간에 끊겨도
다시 실행하면 받은 회사는 건너뛴다. SEC 요청 속도 제한과 HTTP 캐시(~/.edgar)는
edgartools가 처리한다.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

# SEC 공정 접근 정책은 초당 10회. edgartools 기본값(9)과 같게 둔다.
SEC_REQUESTS_PER_SECOND = 9

# edgartools TenK 객체의 속성 이름
SECTION_ATTRS = {"business": "business", "risk_factors": "risk_factors"}

RECORD_COLUMNS = [
    "cik",
    "year",
    "section",
    "status",
    "error",
    "company",
    "sic",
    "sic_description",
    "accession_number",
    "form",
    "filing_date",
    "period_of_report",
    "filing_url",
    "text_raw",
]


def require_identity() -> None:
    """EDGAR_IDENTITY가 없으면 edgartools가 입력 프롬프트를 띄우므로, 미리 막는다."""
    identity = os.environ.get("EDGAR_IDENTITY", "").strip()
    if "@" not in identity:
        raise RuntimeError(
            "EDGAR_IDENTITY 환경변수가 필요합니다. .env 파일에 "
            'EDGAR_IDENTITY="이름 이메일" 형식으로 넣으세요 '
            "(SEC 규정: https://www.sec.gov/os/accessing-edgar-data)."
        )


def _records(base: dict, sections: list[str], **fields) -> pd.DataFrame:
    rows = [{**base, "section": s, **fields} for s in sections]
    return pd.DataFrame(rows).reindex(columns=RECORD_COLUMNS)


def fetch_company(cik: int, year: int, sections: list[str]) -> pd.DataFrame:
    """한 회사의 해당 연도 10-K에서 섹션들을 가져온다. 실패도 status로 기록한다."""
    from edgar import Company

    base: dict = {"cik": cik, "year": year}
    try:
        company = Company(cik)
        base.update(company=company.name, sic=company.sic, sic_description=company.industry)
        filings = company.get_filings(
            form="10-K", amendments=False, filing_date=f"{year}-01-01:{year}-12-31"
        )
        if filings is None or len(filings) == 0:
            return _records(base, sections, status="no_filing", text_raw="")
        # 같은 해에 두 건이면(결산월 변경 등) 가장 늦게 제출된 것을 쓴다
        filing = max(filings, key=lambda f: f.filing_date)
        base.update(
            accession_number=filing.accession_number,
            form=filing.form,
            filing_date=str(filing.filing_date),
            period_of_report=str(filing.period_of_report),
            filing_url=filing.filing_url,
        )
        tenk = filing.obj()
        rows = []
        for section in sections:
            text = getattr(tenk, SECTION_ATTRS[section]) or ""
            status = "fetched" if text.strip() else "missing"
            rows.append({**base, "section": section, "status": status, "text_raw": text})
        return pd.DataFrame(rows).reindex(columns=RECORD_COLUMNS)
    except Exception as exc:  # 한 회사의 실패로 전체 수집이 멈추지 않도록 기록만 한다
        log.warning("CIK %s: %s: %s", cik, type(exc).__name__, exc)
        return _records(base, sections, status="error", error=f"{type(exc).__name__}: {exc}")


def _init_worker(requests_per_second: int) -> None:
    # edgartools는 import 시점에 이 값을 읽는다. spawn으로 띄운 새 프로세스라 아직 import 전이다.
    os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = str(requests_per_second)
    logging.getLogger("edgar").setLevel(logging.ERROR)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def _is_cached(path: Path, sections: list[str]) -> bool:
    if not path.exists():
        return False
    cached = set(pd.read_parquet(path, columns=["section"])["section"])
    return set(sections) <= cached


def ingest(
    ciks: list[int],
    year: int,
    sections: list[str],
    out_dir: Path,
    *,
    workers: int = 4,
    refresh: bool = False,
) -> pd.DataFrame:
    """회사별로 10-K 섹션을 받아 캐시하고, 이번 실행 대상 전체의 기록을 돌려준다.

    status가 error인 회사는 캐시하지 않는다. 다시 실행하면 자동으로 재시도된다.
    """
    require_identity()
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in ciks if refresh or not _is_cached(out_dir / f"{c}.parquet", sections)]
    log.info("10-K %d: %d cached, %d to fetch", year, len(ciks) - len(todo), len(todo))

    errors: dict[int, pd.DataFrame] = {}

    def handle(cik: int, df: pd.DataFrame) -> None:
        if (df["status"] == "error").any():
            errors[cik] = df
        else:
            df.to_parquet(out_dir / f"{cik}.parquet", index=False)

    progress = {"total": len(todo), "desc": f"10-K {year}"}
    if todo and workers <= 1:
        for cik in tqdm(todo, **progress):
            handle(cik, fetch_company(cik, year, sections))
    elif todo:
        # 시간은 대부분 HTML 파싱(CPU)에 들어가서 스레드로는 빨라지지 않는다. 프로세스를 쓴다.
        # edgartools의 속도 제한은 프로세스마다 따로 걸리므로 SEC 한도를 나눠 갖게 한다.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(max(1, SEC_REQUESTS_PER_SECOND // workers),),
        ) as pool:
            futures = {pool.submit(fetch_company, c, year, sections): c for c in todo}
            for fut in tqdm(as_completed(futures), **progress):
                handle(futures[fut], fut.result())
    if errors:
        log.warning("%d companies failed; re-run the same command to retry them", len(errors))

    frames = []
    for cik in ciks:
        if cik in errors:
            frames.append(errors[cik])
            continue
        df = pd.read_parquet(out_dir / f"{cik}.parquet")
        frames.append(df[df["section"].isin(sections)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RECORD_COLUMNS)

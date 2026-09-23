"""실험 설정(YAML) 스키마.

설정 파일 하나가 실험 한 번을 정의한다. 어떤 기업들(universe)의 어느 해 10-K에서
어떤 섹션을 가져와(filings), 어떤 방법들(methods)로 기업 벡터를 만들고,
무엇으로 검증할지(evaluation)를 적는다.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Section = Literal["business", "risk_factors"]
Label = Literal["gics_sector", "gics_sub_industry", "sic2", "sic3", "sic4"]


class _Strict(BaseModel):
    # 오타 난 키를 조용히 무시하면 의도와 다른 실험이 돌아가므로 거부한다
    model_config = ConfigDict(extra="forbid")


class UniverseConfig(_Strict):
    source: Literal["sp500_wikipedia", "csv"] = "sp500_wikipedia"
    path: Path | None = None
    """source가 csv일 때 읽을 파일. ticker 또는 cik 컬럼이 있어야 한다."""
    tickers: list[str] | None = None
    """이 티커들로만 제한한다 (스모크 테스트용)."""
    limit: int | None = None
    cik_overrides: dict[str, int] = Field(default_factory=dict)
    """티커 → 해당 연도 10-K를 낸 CIK. 지주회사 전환 등으로 CIK가 바뀐 회사를 바로잡는다."""

    @model_validator(mode="after")
    def _require_path_for_csv(self) -> UniverseConfig:
        if self.source == "csv" and self.path is None:
            raise ValueError("universe.source가 'csv'이면 universe.path가 필요합니다")
        return self


class FilingsConfig(_Strict):
    year: int
    """이 달력연도에 제출된 10-K 원본(10-K/A 제외)을 쓴다."""
    sections: list[Section] = ["business"]
    workers: int = 4


class TextConfig(_Strict):
    drop_page_markers: bool = True
    drop_table_rows: bool = True
    min_chars: int = 1000
    exclude_suspect: bool = True
    """섹션 추출이 의심스러운(제목이 안 맞는 등) 문서를 분석에서 뺀다."""


class MethodConfig(_Strict):
    name: str
    kind: Literal["tfidf", "sbert", "openai"]
    section: Section = "business"
    model: str | None = None
    dimensions: int | None = None
    max_tokens: int | None = None
    """청크 하나의 최대 토큰 수. 비우면 모델 한도를 쓴다."""
    doc_tokens: int | None = None
    """문서 앞부분 이만큼(토큰)만 쓴다. 참고 논문은 Item 1의 앞 512/1024/1536 토큰을 썼다."""
    batch_size: int = 32
    device: str | None = None
    # tfidf 전용
    min_df: int = 2
    max_df: float = 0.25

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        # 결과 디렉터리 이름으로 쓰인다
        if not re.fullmatch(r"[A-Za-z0-9._-]+", v):
            raise ValueError(f"method 이름은 영문/숫자/._- 만 쓸 수 있습니다: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_kind_options(self) -> MethodConfig:
        if self.kind in ("sbert", "openai") and not self.model:
            raise ValueError(f"method {self.name!r}: kind={self.kind}에는 model이 필요합니다")
        if self.kind == "tfidf" and self.doc_tokens:
            raise ValueError(f"method {self.name!r}: doc_tokens는 sbert/openai에서만 씁니다")
        return self


class ReturnsConfig(_Strict):
    start: date
    end: date
    market: str = "SPY"
    min_obs: int = 150
    """이보다 거래일 관측치가 적은 종목은 수익률 평가에서 뺀다."""


class EvaluationConfig(_Strict):
    k: list[int] = [1, 5, 10]
    labels: list[Label] = ["gics_sector", "gics_sub_industry", "sic2", "sic3"]
    returns: ReturnsConfig | None = None


class Config(_Strict):
    name: str
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    filings: FilingsConfig
    text: TextConfig = Field(default_factory=TextConfig)
    methods: list[MethodConfig]
    center_variants: bool = True
    """dense 방법마다 평균 벡터를 뺀(centering) 버전도 함께 평가한다."""
    top_k: int = 10
    """neighbors 결과에 저장할 이웃 수."""
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _check_methods(self) -> Config:
        names = [m.name for m in self.methods]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"methods 이름이 중복됩니다: {dup}")
        missing = sorted({m.section for m in self.methods} - set(self.filings.sections))
        if missing:
            raise ValueError(f"methods가 쓰는 섹션 {missing}이 filings.sections에 없습니다")
        return self

    @property
    def run_dir(self) -> Path:
        return self.data_dir / "runs" / self.name

    @property
    def sections_dir(self) -> Path:
        # 연도별로 모든 실행이 공유한다 (같은 10-K를 다시 받지 않도록)
        return self.data_dir / "sections" / str(self.filings.year)

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "embeddings.sqlite"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"{self.name}.md"

    def method_dir(self, name: str) -> Path:
        return self.run_dir / "methods" / name

    def method(self, name: str) -> MethodConfig:
        for m in self.methods:
            if m.name == name:
                return m
        raise KeyError(
            f"설정에 없는 method입니다: {name!r} (있는 것: {[m.name for m in self.methods]})"
        )


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))

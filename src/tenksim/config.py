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
    min_chars: int = 3000
    """정제 후 이보다 짧으면 too_short. 큰 은행은 Item 1A를 다른 곳으로 떠넘겨 몇백 자만 남기도 한다."""
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
    market: str = "equal_weight"
    """잔차를 낼 시장 수익률. equal_weight(유니버스 동일가중, 자기 제외) 또는 지수 티커(예: SPY).
    시가총액 가중 지수는 대형주끼리의 잔차 상관을 음(-)으로 치우치게 한다."""
    min_obs: int = 150
    """이보다 거래일 관측치가 적은 종목은 수익률 평가에서 뺀다."""


class EvaluationConfig(_Strict):
    k: list[int] = [1, 5, 10]
    primary_k: int | None = None
    """신뢰구간과 '가장 좋은 방법' 선택에 쓰는 k. 비우면 5(목록에 없으면 가장 큰 k)."""
    labels: list[Label] = ["gics_sector", "gics_sub_industry", "sic2", "sic3"]
    baseline: str | None = None
    """짝지은 차이를 잴 기준 방법. 비우면 methods의 첫 항목(보통 tfidf)."""
    n_boot: int = 1000
    """부트스트랩 재표집 횟수. 기업쌍 회귀는 계산량 때문에 이의 1/5만 쓴다."""
    returns: ReturnsConfig | None = None

    @model_validator(mode="after")
    def _check_primary_k(self) -> EvaluationConfig:
        if self.primary_k is not None and self.primary_k not in self.k:
            raise ValueError(
                f"evaluation.primary_k({self.primary_k})는 k 목록 {self.k}에 있어야 합니다"
            )
        return self

    @property
    def main_k(self) -> int:
        if self.primary_k is not None:
            return self.primary_k
        return 5 if 5 in self.k else max(self.k)


class EnsembleConfig(_Strict):
    """여러 방법의 유사도를 기업쌍 백분위로 바꿔 평균낸 조합."""

    name: str
    members: list[str]
    """방법 이름 또는 '이름+center' 변형."""
    weights: list[float] | None = None

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", v):
            raise ValueError(f"ensemble 이름은 영문/숫자/._- 만 쓸 수 있습니다: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_members(self) -> EnsembleConfig:
        if len(self.members) < 2:
            raise ValueError(f"ensemble {self.name!r}: members가 2개 이상이어야 합니다")
        if self.weights is not None and len(self.weights) != len(self.members):
            raise ValueError(f"ensemble {self.name!r}: weights와 members 개수가 다릅니다")
        return self


class SpanConfig(_Strict):
    include_lead_in: bool = True
    """목록 도입문("Our competitors include:")이나 대명사가 가리키는 앞 문장을 근거 구간에 넣는다."""
    max_chars: int = 1500
    """근거 구간 하나의 최대 글자 수. 넘으면 언급 주변만 자른다."""
    max_per_side: int = 6
    """후보 쌍 하나의 판정 입력에 넣을 근거 구간 수 (한 회사의 10-K 쪽마다)."""


class RelationsConfig(_Strict):
    """기업 관계도 (docs/relation-map-plan.md)."""

    aliases: Path = Path("configs/aliases.yaml")
    """회사명 별칭·외부 기업·문맥 제외 규칙."""
    similarity: str | None = None
    """1차 후보에 쓸 유사도 (방법·변형·앙상블 이름). 비우면 이름 언급만 후보로 쓴다."""
    top_k: int = 20
    spans: SpanConfig = Field(default_factory=SpanConfig)


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
    ensembles: list[EnsembleConfig] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    relations: RelationsConfig | None = None

    @model_validator(mode="after")
    def _check_methods(self) -> Config:
        names = [m.name for m in self.methods]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"methods 이름이 중복됩니다: {dup}")
        missing = sorted({m.section for m in self.methods} - set(self.filings.sections))
        if missing:
            raise ValueError(f"methods가 쓰는 섹션 {missing}이 filings.sections에 없습니다")
        variants = set(self.variant_names)
        for ens in self.ensembles:
            if ens.name in variants:
                raise ValueError(f"ensemble 이름 {ens.name!r}이 method 이름과 겹칩니다")
            unknown = [m for m in ens.members if m not in variants]
            if unknown:
                raise ValueError(
                    f"ensemble {ens.name!r}: 없는 변형 {unknown} (가능: {sorted(variants)})"
                )
        known = variants | {e.name for e in self.ensembles}
        if self.evaluation.baseline and self.evaluation.baseline not in known:
            raise ValueError(f"evaluation.baseline {self.evaluation.baseline!r}이 없는 이름입니다")
        rel = self.relations
        if rel and rel.similarity and rel.similarity not in known:
            raise ValueError(f"relations.similarity {rel.similarity!r}이 없는 이름입니다")
        return self

    @property
    def variant_names(self) -> list[str]:
        """평가되는 유사도 변형 이름: 방법마다 원래 버전과(dense면) '+center' 버전."""
        out = []
        for m in self.methods:
            out.append(m.name)
            if self.center_variants and m.kind != "tfidf":
                out.append(m.name + "+center")
        return out

    @property
    def baseline_name(self) -> str:
        return self.evaluation.baseline or self.methods[0].name

    def ensemble(self, name: str) -> EnsembleConfig | None:
        return next((e for e in self.ensembles if e.name == name), None)

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

    @property
    def relations_dir(self) -> Path:
        return self.run_dir / "relations"

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

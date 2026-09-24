"""관계 판정 (docs/relation-map-plan.md 7.3).

판정 단위는 검수 단위와 같다: 근거 구간 하나 × 그 구간이 언급한 회사 하나.
- X(filer): 근거 문장이 나온 10-K를 낸 회사 (검수 라벨의 doc)
- Y(named): 그 문장이 언급한 회사 (검수 라벨의 target)

요청 하나에 단위 하나를 묻는다. 질문은 늘 그 단위의 문장에 대한 것이고, 주변 글이나 같은 쌍의
다른 문장을 얼마나 함께 넣을지는 입력 방식(CONTEXTS)으로 고른다. Jev는 state에 관계없는 내용이
많을수록 정확도가 떨어진다고 알려져 있어(jev-1.13 알려진 한계), 어느 방식이 나은지는 검수 표본으로
비교해 정한다. 비용은 단위당 입력 약 1~3천 토큰이라 S&P 500 전체도 1달러 안쪽이다.

판정 모델은 RelationJudge 뒤에 두어 바꿔 끼운다(Jev, 텍스트 LLM). 모델이 무엇이든 답은 같은
형식(Judgement.answers)으로 저장하고, 채택·기각은 코드가 임계값으로 정한다(decide).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

Decision = Literal["yes", "no", "abstain"]


@dataclass(frozen=True)
class Party:
    node_id: str
    name: str
    """질문에 넣는 이름 ('The Coca-Cola Company')."""
    description: str
    """state에 넣는 한 줄 설명 (티커, 산업)."""


@dataclass(frozen=True)
class JudgeRequest:
    unit_id: str
    filer: Party
    named: Party
    source: str
    """어느 10-K의 어느 부분인지 ('10-K filed 2024-02-20, Item 1A. Risk Factors')."""
    passage: str
    """도입문 + 근거 구간. Y의 이름은 [[ ]]로 감싼다."""
    context_before: str | None = None
    """입력 방식이 nearby일 때: 같은 섹션에서 구간 바로 앞의 글."""
    context_after: str | None = None
    related: tuple[tuple[str, str], ...] = ()
    """입력 방식이 pair일 때: 같은 쌍의 다른 근거 구간 (출처, 본문)."""


CONTEXTS = ("span", "nearby", "pair")
"""판정 입력 방식.
- span: 근거 구간(도입문 + 문장)만
- nearby: + 같은 섹션의 앞뒤 문단
- pair: + 같은 쌍의 다른 근거 구간(두 회사 10-K 모두, 후보 단계에서 고른 것)
"""


@dataclass
class Judgement:
    unit_id: str
    model_id: str
    question_version: str
    answers: dict[str, dict]
    """질문 이름 → {"score": 0~1} (예/아니오) 또는 {"choice", "probabilities"} (보기 고르기).
    텍스트 LLM은 score 대신 {"decision": "yes"|"no"}를 적는다 (예를 1.0으로 바꾸지 않음)."""
    served_model: str | None = None
    """응답에 적힌 실제 모델 버전."""
    input_tokens: int | None = None
    cached: bool = field(default=False, compare=False)


class RelationJudge(Protocol):
    model_id: str

    def payload(self, request: JudgeRequest) -> dict:
        """모델에 보낼 입력 그대로. 캐시 키(입력 해시)를 여기서 만든다."""
        ...

    def judge(self, requests: Sequence[JudgeRequest]) -> list[Judgement | None]:
        """요청 순서대로 답한다. 실패한 요청은 None."""
        ...


def decide(answer: dict | None, accept: float, reject: float) -> Decision:
    """점수 → 채택(yes)·기각(no)·불확실(abstain). 점수가 없고 decision만 있으면 그것을 쓴다."""
    if not answer:
        return "abstain"
    score = answer.get("score")
    if score is None:
        return answer.get("decision") or "abstain"
    if score >= accept:
        return "yes"
    if score < reject:
        return "no"
    return "abstain"

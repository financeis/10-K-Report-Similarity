"""판정 질문 (docs/relation-map-plan.md 7.3).

v3.0부터 관계는 세 가지만 묻는다(저장소 주인 결정, 2026-09-25): 경쟁, 공급·협력(방향과 세부 유형
없이), 지분(방향 없이). 여기에 회사 식별과 시점을 더한다. 검수 라벨은 더 잘게 적혀 있어도
evaluate.LABEL_GROUPS로 묶어 채점한다.

질문에는 'A', 'B' 같은 기호 대신 실제 회사명을 넣는다. X는 10-K를 낸 회사, Y는 [[ ]]로 표시한
회사다. 질문은 늘 '이 문장(passage)'에 대해 묻고, 주변 글(context_before/after)이나 같은 쌍의 다른
문장(other_passages)은 입력 방식에 따라 state에 더할 뿐이다. 그래야 입력 방식끼리 비교할 수 있다.

질문 문구나 state 구성을 바꾸면 QUESTION_VERSION을 올린다. 캐시 키에는 입력 전체의 해시도
들어가므로 버전을 깜빡해도 예전 답을 잘못 쓰지는 않지만, 버전이 있어야 결과를 비교할 수 있다.
"""

from __future__ import annotations

from . import JudgeRequest

QUESTION_VERSION = "v3.1"

MARK_OPEN, MARK_CLOSE = "[[", "]]"

ENTITY_Q = "is_entity"
STATUS_Q = "status"
RELATION_QUESTIONS = ("competitor", "business", "equity")
YES_NO_QUESTIONS = (ENTITY_Q, *RELATION_QUESTIONS)
QUESTION_NAMES = (*YES_NO_QUESTIONS, STATUS_Q)


def state_for(req: JudgeRequest) -> dict:
    note = (
        f"{req.source}. In the passage, 'we', 'us', 'our' and 'the Company' refer to "
        f"{req.filer.name}. The name of {req.named.name} is marked with {MARK_OPEN} {MARK_CLOSE}."
    )
    if req.context_before or req.context_after:
        note += (
            " context_before and context_after are the text right before and after the passage "
            "in the same section; use them to understand the passage."
        )
    if req.related:
        note += (
            " other_passages are other sentences that mention the two companies, from either "
            "company's 10-K; use them to understand the passage."
        )
    state = {
        "filer": req.filer.description,
        "named_company": req.named.description,
        "source": note,
    }
    if req.context_before:
        state["context_before"] = req.context_before
    state["passage"] = req.passage
    if req.context_after:
        state["context_after"] = req.context_after
    if req.related:
        state["other_passages"] = [{"source": s, "text": t} for s, t in req.related]
    return state


def _noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def questions_for(req: JudgeRequest) -> dict[str, dict]:
    x, y = req.filer.name, req.named.name
    mark = f"{MARK_OPEN} {MARK_CLOSE}"
    return {
        ENTITY_Q: _noul(
            f"Does the name marked with {mark} in the passage refer to {y}?",
            true=f"The marked name means {y} itself, or a subsidiary, division or brand of {y}.",
            false=(
                "The marked name means something else: a person, a place or venue, a stock index "
                "or exchange, a product of another company, a separately owned company that only "
                "shares part of the name (for example an independent bottler, franchisee or joint "
                "venture company), or a different company with a similar name."
            ),
        ),
        "competitor": _noul(
            f"Does the passage show that {x} and {y} compete, or name one of them as a "
            "competitor of the other?"
        ),
        "business": _noul(
            f"Does the passage show that {x} and {y} do business with each other?",
            true=(
                f"One of them sells, provides or manufactures products or services for the other "
                f"(for example {y} is a customer, supplier, tenant, landlord, lender or service "
                f"provider of {x}, or the reverse), or they have a partnership, joint venture, "
                "licensing, distribution or co-development arrangement."
            ),
            false=(
                f"The passage does not show any business dealings between {x} and {y}. Being "
                "competitors, one owning shares of the other, or a one-time purchase or sale of "
                "assets (or an option to buy them) does not count by itself."
            ),
        ),
        "equity": _noul(
            f"Does the passage show that one of {x} and {y} owns or owned shares, an equity "
            "stake, or a controlling interest in the other?",
            true=(
                "One company holds or held shares or an ownership stake in the other, including "
                "being or having been its parent company."
            ),
            false="Neither company is said to own any part of the other.",
        ),
        STATUS_Q: {
            "type": "choice",
            "instructions": (
                f"As of the filing, what is the timing of the relationship between {x} and {y} "
                "described in the passage?"
            ),
            "criteria": {
                "current": "The relationship is in effect at the time of the filing.",
                "historical": "The relationship has ended, or is described only as a past event.",
                "planned": "The relationship has been announced or agreed but has not started yet.",
                "unclear": (
                    "The passage does not make the timing clear, or describes only a possible "
                    "future relationship."
                ),
            },
        },
    }

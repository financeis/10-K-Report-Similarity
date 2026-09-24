"""Jev(TypeSafe AI의 판정 전용 모델) 어댑터. 키는 .env의 TYPESAFE_API_KEY.

Jev는 글을 쓰지 않고 질문마다 점수(확률)만 돌려준다. 근거 문장은 코드가 골라 넣으므로 지어낸
인용이 나올 수 없다. 점수가 이 과제에서 보정됐는지는 검수 표본으로 확인하기 전까지 모른다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from tqdm import tqdm

from . import Judgement, JudgeRequest
from .questions import QUESTION_VERSION, questions_for, state_for

log = logging.getLogger(__name__)

PRICE_PER_MILLION_INPUT = 0.042
"""입력 100만 토큰당 달러 (2026-09 기준, 출력은 무료)."""


class JudgeError(RuntimeError):
    """계속해도 소용없는 오류 (키 없음, 인증 실패 등)."""


class JevJudge:
    def __init__(self, model: str = "jev-1.13.0", concurrency: int = 6, **client_options):
        self.model_id = model
        self.concurrency = concurrency
        self.client_options = client_options
        """AsyncTypeSafeClient에 그대로 넘긴다 (테스트에서 transport, api_key)."""

    def payload(self, request: JudgeRequest) -> dict:
        return {
            "model": self.model_id,
            "state": state_for(request),
            "questions": questions_for(request),
        }

    def judge(self, requests: Sequence[JudgeRequest]) -> list[Judgement | None]:
        return asyncio.run(self._judge_all(list(requests)))

    async def _judge_all(self, requests: list[JudgeRequest]) -> list[Judgement | None]:
        try:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError
        except ImportError as exc:
            raise JudgeError("Jev SDK가 없습니다: uv sync --extra judge") from exc
        options = {"timeout": 30.0, "retry": RetryPolicy(max_retries=4), **self.client_options}
        try:
            client = AsyncTypeSafeClient(model=self.model_id, **options)
        except TypeSafeError as exc:
            raise JudgeError(
                f"Jev 클라이언트를 만들 수 없습니다 (.env의 TYPESAFE_API_KEY): {exc}"
            ) from exc

        gate = asyncio.Semaphore(self.concurrency)
        bar = tqdm(total=len(requests), desc="Jev", unit="건", disable=len(requests) < 50)

        async def one(req: JudgeRequest) -> Judgement | None:
            async with gate:
                try:
                    p = self.payload(req)
                    resp = await client.system_one(state=p["state"], questions=p["questions"])
                except TypeSafeError as exc:
                    if getattr(exc, "status", None) in (401, 403):
                        raise JudgeError(
                            f"Jev 인증 실패 (.env의 TYPESAFE_API_KEY 확인): {exc}"
                        ) from exc
                    log.warning("Jev 판정 실패 %s: %s", req.unit_id, exc)
                    return None
                finally:
                    bar.update(1)
            return Judgement(
                unit_id=req.unit_id,
                model_id=self.model_id,
                question_version=QUESTION_VERSION,
                answers={name: _answer(a) for name, a in resp.answers.items()},
                served_model=resp.model,
                input_tokens=resp.usage.input_tokens,
            )

        async with client:
            try:
                return list(await asyncio.gather(*(one(r) for r in requests)))
            finally:
                bar.close()


def _answer(a) -> dict:
    if a.type == "noul":
        return {"score": float(a.noul)}
    if a.type == "choice":
        return {
            "choice": a.choice,
            "confidence": float(a.confidence),
            "probabilities": {k: float(v) for k, v in a.probabilities.items()},
        }
    return {
        "score": float(a.score),
        "probabilities": {str(k): v for k, v in a.probabilities.items()},
    }

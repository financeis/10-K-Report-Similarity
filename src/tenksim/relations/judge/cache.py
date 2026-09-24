"""판정 캐시 (judgements.sqlite). 키는 (모델 ID, 질문 버전, 입력 해시).

입력 해시는 모델에 보내는 내용 전체(state + 질문 + 모델)로 만든다. 그래서 근거 문장이나 질문
문구가 조금이라도 바뀌면 새로 판정하고, 같은 입력은 다시 돈을 내고 묻지 않는다.
graph.db와 달리 다시 만들지 않고 계속 쌓는다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import Judgement, JudgeRequest, RelationJudge
from .questions import QUESTION_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS judgements (
    model_id TEXT NOT NULL,
    question_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    unit_id TEXT NOT NULL,           -- 처음 판정한 단위 (같은 입력이면 다른 단위도 이 답을 쓴다)
    answers TEXT NOT NULL,           -- JSON {질문: {score} | {choice, probabilities} | {decision}}
    served_model TEXT,
    input_tokens INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_id, question_version, input_hash)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def input_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class JudgeRun:
    judgements: list[Judgement | None]
    """요청 순서대로. 실패한 요청은 None."""
    n_cached: int
    n_new: int
    n_failed: int
    input_tokens: int
    """이번에 새로 쓴 입력 토큰 (캐시에서 온 것은 빼고)."""


def judge_cached(
    judge: RelationJudge,
    requests: Sequence[JudgeRequest],
    cache: sqlite3.Connection,
    question_version: str = QUESTION_VERSION,
    ask: bool = True,
) -> JudgeRun:
    """ask=False면 캐시에 있는 것만 꺼낸다 (모델을 부르지 않음, 없는 것은 None·n_failed)."""
    keys = [input_hash(judge.payload(r)) for r in requests]
    found: dict[str, tuple] = {}
    for key in set(keys):
        row = cache.execute(
            "SELECT answers, served_model, input_tokens FROM judgements "
            "WHERE model_id = ? AND question_version = ? AND input_hash = ?",
            (judge.model_id, question_version, key),
        ).fetchone()
        if row:
            found[key] = row

    # 같은 입력이 여러 번 나와도 한 번만 묻는다
    todo: dict[str, JudgeRequest] = {}
    for key, req in zip(keys, requests, strict=True):
        if key not in found:
            todo.setdefault(key, req)
    if not todo:
        fresh = []
    elif ask:
        fresh = judge.judge(list(todo.values()))
    else:
        fresh = [None] * len(todo)
    now = datetime.now().isoformat(timespec="seconds")
    new: dict[str, Judgement] = {}
    with cache:
        for key, j in zip(todo, fresh, strict=True):
            if j is None:
                continue
            new[key] = j
            cache.execute(
                "INSERT OR REPLACE INTO judgements VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (judge.model_id, question_version, key, j.unit_id, json.dumps(j.answers),
                 j.served_model, j.input_tokens, now),
            )  # fmt: skip

    out: list[Judgement | None] = []
    for key, req in zip(keys, requests, strict=True):
        if key in found:
            answers, served, tokens = found[key]
            out.append(
                Judgement(req.unit_id, judge.model_id, question_version, json.loads(answers),
                          served, tokens, cached=True)
            )  # fmt: skip
        elif key in new:
            j = new[key]
            out.append(
                Judgement(req.unit_id, j.model_id, question_version, j.answers, j.served_model,
                          j.input_tokens)
            )  # fmt: skip
        else:
            out.append(None)
    return JudgeRun(
        judgements=out,
        n_cached=sum(k in found for k in keys),
        n_new=len(new),
        n_failed=len(todo) - len(new),
        input_tokens=sum(j.input_tokens or 0 for j in new.values()),
    )

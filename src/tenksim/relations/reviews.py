"""사람의 검수 기록 (reviews.sqlite). graph.db와 달리 다시 만들지 않고 계속 쌓는다.

지금은 표본 검수(docs/relation-map-plan.md 7.6)만 다룬다.
- 표본(samples): 섹터별로 고른 회사들과, 그 회사가 들어간 검수 단위 목록
- 검수 단위(unit): 근거 구간 하나 × 그 구간이 언급한 회사 하나. "이 문장은 X와 Y의 어떤 관계를 말하나"
- 구간 라벨(span_labels): 검수 결과. 고쳐 쓰면 새 줄을 추가하고, 단위마다 가장 최근 줄을 쓴다

X는 근거 문장이 나온 10-K를 낸 회사(doc), Y는 그 문장이 언급한 회사(target)다.
라벨에는 구간 원문과 해시를 함께 남겨, graph.db를 다시 만들어 위치나 ID가 바뀌어도 되짚을 수 있게 한다.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .mentions import text_hash

REVIEWS_SCHEMA_VERSION = 1

RELATIONS = (
    "competitor",
    "doc_supplies_target",
    "target_supplies_doc",
    "partner",
    "doc_owns_target",
    "target_owns_doc",
)
"""검수자가 고르는 관계. 여러 개를 함께 고를 수 있다. 아무것도 없으면 '관계를 말하지 않음'."""
ENTITY = ("yes", "no", "unsure")
STATUS = ("current", "historical", "planned", "unclear")
PARTNER_TYPES = ("licensing", "distribution", "joint_venture", "co_development", "other")
PURPOSES = ("dev", "confirm")

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,             -- dev(질문·임계값 조정) / confirm(합격 판단, 한 번만 사용)
    created_at TEXT NOT NULL,
    seed INTEGER NOT NULL,
    per_company_cap INTEGER NOT NULL,
    config TEXT NOT NULL,
    graph_created TEXT NOT NULL,       -- 표본을 뽑은 graph.db의 생성 시각
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_companies (
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    node_id TEXT NOT NULL,
    gics_sector TEXT,
    n_units_total INTEGER NOT NULL,    -- 상한을 적용하기 전 단위 수
    PRIMARY KEY (sample_id, node_id)
);
CREATE TABLE IF NOT EXISTS sample_units (
    sample_id TEXT NOT NULL REFERENCES samples(sample_id),
    ord INTEGER NOT NULL,
    unit_id TEXT NOT NULL,             -- <span_id>|<target_node>
    span_id TEXT NOT NULL,
    doc_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    pair_key TEXT NOT NULL,
    span_hash TEXT NOT NULL,           -- 구간 원문(도입문 포함)의 해시
    PRIMARY KEY (sample_id, ord),
    UNIQUE (sample_id, unit_id)
);
CREATE TABLE IF NOT EXISTS span_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT NOT NULL,
    sample_id TEXT,
    span_id TEXT NOT NULL,
    doc_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    accession TEXT NOT NULL,
    section TEXT NOT NULL,
    span_text TEXT NOT NULL,           -- 검수 당시 원문 (도입문 포함)
    span_hash TEXT NOT NULL,
    is_entity TEXT NOT NULL,           -- yes / no / unsure: 이 이름이 그 회사를 가리키나
    relations TEXT NOT NULL,           -- JSON 배열. [] = 관계를 말하지 않음
    partner_type TEXT,
    status TEXT,                       -- current / historical / planned / unclear
    skipped INTEGER NOT NULL DEFAULT 0,  -- 판단 보류
    note TEXT,
    blind INTEGER NOT NULL,            -- 모델 판정을 가린 채 검수했는가
    labeled_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS span_labels_unit ON span_labels(unit_id, label_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def unit_id(span_id: str, target_node: str) -> str:
    return f"{span_id}|{target_node}"


def span_full_text(text: str, lead_text: str | None) -> str:
    return f"{lead_text} … {text}" if lead_text else text


# ---------------------------------------------------------------- 표본 뽑기


@dataclass
class SampleResult:
    sample_id: str
    companies: list[dict]
    n_units: int


def create_sample(
    reviews: sqlite3.Connection,
    graph: sqlite3.Connection,
    *,
    sample_id: str,
    purpose: str,
    n_companies: int,
    per_company_cap: int = 40,
    seed: int = 0,
    config: str = "",
) -> SampleResult:
    """섹터별로 번갈아 회사를 뽑고, 그 회사가 들어간 검수 단위를 모은다.

    - 이미 다른 표본에 들어간 회사는 다시 뽑지 않는다(회사 단위 분리).
    - 이미 다른 표본에 들어간 단위와, 다른 표본 회사의 10-K에서 나온 단위는 넣지 않는다.
      같은 10-K 문장이 개발 표본과 확인 표본에 함께 들어가지 않게 하려는 것이다.
    - 많이 언급되는 회사(Microsoft 등)는 회사당 per_company_cap개까지만 무작위로 뽑는다.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose는 {PURPOSES} 중 하나여야 합니다: {purpose!r}")
    if reviews.execute("SELECT 1 FROM samples WHERE sample_id = ?", (sample_id,)).fetchone():
        raise ValueError(f"이미 있는 표본 이름입니다: {sample_id}")
    rng = random.Random(seed)

    used_companies = {r[0] for r in reviews.execute("SELECT node_id FROM sample_companies")}
    used_units = {r[0] for r in reviews.execute("SELECT unit_id FROM sample_units")}

    units_by_company: dict[str, list[dict]] = defaultdict(list)
    for r in graph.execute(
        """
        SELECT DISTINCT m.span_id, m.doc_node, m.target_node, s.text, s.lead_text
        FROM mentions m JOIN spans s USING (span_id)
        WHERE m.excluded IS NULL
        ORDER BY m.doc_node, m.span_id, m.target_node
        """
    ):
        u = dict(r)
        u["unit_id"] = unit_id(u["span_id"], u["target_node"])
        if u["unit_id"] in used_units or u["doc_node"] in used_companies:
            continue
        units_by_company[u["doc_node"]].append(u)
        units_by_company[u["target_node"]].append(u)

    sectors: dict[str, list[str]] = defaultdict(list)
    for node_id, sector in graph.execute(
        "SELECT node_id, gics_sector FROM nodes WHERE kind = 'company' AND analyzed = 1 "
        "ORDER BY node_id"
    ):
        if node_id not in used_companies and units_by_company.get(node_id):
            sectors[sector or "(none)"].append(node_id)
    for members in sectors.values():
        rng.shuffle(members)
    order = sorted(sectors)
    rng.shuffle(order)
    picked: list[str] = []
    while len(picked) < n_companies and any(sectors.values()):
        for sector in order:
            if sectors[sector] and len(picked) < n_companies:
                picked.append(sectors[sector].pop())

    chosen: dict[str, dict] = {}
    companies = []
    for node in picked:
        units = units_by_company[node]
        take = units if len(units) <= per_company_cap else rng.sample(units, per_company_cap)
        for u in take:
            chosen.setdefault(u["unit_id"], {**u, "company": node})
        companies.append({"node_id": node, "n_units_total": len(units), "n_units": len(take)})

    # 같은 회사 → 같은 쌍 → 문서 순서로 모아 두면 검수할 때 맥락을 잇기 쉽다
    ordered = sorted(
        chosen.values(),
        key=lambda u: (picked.index(u["company"]), _pair(u), u["doc_node"], _pos(u["span_id"])),
    )
    sector_of = dict(graph.execute("SELECT node_id, gics_sector FROM nodes"))
    graph_created = graph.execute("SELECT value FROM meta WHERE key = 'created'").fetchone()[0]
    with reviews:
        reviews.execute(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sample_id, purpose, _now(), seed, per_company_cap, config, graph_created,
             REVIEWS_SCHEMA_VERSION),
        )  # fmt: skip
        reviews.executemany(
            "INSERT INTO sample_companies VALUES (?, ?, ?, ?)",
            [(sample_id, c["node_id"], sector_of.get(c["node_id"]), c["n_units_total"])
             for c in companies],
        )  # fmt: skip
        reviews.executemany(
            "INSERT INTO sample_units VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (sample_id, i, u["unit_id"], u["span_id"], u["doc_node"], u["target_node"],
                 _pair(u), text_hash(span_full_text(u["text"], u["lead_text"])))
                for i, u in enumerate(ordered)
            ],
        )  # fmt: skip
    return SampleResult(sample_id, companies, len(ordered))


def _pair(u: dict) -> str:
    a, b = sorted((u["doc_node"], u["target_node"]))
    return f"{a}|{b}"


def _pos(span_id: str) -> tuple[str, int]:
    # s:<accession>:<section>:<start>-<end>
    *_, section, span = span_id.split(":")
    return section, int(span.split("-")[0])


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 라벨


class LabelError(ValueError):
    pass


def validate_label(
    is_entity: str,
    relations: list[str],
    status: str | None,
    partner_type: str | None,
    skipped: bool,
) -> None:
    if skipped:
        return
    if is_entity not in ENTITY:
        raise LabelError(f"is_entity는 {ENTITY} 중 하나여야 합니다")
    bad = [r for r in relations if r not in RELATIONS]
    if bad:
        raise LabelError(f"모르는 관계: {bad}")
    if len(set(relations)) != len(relations):
        raise LabelError("같은 관계를 두 번 골랐습니다")
    if is_entity != "yes" and relations:
        raise LabelError("같은 회사가 아니면 관계를 고를 수 없습니다")
    if relations and status not in STATUS:
        raise LabelError(f"관계를 골랐으면 시점({STATUS})도 골라야 합니다")
    if partner_type is not None and (
        "partner" not in relations or partner_type not in PARTNER_TYPES
    ):
        raise LabelError("협력 유형은 협력을 골랐을 때만, 정해진 값으로 적습니다")


def record_label(
    reviews: sqlite3.Connection,
    graph: sqlite3.Connection,
    *,
    unit: str,
    sample_id: str | None,
    is_entity: str,
    relations: list[str],
    status: str | None = None,
    partner_type: str | None = None,
    skipped: bool = False,
    note: str | None = None,
    blind: bool = True,
) -> int:
    validate_label(is_entity, relations, status, partner_type, skipped)
    span_id, target = unit.rsplit("|", 1)
    span = graph.execute(
        "SELECT doc_node, accession, section, text, lead_text FROM spans WHERE span_id = ?",
        (span_id,),
    ).fetchone()
    if span is None:
        raise LabelError(f"graph.db에 없는 근거 구간입니다: {span_id}")
    full = span_full_text(span[3], span[4])
    ordered = [r for r in RELATIONS if r in relations]
    with reviews:
        cur = reviews.execute(
            """
            INSERT INTO span_labels (unit_id, sample_id, span_id, doc_node, target_node, accession,
                section, span_text, span_hash, is_entity, relations, partner_type, status,
                skipped, note, blind, labeled_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (unit, sample_id, span_id, span[0], target, span[1], span[2], full, text_hash(full),
             is_entity, json.dumps(ordered), partner_type if "partner" in ordered else None,
             status if ordered else None, int(skipped), note or None, int(blind), _now(),
             REVIEWS_SCHEMA_VERSION),
        )  # fmt: skip
    return int(cur.lastrowid)


LATEST_LABELS = """
SELECT l.* FROM span_labels l
JOIN (SELECT unit_id, MAX(label_id) AS label_id FROM span_labels GROUP BY unit_id) last
  USING (unit_id, label_id)
"""
"""단위마다 가장 최근 라벨."""


def latest_label(reviews: sqlite3.Connection, unit: str) -> dict | None:
    r = reviews.execute(
        "SELECT * FROM span_labels WHERE unit_id = ? ORDER BY label_id DESC LIMIT 1", (unit,)
    ).fetchone()
    if r is None:
        return None
    out = dict(r)
    out["relations"] = json.loads(out["relations"])
    return out


def sample_progress(reviews: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in reviews.execute(
            f"""
            SELECT s.sample_id, s.purpose, s.created_at,
                   (SELECT COUNT(*) FROM sample_companies c WHERE c.sample_id = s.sample_id)
                       AS n_companies,
                   COUNT(u.unit_id) AS n_units,
                   COUNT(l.unit_id) AS n_labeled,
                   SUM(l.skipped) AS n_skipped
            FROM samples s
            LEFT JOIN sample_units u USING (sample_id)
            LEFT JOIN ({LATEST_LABELS}) l ON l.unit_id = u.unit_id
            GROUP BY s.sample_id
            ORDER BY s.created_at
            """
        )
    ]

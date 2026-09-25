"""graph.db (SQLite) 내보내기. 파이프라인이 만들고 웹앱은 읽기만 한다.

만들 때마다 새로 쓴다. 사람의 검수 기록은 여기 두지 않고 reviews.sqlite에 따로 쌓는다
(docs/relation-map-plan.md 7.5). 웹앱이 읽는 중일 수 있으므로 임시 파일에 다 쓴 뒤 바꿔 끼운다.

판정(unit_judgements)과 관계(edges, edge_evidence)는 판정 캐시(judgements.sqlite)에 있는 것만
채운다. 내보내면서 Jev를 부르지는 않는다 (`tenksim judge --all`이 판정한 뒤 다시 내보낸다).

스키마 v2 (2026-09-25): 판정을 단위(근거 구간 × 언급된 회사)별로 저장하고, 관계를 경쟁 · 공급·협력 ·
지분 세 가지(방향 없음)로 바꿨다.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import __version__
from .candidates import Candidates
from .mentions import MentionTables, text_hash
from .merge import RelationTables
from .names import company_node

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,          -- cik:<번호> / ext:<이름> / anon:<cik>:<accession>:<표기>
    kind TEXT NOT NULL,                -- company / external / anonymous
    cik INTEGER,
    ticker TEXT,
    name TEXT NOT NULL,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    analyzed INTEGER NOT NULL DEFAULT 0,  -- 이 실행에서 10-K를 분석한 회사
    parent_node TEXT                   -- 익명 노드: 공시한 회사
);
CREATE TABLE filings (
    accession TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    form TEXT,
    filing_date TEXT,
    period_of_report TEXT,
    filing_url TEXT
);
CREATE TABLE documents (               -- 정제 텍스트. 근거 위치(char_start/end)의 기준
    accession TEXT NOT NULL REFERENCES filings(accession),
    section TEXT NOT NULL,             -- business / risk_factors
    status TEXT NOT NULL,              -- 추출 품질 판정 (ok, suspect ...)
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (accession, section)
);
CREATE TABLE spans (
    span_id TEXT PRIMARY KEY,
    doc_node TEXT NOT NULL REFERENCES nodes(node_id),
    accession TEXT NOT NULL,
    section TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    lead_start INTEGER,                -- 목록 도입문 ("Our competitors include:")
    lead_end INTEGER,
    text TEXT NOT NULL,
    lead_text TEXT
);
CREATE TABLE mentions (
    mention_id TEXT PRIMARY KEY,
    doc_node TEXT NOT NULL REFERENCES nodes(node_id),
    target_node TEXT NOT NULL REFERENCES nodes(node_id),
    matched_name TEXT NOT NULL,
    name_kind TEXT NOT NULL,           -- base / alias / product / anonymous
    accession TEXT NOT NULL,
    section TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    excluded TEXT                      -- 제외 이유 (exec_bio, context:listing ...). NULL이면 사용
);
CREATE TABLE figures (                 -- 매출 비중 후보. 귀속이 불분명하면 value가 NULL
    figure_id TEXT PRIMARY KEY,
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    target_node TEXT NOT NULL REFERENCES nodes(node_id),
    subject TEXT,                      -- single / each / combined / unclear
    value REAL,
    operator TEXT,                     -- = / >= / ~ / <
    period TEXT,
    denominator TEXT,
    raw TEXT
);
CREATE TABLE candidates (              -- 방향 없는 쌍, node_a < node_b
    pair_key TEXT PRIMARY KEY,
    node_a TEXT NOT NULL REFERENCES nodes(node_id),
    node_b TEXT NOT NULL REFERENCES nodes(node_id),
    source TEXT NOT NULL,              -- similarity / mention / both
    rank_ab INTEGER,                   -- A의 유사 기업 중 B의 순위
    rank_ba INTEGER,
    similarity_pct REAL,
    mentions_ab INTEGER NOT NULL,      -- A의 10-K가 B를 언급한 횟수
    mentions_ba INTEGER NOT NULL,
    n_spans INTEGER NOT NULL,
    status TEXT NOT NULL               -- pending / accepted / uncertain / rejected / rejected_by_review / similar
);
CREATE TABLE candidate_spans (         -- 쌍별 판정 입력 근거 구간
    pair_key TEXT NOT NULL REFERENCES candidates(pair_key),
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    doc_node TEXT NOT NULL,
    ord INTEGER NOT NULL,
    cues TEXT,
    PRIMARY KEY (pair_key, span_id)
);
CREATE TABLE unit_judgements (         -- 단위별 판정 (d_*: yes / no / abstain, 회사 식별과 합친 결과)
    unit_id TEXT PRIMARY KEY,          -- <span_id>|<target_node>
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    doc_node TEXT NOT NULL,
    target_node TEXT NOT NULL REFERENCES nodes(node_id),
    pair_key TEXT NOT NULL,
    s_is_entity REAL, d_is_entity TEXT,
    s_competitor REAL, d_competitor TEXT,
    s_business REAL, d_business TEXT,
    s_equity REAL, d_equity TEXT,
    status TEXT,                       -- 시점 (Jev 답, 검수하지 않음)
    model_id TEXT NOT NULL,
    question_version TEXT NOT NULL
);
CREATE TABLE edges (                   -- 두 회사 관계. 방향 없음: src < dst
    edge_id TEXT PRIMARY KEY,          -- <relation>|<pair_key>
    src TEXT NOT NULL REFERENCES nodes(node_id),
    dst TEXT NOT NULL REFERENCES nodes(node_id),
    relation TEXT NOT NULL,            -- competitor / business / equity
    subtype TEXT,
    directed INTEGER NOT NULL,
    status TEXT,                       -- current / historical / planned / unclear
    basis TEXT NOT NULL,               -- disclosed
    decision TEXT NOT NULL,            -- accepted(채택) / uncertain(불확실: 검수 대기)
    validated INTEGER NOT NULL,        -- 이 관계 유형이 확인 표본에서 검증됐는가 (0이면 '검증 전')
    score REAL,                        -- 근거 문장 가운데 가장 높은 관계 점수
    n_evidence INTEGER NOT NULL,
    model_id TEXT,
    question_version TEXT,
    similarity_pct REAL,
    resid_corr REAL,
    as_of TEXT,                        -- 가장 점수가 높은 근거의 공시 accession
    review_state TEXT                  -- NULL / accepted / rejected / needs_recheck (운영 검수)
);
CREATE TABLE edge_evidence (           -- 관계를 채택(또는 불확실로 남긴) 근거 문장
    edge_id TEXT NOT NULL REFERENCES edges(edge_id),
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    target_node TEXT NOT NULL,         -- 이 문장이 언급한 쪽 (다른 쪽은 문장이 나온 10-K의 회사)
    question TEXT,
    score REAL,
    decision TEXT,
    ord INTEGER NOT NULL,              -- 점수 높은 순
    PRIMARY KEY (edge_id, span_id, target_node)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE INDEX nodes_ticker ON nodes(ticker);
CREATE INDEX mentions_doc ON mentions(doc_node);
CREATE INDEX mentions_target ON mentions(target_node);
CREATE INDEX mentions_span ON mentions(span_id);
CREATE INDEX spans_accession ON spans(accession, section);
CREATE INDEX figures_span ON figures(span_id);
CREATE INDEX candidates_a ON candidates(node_a);
CREATE INDEX candidates_b ON candidates(node_b);
CREATE INDEX candidate_spans_span ON candidate_spans(span_id);
CREATE INDEX edges_src ON edges(src);
CREATE INDEX edges_dst ON edges(dst);
CREATE INDEX edge_evidence_span ON edge_evidence(span_id);
CREATE INDEX unit_judgements_pair ON unit_judgements(pair_key);
"""


def build_nodes(
    tables: MentionTables, universe: pd.DataFrame, analyzed_ciks: set[int]
) -> pd.DataFrame:
    nodes = tables.nodes.copy()
    gics = universe.drop_duplicates("cik").set_index("cik")
    by_node = {company_node(c): c for c in gics.index}
    cik = nodes["node_id"].map(by_node)
    nodes["gics_sector"] = cik.map(gics["gics_sector"])
    nodes["gics_sub_industry"] = cik.map(gics["gics_sub_industry"])
    nodes["analyzed"] = cik.isin(analyzed_ciks).astype(int)
    anon = nodes["kind"] == "anonymous"
    nodes["parent_node"] = None
    nodes.loc[anon, "parent_node"] = "cik:" + nodes.loc[anon, "node_id"].str.split(":").str[1]
    return nodes


def export_graph(
    path: Path,
    *,
    tables: MentionTables,
    candidates: Candidates,
    universe: pd.DataFrame,
    documents: pd.DataFrame,
    meta: dict,
    relations: Callable[[sqlite3.Connection], tuple[RelationTables, dict]] | None = None,
) -> Path:
    """graph.db를 새로 만든다. documents는 분석에 쓴 문서만 (documents.parquet 형식).

    relations: 기본 테이블을 쓴 임시 DB를 받아 판정·관계 테이블과 meta에 더할 값을 돌려준다.
    판정 입력(근거 구간, 앞뒤 문단)을 graph.db 기준으로 만들기 때문에 이렇게 끼워 넣는다."""
    nodes = build_nodes(tables, universe, set(documents["cik"]))
    filings = (
        documents.rename(columns={"accession_number": "accession"})
        .assign(node_id=lambda d: d["cik"].map(company_node))
        .drop_duplicates("accession")[
            ["accession", "node_id", "form", "filing_date", "period_of_report", "filing_url"]
        ]
    )
    docs = documents.rename(columns={"accession_number": "accession"}).assign(
        text_hash=lambda d: d["text"].map(text_hash)
    )[["accession", "section", "status", "text_hash", "text"]]
    frames = {
        "nodes": nodes,
        "filings": filings,
        "documents": docs,
        "spans": tables.spans,
        "mentions": tables.mentions,
        "figures": tables.figures,
        "candidates": candidates.candidates,
        "candidate_spans": candidates.candidate_spans.rename(columns={"order": "ord"}),
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        **meta,
    }

    counts_rel: dict[str, int] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        for table, df in frames.items():
            _insert(con, table, df)
        if relations is not None:
            rel, rel_meta = relations(con)
            meta.update(rel_meta)
            for table in ("unit_judgements", "edges", "edge_evidence"):
                df = getattr(rel, table)
                if len(df):
                    _insert(con, table, df)
                counts_rel[table] = len(df)
            con.executemany(
                "UPDATE candidates SET status = ? WHERE pair_key = ?",
                [(s, k) for k, s in rel.candidate_status.items()],
            )
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [(k, v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()],
        )
        con.commit()
        _check_integrity(con)
    finally:
        con.close()
    os.replace(tmp, path)
    counts = {t: len(df) for t, df in frames.items()} | counts_rel
    log.info("graph.db written to %s: %s", path, counts)
    return path


def _insert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    df[cols].to_sql(table, con, if_exists="append", index=False)


def _check_integrity(con: sqlite3.Connection) -> None:
    """참조가 끊긴 행이 없는지. 스키마의 REFERENCES는 문서용이라 여기서 직접 확인한다."""
    checks = {
        "mentions.target_node": "SELECT COUNT(*) FROM mentions WHERE target_node NOT IN "
        "(SELECT node_id FROM nodes)",
        "mentions.span_id": "SELECT COUNT(*) FROM mentions WHERE span_id NOT IN "
        "(SELECT span_id FROM spans)",
        "candidates.node": "SELECT COUNT(*) FROM candidates WHERE node_a NOT IN "
        "(SELECT node_id FROM nodes) OR node_b NOT IN (SELECT node_id FROM nodes)",
        "candidate_spans.span_id": "SELECT COUNT(*) FROM candidate_spans WHERE span_id NOT IN "
        "(SELECT span_id FROM spans)",
        "edges.node": "SELECT COUNT(*) FROM edges WHERE src NOT IN (SELECT node_id FROM nodes) "
        "OR dst NOT IN (SELECT node_id FROM nodes) OR src >= dst",
        "edge_evidence.span_id": "SELECT COUNT(*) FROM edge_evidence WHERE span_id NOT IN "
        "(SELECT span_id FROM spans) OR edge_id NOT IN (SELECT edge_id FROM edges)",
        "unit_judgements.span_id": "SELECT COUNT(*) FROM unit_judgements WHERE span_id NOT IN "
        "(SELECT span_id FROM spans)",
        "spans.document": "SELECT COUNT(*) FROM spans s LEFT JOIN documents d ON "
        "s.accession = d.accession AND s.section = d.section WHERE d.text_hash IS NULL "
        "OR d.text_hash != s.text_hash",
    }
    bad = {name: n for name, sql in checks.items() if (n := con.execute(sql).fetchone()[0])}
    if bad:
        raise RuntimeError(f"graph.db 참조 오류: {bad}")

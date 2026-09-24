"""graph.db의 근거 구간 → 판정 요청 (검수 화면에 보이는 것과 같은 도입문 + 본문).

입력 방식(CONTEXTS)에 따라 주변 글을 더한다.
- nearby: 정제 본문에서 구간이 든 문단과 그 앞뒤 문단 (쪽마다 context_chars까지, 문장 단위로 자름)
- pair: 같은 쌍의 다른 근거 구간. 후보 단계에서 쌍마다 고른 판정 입력 구간(candidate_spans)을 쓴다
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

from ..reviews import span_full_text
from . import CONTEXTS, JudgeRequest, Party
from .questions import MARK_CLOSE, MARK_OPEN

SECTION_TITLE = {"business": "Item 1. Business", "risk_factors": "Item 1A. Risk Factors"}

_TRAILING_PARENS = re.compile(r"(.*?)\s*\(([^()]+)\)")
_SENTENCE_END = re.compile(r"[.!?;:][\"”’)]*\s+")


def display_name(name: str, kind: str = "company") -> str:
    """기업 목록의 이름을 문장에 넣기 좋게: 'Coca-Cola Company (The)' → 'The Coca-Cola Company',
    'Lilly (Eli)' → 'Eli Lilly', 'Alphabet Inc. (Class A)' → 'Alphabet Inc.'"""
    m = _TRAILING_PARENS.fullmatch(name)
    if kind != "company" or not m:
        return name
    base, paren = m.groups()
    return base if paren.startswith("Class ") else f"{paren} {base}"


def party(node: sqlite3.Row | dict) -> Party:
    name = display_name(node["name"], node["kind"])
    if node["kind"] == "company":
        parts = [f"ticker {node['ticker']}"] if node["ticker"] else []
        if node["gics_sub_industry"]:
            parts.append(f"industry: {node['gics_sub_industry']}")
        description = f"{name} ({'; '.join(parts)})" if parts else name
    elif node["kind"] == "anonymous":
        description = f"{name} (a party the filer does not name)"
    else:
        description = name
    return Party(node["node_id"], name, description)


def mark_names(text: str, ranges: Iterable[tuple[int, int]]) -> str:
    """text 안의 (start, end) 구간을 [[ ]]로 감싼다."""
    out, pos = [], 0
    for start, end in sorted(set(ranges)):
        if start < pos or not 0 <= start < end <= len(text):
            continue
        out += [text[pos:start], MARK_OPEN, text[start:end], MARK_CLOSE]
        pos = end
    out.append(text[pos:])
    return "".join(out)


def nearby_text(text: str, start: int, end: int, max_chars: int) -> tuple[str, str]:
    """[start, end) 앞뒤의 글: 그 구간이 든 문단 + 앞뒤 한 문단씩, 쪽마다 max_chars까지.
    잘라야 하면 문장 경계에서 자른다."""
    para_start = text.rfind("\n", 0, start) + 1
    lo = text.rfind("\n", 0, para_start - 1) + 1 if para_start > 0 else 0
    para_end = text.find("\n", end)
    para_end = len(text) if para_end < 0 else para_end
    hi = text.find("\n", para_end + 1) if para_end < len(text) else -1
    hi = len(text) if hi < 0 else hi

    before = text[max(lo, start - max_chars) : start]
    if start - max_chars > lo and (m := _SENTENCE_END.search(before)):
        before = before[m.end() :]
    after = text[end : min(hi, end + max_chars)]
    if end + max_chars < hi:
        ends = list(_SENTENCE_END.finditer(after))
        if ends:
            after = after[: ends[-1].start() + 1]
    return before.strip(), after.strip()


def requests_for_units(
    graph: sqlite3.Connection,
    unit_ids: Iterable[str],
    context: str = "span",
    context_chars: int = 1000,
    max_related: int = 12,
) -> list[JudgeRequest]:
    """검수 단위(<span_id>|<target_node>)마다 판정 요청 하나. graph.db에 없는 구간은 건너뛴다."""
    if context not in CONTEXTS:
        raise ValueError(f"context는 {CONTEXTS} 중 하나여야 합니다: {context!r}")
    cur = graph.cursor()
    cur.row_factory = sqlite3.Row
    parties: dict[str, Party] = {}
    documents: dict[tuple[str, str], str] = {}

    def get_party(node_id: str) -> Party:
        if node_id not in parties:
            row = cur.execute(
                "SELECT node_id, kind, ticker, name, gics_sub_industry FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            parties[node_id] = party(row) if row else Party(node_id, node_id, node_id)
        return parties[node_id]

    def get_document(accession: str, section: str) -> str:
        key = (accession, section)
        if key not in documents:
            row = cur.execute(
                "SELECT text FROM documents WHERE accession = ? AND section = ?", key
            ).fetchone()
            documents[key] = row[0] if row else ""
        return documents[key]

    def source(doc_node: str, section: str, filing_date: str | None) -> str:
        filed = f" filed {filing_date}" if filing_date else ""
        title = SECTION_TITLE.get(section, section)
        return f"{get_party(doc_node).name}'s 10-K annual report{filed}, {title}"

    out = []
    for unit in unit_ids:
        span_id, target = unit.rsplit("|", 1)
        span = cur.execute(
            "SELECT s.doc_node, s.accession, s.section, s.char_start, s.char_end, "
            "s.text, s.lead_text, f.filing_date "
            "FROM spans s LEFT JOIN filings f USING (accession) WHERE s.span_id = ?",
            (span_id,),
        ).fetchone()
        if span is None:
            continue
        ranges = [
            (m[0] - span["char_start"], m[1] - span["char_start"])
            for m in cur.execute(
                "SELECT char_start, char_end FROM mentions "
                "WHERE span_id = ? AND target_node = ? AND excluded IS NULL",
                (span_id, target),
            )
        ]
        extra: dict = {}
        if context == "nearby":
            doc = get_document(span["accession"], span["section"])
            before, after = nearby_text(doc, span["char_start"], span["char_end"], context_chars)
            extra = {"context_before": before or None, "context_after": after or None}
        elif context == "pair":
            a, b = sorted((span["doc_node"], target))
            extra["related"] = tuple(
                (source(r["doc_node"], r["section"], r["filing_date"]),
                 span_full_text(r["text"], r["lead_text"]))
                for r in cur.execute(
                    "SELECT s.doc_node, s.section, s.text, s.lead_text, f.filing_date "
                    "FROM candidate_spans c JOIN spans s USING (span_id) "
                    "LEFT JOIN filings f USING (accession) "
                    "WHERE c.pair_key = ? AND c.span_id != ? ORDER BY c.doc_node, c.ord LIMIT ?",
                    (f"{a}|{b}", span_id, max_related),
                ).fetchall()
            )  # fmt: skip
        out.append(
            JudgeRequest(
                unit_id=unit,
                filer=get_party(span["doc_node"]),
                named=get_party(target),
                source=source(span["doc_node"], span["section"], span["filing_date"]),
                passage=span_full_text(mark_names(span["text"], ranges), span["lead_text"]),
                **extra,
            )
        )
    return out

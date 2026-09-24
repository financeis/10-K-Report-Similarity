"""관계도 웹앱 백엔드 (FastAPI). 127.0.0.1에서만 실행한다.

graph.db는 읽기만 한다. 요청마다 읽기 전용 연결을 열고 닫는다. 파이프라인이 graph.db를 새로 만들어
바꿔 끼울 때 파일이 열려 있으면 Windows에서 교체가 실패하기 때문이다.

주소는 티커가 아니라 node_id를 쓴다(외부 기업·익명 고객도 같은 방식으로 찾기 위해).
node_id(cik:1045810)와 pair_key(cik:1|cik:2)는 경로에 넣을 때 URL 인코딩한다.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def create_app(graph_db: Path) -> FastAPI:
    app = FastAPI(title="tenksim relations", docs_url="/api/docs", openapi_url="/api/openapi.json")
    uri = f"file:{graph_db.resolve().as_posix()}?mode=ro"

    @contextmanager
    def db():
        if not graph_db.exists():
            raise HTTPException(503, f"{graph_db}가 없습니다. 먼저 `tenksim export`를 실행하세요")
        con = sqlite3.connect(uri, uri=True)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def rows(con, sql: str, *params) -> list[dict]:
        return [dict(r) for r in con.execute(sql, params)]

    def one(con, sql: str, *params) -> dict:
        r = con.execute(sql, params).fetchone()
        if r is None:
            raise HTTPException(404, "찾을 수 없습니다")
        return dict(r)

    @app.get("/api/meta")
    def meta():
        with db() as con:
            return {r["key"]: r["value"] for r in rows(con, "SELECT key, value FROM meta")}

    @app.get("/api/nodes")
    def search_nodes(q: str = "", limit: int = Query(30, le=200)):
        """티커·이름 검색. 검색어가 없으면 이름 언급 후보가 많은 회사부터."""
        q = q.strip()
        with db() as con:
            return rows(
                con,
                """
                SELECT n.node_id, n.kind, n.ticker, n.name, n.gics_sector, n.analyzed,
                       COUNT(c.pair_key) AS n_candidates,
                       COALESCE(SUM(c.source != 'similarity'), 0) AS n_mention_candidates
                FROM nodes n
                LEFT JOIN candidates c ON n.node_id IN (c.node_a, c.node_b)
                WHERE ?1 = '' OR n.ticker LIKE ?2 OR n.name LIKE ?2
                GROUP BY n.node_id
                HAVING n_candidates > 0
                ORDER BY (?1 != '' AND UPPER(n.ticker) = UPPER(?1)) DESC, n.analyzed DESC,
                         n_mention_candidates DESC, n.name
                LIMIT ?3
                """,
                q, f"%{q}%", limit,
            )  # fmt: skip

    @app.get("/api/nodes/{node_id}")
    def get_node(node_id: str):
        with db() as con:
            node = one(con, "SELECT * FROM nodes WHERE node_id = ?", node_id)
            node["filings"] = rows(
                con,
                "SELECT f.*, GROUP_CONCAT(d.section) AS sections FROM filings f "
                "LEFT JOIN documents d USING (accession) WHERE f.node_id = ? GROUP BY f.accession",
                node_id,
            )
            if node["parent_node"]:
                node["parent"] = one(
                    con, "SELECT node_id, ticker, name FROM nodes WHERE node_id = ?",
                    node["parent_node"],
                )  # fmt: skip
            return node

    @app.get("/api/nodes/{node_id}/candidates")
    def node_candidates(node_id: str, source: str | None = None, status: str | None = None):
        """이 회사가 들어간 후보 쌍. 순위·언급 수는 이 회사 기준으로 바꿔 돌려준다."""
        with db() as con:
            one(con, "SELECT node_id FROM nodes WHERE node_id = ?", node_id)
            out = rows(
                con,
                """
                SELECT c.*, o.node_id AS other_id, o.kind AS other_kind, o.ticker AS other_ticker,
                       o.name AS other_name, o.gics_sector AS other_sector,
                       CASE WHEN c.node_a = ?1 THEN c.rank_ab ELSE c.rank_ba END AS rank_mine,
                       CASE WHEN c.node_a = ?1 THEN c.rank_ba ELSE c.rank_ab END AS rank_theirs,
                       CASE WHEN c.node_a = ?1 THEN c.mentions_ab ELSE c.mentions_ba END
                           AS mentions_out,
                       CASE WHEN c.node_a = ?1 THEN c.mentions_ba ELSE c.mentions_ab END
                           AS mentions_in
                FROM candidates c
                JOIN nodes o ON o.node_id = CASE WHEN c.node_a = ?1 THEN c.node_b ELSE c.node_a END
                WHERE ?1 IN (c.node_a, c.node_b)
                  AND (?2 IS NULL OR c.source = ?2)
                  AND (?3 IS NULL OR c.status = ?3)
                ORDER BY rank_mine IS NULL, rank_mine, (mentions_out + mentions_in) DESC, o.name
                """,
                node_id, source, status,
            )  # fmt: skip
            return out

    @app.get("/api/candidates/{pair_key}")
    def get_candidate(pair_key: str):
        """후보 쌍 하나: 두 회사, 양쪽 10-K의 모든 언급 구간(판정 입력 여부 포함), 매출 비중."""
        with db() as con:
            cand = one(con, "SELECT * FROM candidates WHERE pair_key = ?", pair_key)
            nodes = {
                n["node_id"]: n
                for n in rows(
                    con,
                    "SELECT node_id, kind, ticker, name, gics_sector, gics_sub_industry "
                    "FROM nodes WHERE node_id IN (?, ?)",
                    cand["node_a"],
                    cand["node_b"],
                )  # fmt: skip
            }
            cand["a"], cand["b"] = nodes[cand["node_a"]], nodes[cand["node_b"]]
            spans = rows(
                con,
                """
                SELECT s.span_id, s.doc_node, s.accession, s.section, s.char_start, s.char_end,
                       s.text, s.lead_text, f.filing_date, f.filing_url,
                       cs.ord AS input_order, cs.cues,
                       GROUP_CONCAT(DISTINCT m.excluded) AS excluded,
                       SUM(m.excluded IS NULL) AS n_live
                FROM mentions m
                JOIN spans s USING (span_id)
                LEFT JOIN filings f ON f.accession = s.accession
                LEFT JOIN candidate_spans cs ON cs.pair_key = ?1 AND cs.span_id = s.span_id
                WHERE (m.doc_node = ?2 AND m.target_node = ?3)
                   OR (m.doc_node = ?3 AND m.target_node = ?2)
                GROUP BY s.span_id
                ORDER BY s.doc_node, s.section, s.char_start
                """,
                pair_key, cand["node_a"], cand["node_b"],
            )  # fmt: skip
            for s in spans:
                s["excluded"] = None if s["n_live"] else s["excluded"]
                s["highlights"] = _highlights(con, s, cand["node_a"], cand["node_b"])
            cand["spans"] = spans
            cand["figures"] = rows(
                con,
                "SELECT f.* FROM figures f JOIN spans s USING (span_id) "
                "WHERE (s.doc_node = ?1 AND f.target_node = ?2) "
                "   OR (s.doc_node = ?2 AND f.target_node = ?1)",
                cand["node_a"], cand["node_b"],
            )  # fmt: skip
            return cand

    @app.get("/api/spans/{span_id}/context")
    def span_context(span_id: str, window: int = Query(2500, le=20000)):
        """근거 구간 앞뒤의 정제 본문. 강조 위치는 돌려주는 excerpt 기준."""
        with db() as con:
            s = one(con, "SELECT * FROM spans WHERE span_id = ?", span_id)
            doc = one(
                con,
                "SELECT d.text, d.status, f.filing_url, f.filing_date, n.name, n.ticker "
                "FROM documents d JOIN filings f USING (accession) "
                "JOIN nodes n ON n.node_id = f.node_id "
                "WHERE d.accession = ? AND d.section = ?",
                s["accession"], s["section"],
            )  # fmt: skip
            text = doc.pop("text")
            first = s["lead_start"] if s["lead_start"] is not None else s["char_start"]
            start = max(0, first - window)
            end = min(len(text), s["char_end"] + window)
            # 줄 중간에서 자르지 않도록
            if start > 0 and (nl := text.find("\n", start, first)) >= 0:
                start = nl + 1
            if end < len(text) and (nl := text.rfind("\n", s["char_end"], end)) >= 0:
                end = nl
            marks = [
                {"start": s["char_start"] - start, "end": s["char_end"] - start, "kind": "span"}
            ]
            if s["lead_start"] is not None:
                marks.append(
                    {"start": s["lead_start"] - start, "end": s["lead_end"] - start, "kind": "lead"}
                )
            names = rows(
                con,
                "SELECT char_start, char_end FROM mentions WHERE span_id = ?",
                span_id,
            )
            marks += [
                {"start": m["char_start"] - start, "end": m["char_end"] - start, "kind": "name"}
                for m in names
            ]
            return {
                **doc,
                "span_id": span_id,
                "section": s["section"],
                "excerpt": text[start:end],
                "offset": start,
                "doc_length": len(text),
                "marks": marks,
            }

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _highlights(con, span: dict, a: str, b: str) -> list[dict]:
    """구간 텍스트 안에서 두 회사 이름의 위치 (span.text 기준)."""
    out = []
    for m in con.execute(
        "SELECT char_start, char_end, target_node, excluded FROM mentions "
        "WHERE span_id = ? AND target_node IN (?, ?)",
        (span["span_id"], a, b),
    ):
        start, end = m[0] - span["char_start"], m[1] - span["char_start"]
        if 0 <= start < end <= len(span["text"]):
            out.append({"start": start, "end": end, "node_id": m[2], "excluded": m[3]})
    return out

"""관계도 웹앱 백엔드 (FastAPI). 127.0.0.1에서만 실행한다.

graph.db는 읽기만 하고, 검수 기록은 reviews.sqlite에만 쓴다. 두 파일 모두 요청마다 연결을 열고
닫는다. 파이프라인이 graph.db를 새로 만들어 바꿔 끼울 때 파일이 열려 있으면 Windows에서 교체가
실패하기 때문이다.

주소는 티커가 아니라 node_id를 쓴다(외부 기업·익명 고객도 같은 방식으로 찾기 위해).
node_id(cik:1045810)와 pair_key(cik:1|cik:2)는 경로에 넣을 때 URL 인코딩한다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..relations import reviews as reviews_mod
from ..relations.mentions import text_hash

STATIC_DIR = Path(__file__).parent / "static"


class EdgeReviewIn(BaseModel):
    edge_id: str
    verdict: str
    note: str | None = None
    evidence_hash: str
    """검수자가 본 근거의 해시 (GET /api/edges가 준 값). 지금 근거와 다르면 409."""


class LabelIn(BaseModel):
    unit_id: str
    sample_id: str | None = None
    is_entity: str = "yes"
    relations: list[str] = Field(default_factory=list)
    skipped: bool = False
    note: str | None = None


def create_app(graph_db: Path, reviews_db: Path | None = None) -> FastAPI:
    reviews_db = reviews_db or graph_db.with_name("reviews.sqlite")
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
        """티커·이름 검색. 검색어가 없으면 관계(검수 반영)가 많은 회사부터."""
        q = q.strip()
        with db() as con:
            found = rows(
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
                """,
                q, f"%{q}%",
            )  # fmt: skip
            counts = relation_counts(con)
        for n in found:
            n["n_relations"] = counts.get(n["node_id"], 0)
        found.sort(
            key=lambda n: (
                not (q and (n["ticker"] or "").upper() == q.upper()), -n["analyzed"],
                -n["n_relations"], -n["n_mention_candidates"], n["name"],
            )
        )  # fmt: skip
        return found[:limit]

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
                ORDER BY rank_mine IS NULL, rank_mine, (mentions_out + mentions_in) DESC, o.name
                """,
                node_id, source,
            )  # fmt: skip
            by_pair = edges_by_pair(con, "WHERE ? IN (e.src, e.dst)", node_id)
            for c in out:
                c["status"] = _pair_status(by_pair.get(c["pair_key"], []), c["status"])
            return [c for c in out if status is None or c["status"] == status]

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
            edges = edges_by_pair(con, "WHERE e.src = ? AND e.dst = ?", cand["node_a"],
                                  cand["node_b"]).get(pair_key, [])  # fmt: skip
            cand["edges"] = [{k: e[k] for k in ("edge_id", "relation", "state")} for e in edges]
            cand["status"] = _pair_status(edges, cand["status"])
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

    # ------------------------------------------------------------ 표본 검수 (reviews.sqlite)

    @contextmanager
    def rdb():
        con = reviews_mod.connect(reviews_db)
        try:
            yield con
        finally:
            con.close()

    def node_names(con, ids) -> dict[str, dict]:
        ids = list(set(ids))
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        return {
            r["node_id"]: r
            for r in rows(
                con,
                f"SELECT node_id, kind, ticker, name, gics_sector FROM nodes "
                f"WHERE node_id IN ({marks})",
                *ids,
            )
        }

    @app.get("/api/review/samples")
    def review_samples():
        with rdb() as r:
            return reviews_mod.sample_progress(r)

    @app.get("/api/review/samples/{sample_id}")
    def review_sample(sample_id: str):
        """표본 하나: 회사 목록과 검수 단위 목록(검수 여부만, 모델 판정은 없음)."""
        with rdb() as r, db() as g:
            sample = one(r, "SELECT * FROM samples WHERE sample_id = ?", sample_id)
            companies = rows(r, "SELECT * FROM sample_companies WHERE sample_id = ?", sample_id)
            units = rows(
                r,
                f"""
                SELECT u.ord, u.unit_id, u.doc_node, u.target_node, u.pair_key,
                       l.unit_id IS NOT NULL AS labeled, COALESCE(l.skipped, 0) AS skipped
                FROM sample_units u
                LEFT JOIN ({reviews_mod.LATEST_LABELS}) l ON l.unit_id = u.unit_id
                WHERE u.sample_id = ? ORDER BY u.ord
                """,
                sample_id,
            )
            names = node_names(
                g, [c["node_id"] for c in companies] + [u["target_node"] for u in units]
                + [u["doc_node"] for u in units],
            )  # fmt: skip
            for c in companies:
                c["node"] = names.get(c["node_id"])
            return {"sample": sample, "companies": companies, "units": units, "nodes": names}

    @app.get("/api/review/samples/{sample_id}/units/{ord}")
    def review_unit(sample_id: str, ord: int):
        """검수 단위 하나. 가림 검수이므로 유사도·출처·단서 단어·모델 판정은 보내지 않는다."""
        with rdb() as r, db() as g:
            u = one(r, "SELECT * FROM sample_units WHERE sample_id = ? AND ord = ?", sample_id, ord)
            n = r.execute(
                "SELECT COUNT(*) FROM sample_units WHERE sample_id = ?", (sample_id,)
            ).fetchone()[0]
            span = g.execute(
                "SELECT s.*, f.filing_date, f.filing_url FROM spans s "
                "LEFT JOIN filings f USING (accession) WHERE s.span_id = ?",
                (u["span_id"],),
            ).fetchone()
            if span is None:
                raise HTTPException(
                    409,
                    "graph.db를 다시 만들면서 이 근거 구간이 사라졌습니다. 표본을 새로 뽑아 주세요",
                )
            span = dict(span)
            full = reviews_mod.span_full_text(span["text"], span["lead_text"])
            names = node_names(g, [u["doc_node"], u["target_node"]])
            highlights = [
                {"start": m["char_start"] - span["char_start"],
                 "end": m["char_end"] - span["char_start"]}
                for m in rows(
                    g,
                    "SELECT char_start, char_end FROM mentions "
                    "WHERE span_id = ? AND target_node = ? AND excluded IS NULL",
                    u["span_id"], u["target_node"],
                )
            ]  # fmt: skip
            return {
                "sample_id": sample_id,
                "ord": ord,
                "n_units": n,
                "unit_id": u["unit_id"],
                "doc": names.get(u["doc_node"]),
                "target": names.get(u["target_node"]),
                "span_id": u["span_id"],
                "section": span["section"],
                "filing_date": span["filing_date"],
                "filing_url": span["filing_url"],
                "text": span["text"],
                "lead_text": span["lead_text"],
                "highlights": [h for h in highlights if 0 <= h["start"] < h["end"]],
                "changed": text_hash(full) != u["span_hash"],
                "label": reviews_mod.latest_label(r, u["unit_id"]),
            }

    @app.post("/api/review/labels")
    def review_label(body: LabelIn):
        with rdb() as r, db() as g:
            if (
                body.sample_id
                and not r.execute(
                    "SELECT 1 FROM sample_units WHERE sample_id = ? AND unit_id = ?",
                    (body.sample_id, body.unit_id),
                ).fetchone()
            ):
                raise HTTPException(404, "이 표본에 없는 검수 단위입니다")
            try:
                label_id = reviews_mod.record_label(
                    r, g, unit=body.unit_id, sample_id=body.sample_id, is_entity=body.is_entity,
                    relations=body.relations, skipped=body.skipped, note=body.note, blind=True,
                )  # fmt: skip
            except reviews_mod.LabelError as exc:
                raise HTTPException(422, str(exc)) from exc
            return {"label_id": label_id, "label": reviews_mod.latest_label(r, body.unit_id)}

    # ------------------------------------------------------------ 관계 (2단계)

    def overlay_reviews(con, edges: list[dict]) -> list[dict]:
        """reviews.sqlite의 최신 관계 검수를 덧씌우고 화면용 상태(state)를 붙인다.
        export를 다시 하지 않아도 검수 결과가 바로 보이게 하려는 것이다."""
        latest = {}
        if reviews_db.exists():
            with rdb() as r:
                latest = reviews_mod.latest_edge_reviews(r)
        for e in edges:
            rv = latest.get(e["edge_id"])
            if rv is not None:
                e["review_state"] = reviews_mod.review_state(rv, evidence_hash(con, e["edge_id"]))
                e["review"] = {k: rv[k] for k in ("verdict", "note", "reviewed_at")}
            e["state"] = _edge_state(e)
        return edges

    def evidence_hash(con, edge_id: str) -> str:
        """관계의 근거 문장(도입문 포함) 해시. export의 apply_edge_reviews와 같은 방식이어야 한다."""
        return reviews_mod.evidence_hash(
            [
                (t["span_id"], reviews_mod.span_full_text(t["text"], t["lead_text"]))
                for t in rows(
                    con,
                    "SELECT v.span_id, s.text, s.lead_text FROM edge_evidence v "
                    "JOIN spans s USING (span_id) WHERE v.edge_id = ?",
                    edge_id,
                )
            ]
        )

    def relation_counts(con) -> dict[str, int]:
        """회사마다 화면에 보이는 관계 수 (모델 채택 + 검수로 확인, 검수로 거절한 것은 빼고)."""
        counts: dict[str, int] = {}
        for e in overlay_reviews(
            con, rows(con, "SELECT edge_id, src, dst, decision, review_state FROM edges")
        ):
            if e["state"] in ("accepted", "confirmed"):
                for n in (e["src"], e["dst"]):
                    counts[n] = counts.get(n, 0) + 1
        return counts

    def edges_by_pair(con, where: str, *params) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for e in overlay_reviews(
            con,
            rows(
                con,
                f"SELECT edge_id, src, dst, relation, decision, review_state FROM edges e {where}",
                *params,
            ),
        ):
            out.setdefault(f"{e['src']}|{e['dst']}", []).append(e)
        return out

    @app.get("/api/nodes/{node_id}/relations")
    def node_relations(node_id: str):
        """이 회사의 관계 전부 (채택·불확실·검수로 거절한 것). 거르기는 화면에서 한다."""
        with db() as con:
            one(con, "SELECT node_id FROM nodes WHERE node_id = ?", node_id)
            edges = rows(
                con,
                """
                SELECT e.*, o.node_id AS other_id, o.kind AS other_kind, o.ticker AS other_ticker,
                       o.name AS other_name, o.gics_sector AS other_sector
                FROM edges e
                JOIN nodes o ON o.node_id = CASE WHEN e.src = ?1 THEN e.dst ELSE e.src END
                WHERE ?1 IN (e.src, e.dst)
                ORDER BY e.relation, e.score DESC
                """,
                node_id,
            )  # fmt: skip
            return overlay_reviews(con, edges)

    @app.get("/api/edges/{edge_id}")
    def get_edge(edge_id: str):
        """관계 하나: 두 회사, 근거 문장(점수 높은 순, 두 회사 이름 강조), 매출 비중, 유사도."""
        with db() as con:
            edge = one(con, "SELECT * FROM edges WHERE edge_id = ?", edge_id)
            nodes = {
                n["node_id"]: n
                for n in rows(
                    con,
                    "SELECT node_id, kind, ticker, name, gics_sector, gics_sub_industry "
                    "FROM nodes WHERE node_id IN (?, ?)",
                    edge["src"],
                    edge["dst"],
                )  # fmt: skip
            }
            edge["a"], edge["b"] = nodes[edge["src"]], nodes[edge["dst"]]
            pair_key = f"{edge['src']}|{edge['dst']}"
            cand = con.execute(
                "SELECT rank_ab, rank_ba, source FROM candidates WHERE pair_key = ?", (pair_key,)
            ).fetchone()
            edge["candidate"] = dict(cand) if cand else None
            evidence = rows(
                con,
                """
                SELECT v.span_id, v.target_node, v.score, v.decision, v.ord,
                       s.doc_node, s.section, s.char_start, s.char_end, s.text, s.lead_text,
                       f.filing_date, f.filing_url, u.s_is_entity, u.status AS unit_status
                FROM edge_evidence v
                JOIN spans s USING (span_id)
                LEFT JOIN filings f ON f.accession = s.accession
                LEFT JOIN unit_judgements u ON u.unit_id = v.span_id || '|' || v.target_node
                WHERE v.edge_id = ?
                ORDER BY v.ord
                """,
                edge_id,
            )  # fmt: skip
            for ev in evidence:
                ev["highlights"] = [
                    h for h in _highlights(con, ev, edge["src"], edge["dst"]) if not h["excluded"]
                ]
            edge["evidence"] = evidence
            edge["evidence_hash"] = evidence_hash(con, edge_id)
            edge["thresholds"] = judge_thresholds(con)
            # 매출 비중은 거래 관계의 수치라 공급·협력에서만 보여준다 (경쟁 관계 밑에 두면 오해한다)
            edge["figures"] = [] if edge["relation"] != "business" else rows(
                con,
                "SELECT f.*, s.doc_node FROM figures f JOIN spans s USING (span_id) "
                "WHERE (s.doc_node = ?1 AND f.target_node = ?2) "
                "   OR (s.doc_node = ?2 AND f.target_node = ?1)",
                edge["src"], edge["dst"],
            )  # fmt: skip
            return overlay_reviews(con, [edge])[0]

    def judge_thresholds(con) -> dict:
        """graph.db를 만들 때 쓴 질문별 (채택, 기각) 임계값. 불확실한 이유를 설명하는 데 쓴다."""
        r = con.execute("SELECT value FROM meta WHERE key = 'judge'").fetchone()
        try:
            return json.loads(r[0]).get("thresholds", {}) if r else {}
        except ValueError:
            return {}

    @app.get("/api/review/edges")
    def review_edges(include_done: bool = False):
        """관계 검수 대기열: 모델이 불확실로 남긴 관계와, 검수 뒤 근거가 바뀐 관계.
        include_done이면 이미 검수한 불확실 관계도 함께 (검수 순서를 유지하려고)."""
        with db() as con:
            edges = rows(
                con,
                """
                SELECT e.*, a.name AS a_name, a.ticker AS a_ticker, a.gics_sector AS a_sector,
                       a.kind AS a_kind, b.name AS b_name, b.ticker AS b_ticker,
                       b.gics_sector AS b_sector, b.kind AS b_kind
                FROM edges e JOIN nodes a ON a.node_id = e.src JOIN nodes b ON b.node_id = e.dst
                WHERE e.decision = 'uncertain' OR e.review_state = 'needs_recheck'
                ORDER BY CASE e.relation WHEN 'competitor' THEN 0 WHEN 'business' THEN 1 ELSE 2 END,
                         e.score DESC
                """,
            )  # fmt: skip
            edges = overlay_reviews(con, edges)
            if not include_done:
                edges = [e for e in edges if e.get("review_state") in (None, "needs_recheck")]
            return edges

    @app.post("/api/review/edges")
    def save_edge_review(body: EdgeReviewIn):
        with rdb() as r, db() as g:
            edge = one(g, "SELECT * FROM edges WHERE edge_id = ?", body.edge_id)
            current = evidence_hash(g, body.edge_id)
            if body.evidence_hash != current:
                raise HTTPException(
                    409,
                    "그 사이 graph.db가 바뀌어 근거 문장이 달라졌습니다. 다시 불러와 확인해 주세요",
                )
            try:
                review_id = reviews_mod.record_edge_review(
                    r, edge=edge, evidence_hash=current, verdict=body.verdict, note=body.note
                )
            except reviews_mod.LabelError as exc:
                raise HTTPException(422, str(exc)) from exc
            return {"review_id": review_id, "edge": overlay_reviews(g, [edge])[0]}

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _edge_state(e: dict) -> str:
    """화면에 쓰는 최종 상태. 사람의 검수가 모델 판정보다 우선한다.
    confirmed(사람이 맞다고 함) / accepted(모델 채택) / uncertain(검수 대기) / rejected(사람이 아니라고 함)."""
    if e.get("review_state") == "rejected":
        return "rejected"
    if e.get("review_state") == "accepted":
        return "confirmed"
    return "accepted" if e["decision"] == "accepted" else "uncertain"


def _pair_status(edges: list[dict], fallback: str) -> str:
    """후보 쌍의 판정 결과를 관계 상태(검수 반영)로 정한다. 관계가 없으면 graph.db의 값."""
    states = {e["state"] for e in edges}
    if states & {"accepted", "confirmed"}:
        return "accepted"
    if "uncertain" in states:
        return "uncertain"
    if edges:
        return "rejected_by_review"  # 관계는 있었지만 사람이 모두 거절
    return fallback


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

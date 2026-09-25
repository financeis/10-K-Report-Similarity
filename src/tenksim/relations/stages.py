"""관계도 파이프라인 단계와 산출물 입출력. 산출물은 data/runs/{name}/relations/ 아래에 둔다."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import Config, RelationsConfig
from ..pipeline import load_documents, load_universe, similarity_for
from .candidates import Candidates, build_candidates
from .export import export_graph
from .mentions import SPAN_RULES, MentionTables, find_mentions, text_hash
from .names import build_dictionary, load_aliases

log = logging.getLogger(__name__)

MENTION_FILES = ("mentions", "spans", "figures", "nodes")


def relations_config(cfg: Config) -> RelationsConfig:
    if cfg.relations is None:
        raise SystemExit(f"{cfg.name}: 설정 파일에 relations: 절이 없습니다")
    return cfg.relations


def graph_path(cfg: Config) -> Path:
    return cfg.relations_dir / "graph.db"


def reviews_path(cfg: Config) -> Path:
    return cfg.relations_dir / "reviews.sqlite"


def judgements_path(cfg: Config) -> Path:
    return cfg.relations_dir / "judgements.sqlite"


def open_graph(cfg: Config) -> sqlite3.Connection:
    """graph.db를 읽기 전용으로 연다 (웹앱이 열어 둔 채여도 된다)."""
    path = graph_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 먼저 `tenksim export --all -c ...`를 실행하세요")
    con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def analyzed_documents(cfg: Config) -> pd.DataFrame:
    """이름 언급을 찾는 10-K 섹션: 추출 품질 판정을 통과한 것."""
    docs = load_documents(cfg)
    allowed = {"ok"} if cfg.text.exclude_suspect else {"ok", "suspect"}
    return docs[docs["status"].isin(allowed)]


def stage_mentions(cfg: Config) -> MentionTables:
    """분석 가능한 10-K 섹션에서 회사 이름 언급을 찾는다. 몇 초면 끝나므로 매번 새로 만든다."""
    rel = relations_config(cfg)
    universe = load_universe(cfg)
    docs = analyzed_documents(cfg)

    aliases = load_aliases(rel.aliases)
    dictionary = build_dictionary(universe, aliases)
    tables = find_mentions(
        docs,
        dictionary,
        include_lead_in=rel.spans.include_lead_in,
        max_span_chars=rel.spans.max_chars,
    )

    out = cfg.relations_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in MENTION_FILES:
        getattr(tables, name).to_parquet(out / f"{name}.parquet", index=False)
    m = tables.mentions
    live = m[m["excluded"].isna()]
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "aliases_hash": text_hash(rel.aliases.read_text(encoding="utf-8")),
        "span_rules": SPAN_RULES,
        "n_documents": int(len(docs)),
        "n_names": len(dictionary.entries),
        "n_mentions": int(len(m)),
        "n_mentions_used": int(len(live)),
        "excluded": {k: int(v) for k, v in m["excluded"].value_counts().items()},
        "n_spans": int(len(tables.spans)),
        "n_pairs": int(live[["doc_node", "target_node"]].drop_duplicates().shape[0]),
        "n_figures": int(len(tables.figures)),
    }
    (out / "mentions_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(
        "Mentions: %d (used %d, excluded %s) in %d documents; %d (doc, target) pairs; %d figures",
        meta["n_mentions"], meta["n_mentions_used"], meta["excluded"], meta["n_documents"],
        meta["n_pairs"], meta["n_figures"],
    )  # fmt: skip
    return tables


def load_mentions(cfg: Config) -> MentionTables:
    d = cfg.relations_dir
    missing = [n for n in MENTION_FILES if not (d / f"{n}.parquet").exists()]
    if missing:
        raise SystemExit(f"{d}에 {missing}가 없습니다. 먼저 `tenksim mentions`를 실행하세요")
    return MentionTables(*(pd.read_parquet(d / f"{n}.parquet") for n in MENTION_FILES))


def stage_candidates(cfg: Config, tables: MentionTables | None = None) -> Candidates:
    """유사도 상위 K ∪ 이름 언급 → 후보 쌍(pending)과 쌍별 판정 입력 근거 구간."""
    rel = relations_config(cfg)
    tables = tables if tables is not None else load_mentions(cfg)
    companies, sim = similarity_for(cfg, rel.similarity) if rel.similarity else (None, None)
    cands = build_candidates(
        tables, companies, sim, top_k=rel.top_k, max_per_side=rel.spans.max_per_side
    )
    out = cfg.relations_dir
    cands.candidates.to_parquet(out / "candidates.parquet", index=False)
    cands.candidate_spans.to_parquet(out / "candidate_spans.parquet", index=False)

    c = cands.candidates
    both_company = c["rank_ab"].notna()
    by_k = {}
    if sim is not None:
        mention = c["source"] != "similarity"
        best = c[["rank_ab", "rank_ba"]].min(axis=1)
        for k in sorted({10, 20, 30, rel.top_k}):
            by_k[k] = int((mention | (both_company & (best <= k))).sum())
            if k > rel.top_k:
                by_k[k] = None  # 후보를 top_k로 만들었으므로 더 큰 K는 셀 수 없다
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "similarity": rel.similarity,
        "top_k": rel.top_k,
        "n_candidates": int(len(c)),
        "by_source": {k: int(v) for k, v in c["source"].value_counts().items()},
        "n_with_external_or_anonymous": int((~both_company).sum()),
        "n_candidates_by_k": by_k,
        "n_candidate_spans": int(len(cands.candidate_spans)),
    }
    (out / "candidates_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(
        "Candidates: %d pairs %s; %d involve external/anonymous nodes; %d evidence spans",
        meta["n_candidates"], meta["by_source"], meta["n_with_external_or_anonymous"],
        meta["n_candidate_spans"],
    )  # fmt: skip
    return cands


def load_candidates(cfg: Config) -> Candidates:
    d = cfg.relations_dir
    if not (d / "candidates.parquet").exists():
        raise SystemExit(f"{d}에 후보가 없습니다. 먼저 `tenksim candidates`를 실행하세요")
    return Candidates(
        pd.read_parquet(d / "candidates.parquet"), pd.read_parquet(d / "candidate_spans.parquet")
    )


def stage_export(
    cfg: Config, tables: MentionTables | None = None, cands: Candidates | None = None
) -> Path:
    """언급·후보를 웹앱이 읽는 graph.db로 내보낸다."""
    rel = relations_config(cfg)
    tables = tables if tables is not None else load_mentions(cfg)
    cands = cands if cands is not None else load_candidates(cfg)
    d = cfg.relations_dir
    meta = {
        "config": cfg.name,
        "filings_year": str(cfg.filings.year),
        "similarity": rel.similarity or "",
        "top_k": str(rel.top_k),
        "mentions": json.loads((d / "mentions_meta.json").read_text()),
        "candidates": json.loads((d / "candidates_meta.json").read_text()),
    }
    return export_graph(
        graph_path(cfg),
        tables=tables,
        candidates=cands,
        universe=load_universe(cfg),
        documents=analyzed_documents(cfg),
        meta=meta,
        relations=lambda con: relations_from_cache(cfg, con),
    )


def stage_relations(cfg: Config, judge: bool = True) -> Path:
    """이름 언급 → 후보 → graph.db → (판정 → 관계를 합쳐 graph.db 다시 쓰기)."""
    tables = stage_mentions(cfg)
    cands = stage_candidates(cfg, tables)
    path = stage_export(cfg, tables, cands)
    if not judge:
        return path
    try:
        return stage_judge_all(cfg, tables=tables, cands=cands)
    except SystemExit as exc:  # 키가 없거나 인증 실패: 판정 없이 둔다
        log.warning("판정을 건너뜁니다: %s", exc)
        return path


def all_units(con: sqlite3.Connection) -> pd.DataFrame:
    """판정할 단위 전부: 제외되지 않은 이름 언급의 (근거 구간, 언급된 회사)."""
    units = pd.read_sql(
        "SELECT DISTINCT span_id, doc_node, target_node, accession FROM mentions "
        "WHERE excluded IS NULL ORDER BY doc_node, span_id, target_node",
        con,
    )
    units["unit_id"] = units["span_id"] + "|" + units["target_node"]
    a, b = units["doc_node"], units["target_node"]
    units["pair_key"] = a.where(a < b, b) + "|" + b.where(a < b, a)
    return units


def _unit_requests(cfg: Config, con: sqlite3.Connection, units: pd.DataFrame):
    from .judge.inputs import requests_for_units

    j = relations_config(cfg).judge
    return requests_for_units(
        con, units["unit_id"], context=j.context, context_chars=j.context_chars
    )


def relations_from_cache(cfg: Config, con: sqlite3.Connection):
    """graph.db(쓰는 중)의 모든 단위에 대해 캐시에 있는 판정만 꺼내 관계를 만든다. 모델은 부르지 않는다."""
    from .judge.cache import connect, judge_cached
    from .merge import build_relations, candidate_status

    rel = relations_config(cfg)
    units = all_units(con)
    judge = make_judge(cfg)
    run = judge_cached(
        judge, _unit_requests(cfg, con, units), connect(judgements_path(cfg)), ask=False
    )
    judgements = {j.unit_id: j for j in run.judgements if j is not None}
    candidates = pd.read_sql("SELECT pair_key, similarity_pct FROM candidates", con)
    from .reviews import span_full_text

    span_text = {
        sid: span_full_text(t, lead)
        for sid, t, lead in con.execute("SELECT span_id, text, lead_text FROM spans")
    }
    tables = build_relations(
        units, judgements, rel.judge.threshold, candidates, set(rel.judge.validated),
        span_text=span_text,
    )  # fmt: skip
    n_reviewed = apply_edge_reviews(cfg, con, tables)
    if n_reviewed:  # 검수로 확인·거절한 관계를 후보 상태에도 반영
        tables.candidate_status = candidate_status(
            candidates, units, tables.unit_judgements, tables.edges
        )
    e = tables.edges
    summary = {
        r: {d: int(((e["relation"] == r) & (e["decision"] == d)).sum()) for d in ("accepted", "uncertain")}
        for r in ("competitor", "business", "equity")
    }  # fmt: skip
    if run.n_failed:
        log.warning(
            "판정하지 않은 단위 %d개는 관계에 넣지 않았습니다. `tenksim judge --all -c ...`로 판정하세요",
            run.n_failed,
        )
    log.info("관계: 판정한 단위 %d/%d, %s", len(judgements), len(units), summary)
    meta = {
        "judge": judge_settings(cfg),
        "validated_relations": rel.judge.validated,
        "units": {"total": len(units), "judged": len(judgements)},
        "edges": summary,
        "edge_reviews": n_reviewed,
    }
    return tables, meta


def edge_evidence_texts(con: sqlite3.Connection, evidence: pd.DataFrame) -> dict[str, list]:
    """edge_id → [(span_id, 도입문 포함 원문)] (근거 해시용). evidence: edge_evidence 형식."""
    if evidence.empty:
        return {}
    from .reviews import span_full_text

    text = {
        sid: span_full_text(t, lead)
        for sid, t, lead in con.execute("SELECT span_id, text, lead_text FROM spans")
    }
    out: dict[str, list] = {}
    for edge_id, span_id in zip(evidence["edge_id"], evidence["span_id"], strict=True):
        out.setdefault(edge_id, []).append((span_id, text.get(span_id, "")))
    return out


def apply_edge_reviews(cfg: Config, con: sqlite3.Connection, tables) -> dict:
    """reviews.sqlite의 관계 검수를 edges.review_state에 반영한다 (사람의 결정이 우선, 7.5).
    근거가 바뀐 관계는 needs_recheck로 두고, 사라진 관계의 검수는 세기만 한다."""
    from . import reviews

    path = reviews_path(cfg)
    if not path.exists() or tables.edges.empty:
        return {}
    with closing(reviews.connect(path)) as rev:
        latest = reviews.latest_edge_reviews(rev)
    if not latest:
        return {}
    texts = edge_evidence_texts(con, tables.edge_evidence)
    states = {
        edge_id: reviews.review_state(latest.get(edge_id), reviews.evidence_hash(texts.get(edge_id, [])))
        for edge_id in tables.edges["edge_id"]
    }  # fmt: skip
    tables.edges["review_state"] = tables.edges["edge_id"].map(states)
    counts = tables.edges["review_state"].value_counts().to_dict()
    counts["missing"] = len(set(latest) - set(tables.edges["edge_id"]))
    if counts["missing"]:
        log.warning(
            "검수했지만 이번 판정에 없는 관계 %d개 (검수 기록은 그대로 둡니다)", counts["missing"]
        )
    return {k: int(v) for k, v in counts.items()}


def stage_judge_all(cfg: Config, model: str | None = None, tables=None, cands=None) -> Path:
    """graph.db의 모든 단위를 판정(캐시 우선)하고, 관계를 합쳐 graph.db를 다시 쓴다."""
    from .judge.cache import connect, judge_cached
    from .judge.jev import PRICE_PER_MILLION_INPUT, JudgeError

    con = open_graph(cfg)
    units = all_units(con)
    requests = _unit_requests(cfg, con, units)
    con.close()
    judge = make_judge(cfg, model)
    try:
        run = judge_cached(judge, requests, connect(judgements_path(cfg)))
    except JudgeError as exc:
        raise SystemExit(str(exc)) from exc
    log.info(
        "전체 판정 %d건 (%s): 캐시 %d, 새로 %d, 실패 %d · 새 입력 %d토큰 (약 $%.3f)",
        len(requests), judge.model_id, run.n_cached, run.n_new, run.n_failed, run.input_tokens,
        run.input_tokens / 1e6 * PRICE_PER_MILLION_INPUT,
    )  # fmt: skip
    return stage_export(cfg, tables, cands)


# ---------------------------------------------------------------- 판정과 채점 (1단계)


def make_judge(cfg: Config, model: str | None = None):
    from .judge.jev import JevJudge

    j = relations_config(cfg).judge
    return JevJudge(model or j.model, j.concurrency)


def judge_settings(cfg: Config, model: str | None = None, context: str | None = None) -> dict:
    """판정 결과를 바꾸는 설정 전부. 확인 표본은 뽑을 때 이것을 고정해 둔다."""
    from .judge.questions import QUESTION_VERSION, YES_NO_QUESTIONS

    j = relations_config(cfg).judge
    context = context or j.context
    return {
        "model": model or j.model,
        "question_version": QUESTION_VERSION,
        "context": context,
        "context_chars": j.context_chars if context == "nearby" else None,
        "thresholds": {q: list(j.threshold(q)) for q in YES_NO_QUESTIONS},
    }


def sample_requests(cfg: Config, sample: str, limit: int | None = None, context: str | None = None):
    """표본의 검수 단위 → 판정 요청 (검수 순서대로). context를 비우면 설정값."""
    from . import reviews
    from .judge.inputs import requests_for_units

    rev = reviews.connect(reviews_path(cfg))
    units = [
        r[0]
        for r in rev.execute(
            "SELECT unit_id FROM sample_units WHERE sample_id = ? ORDER BY ord", (sample,)
        )
    ]
    if not units:
        raise SystemExit(f"표본이 없습니다: {sample} (`tenksim sample -c ...`로 목록 확인)")
    units = units[:limit] if limit else units
    j = relations_config(cfg).judge
    requests = requests_for_units(
        open_graph(cfg), units, context=context or j.context, context_chars=j.context_chars
    )
    if len(requests) < len(units):
        log.warning(
            "graph.db를 다시 만들면서 사라진 근거 구간 %d개는 판정하지 않습니다",
            len(units) - len(requests),
        )
    return requests


def stage_judge(cfg: Config, sample: str, model: str | None = None, limit: int | None = None,
                judge=None, context: str | None = None):  # fmt: skip
    """표본 검수 단위를 판정한다. 이미 판정한 입력은 캐시(judgements.sqlite)에서 꺼낸다."""
    from .judge.cache import connect, judge_cached
    from .judge.jev import PRICE_PER_MILLION_INPUT, JudgeError

    context = context or relations_config(cfg).judge.context
    requests = sample_requests(cfg, sample, limit, context)
    judge = judge or make_judge(cfg, model)
    try:
        run = judge_cached(judge, requests, connect(judgements_path(cfg)))
    except JudgeError as exc:
        raise SystemExit(str(exc)) from exc
    cost = run.input_tokens / 1e6 * PRICE_PER_MILLION_INPUT
    log.info(
        "판정 %d건 (%s, 입력 %s): 캐시 %d, 새로 %d, 실패 %d · 새 입력 %d토큰 (약 $%.4f)",
        len(requests), judge.model_id, context, run.n_cached, run.n_new, run.n_failed, run.input_tokens,
        cost,
    )  # fmt: skip
    if run.n_failed:
        log.warning("실패한 %d건은 다시 실행하면 그것만 다시 묻습니다", run.n_failed)
    return requests, run


def stage_eval(
    cfg: Config,
    sample: str,
    model: str | None = None,
    judge=None,
    context: str | None = None,
    force: bool = False,
):
    """표본을 판정(캐시 우선)하고 검수 라벨로 채점한다. 결과는 relations/eval/ 아래에 저장.

    확인 표본은 뽑을 때 고정한 판정 설정과 지금 설정이 다르면 멈춘다. force로 채점할 수는 있지만,
    그 결과는 합격 판단에 쓰지 않는다고 보고서에 적는다."""
    from . import reviews
    from .evaluate import disagreements, evaluate, format_report, load_labels, unit_table
    from .judge.questions import QUESTION_VERSION, YES_NO_QUESTIONS

    rel = relations_config(cfg)
    context = context or rel.judge.context
    judge = judge or make_judge(cfg, model)
    rev = reviews.connect(reviews_path(cfg))
    purpose = rev.execute("SELECT purpose FROM samples WHERE sample_id = ?", (sample,)).fetchone()
    frozen = reviews.frozen_settings(rev, sample)
    current = judge_settings(cfg, judge.model_id, context)
    changed = reviews.settings_changes(frozen, current) if frozen else []
    if changed and not force:
        raise SystemExit(
            f"{sample}은 확인 표본이고, 뽑을 때 고정한 판정 설정과 지금 설정이 다릅니다: "
            f"{', '.join(changed)}. 설정을 되돌리거나 확인 표본을 새로 뽑으세요 "
            "(--force로 채점할 수는 있지만 합격 판단에는 쓰지 않습니다)"
        )
    if purpose and purpose[0] == "confirm":
        log.warning(
            "확인 표본입니다. 이 결과를 보고 질문·임계값을 고치면 확인 표본을 새로 뽑아야 합니다"
        )
    _, run = stage_judge(cfg, sample, judge=judge, context=context)
    labels = load_labels(rev, sample)
    judgements = {j.unit_id: j for j in run.judgements if j is not None}
    thresholds = {q: rel.judge.threshold(q) for q in YES_NO_QUESTIONS}
    table = unit_table(labels, judgements, rel.judge.accept, rel.judge.reject, thresholds)
    metrics = evaluate(table)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "sample": sample,
        "purpose": purpose[0] if purpose else None,
        "model": judge.model_id,
        "served_models": sorted({j.served_model for j in judgements.values() if j.served_model}),
        "question_version": QUESTION_VERSION,
        "context": context,
        "accept": rel.judge.accept,
        "reject": rel.judge.reject,
        "thresholds": thresholds,
        "labels": f"reviews.sqlite의 표본 {sample} 라벨 (단위마다 가장 최근 것)",
        "frozen": frozen is not None,
        "settings_changed": changed,
    }
    out = cfg.relations_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{sample}_{judge.model_id}_{QUESTION_VERSION}_{context}"
    report = format_report(metrics, meta)
    (out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / f"{stem}.md").write_text(report, encoding="utf-8")
    # 엑셀에서 바로 열리게 BOM을 붙인다
    disagreements(table).to_csv(out / f"{stem}_disagreements.csv", index=False,
                                encoding="utf-8-sig")  # fmt: skip
    return metrics, report, out / f"{stem}.md"

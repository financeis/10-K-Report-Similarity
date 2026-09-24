"""관계도 파이프라인 단계와 산출물 입출력. 산출물은 data/runs/{name}/relations/ 아래에 둔다."""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd

from ..config import Config, RelationsConfig
from ..pipeline import load_documents, load_universe, similarity_for
from .candidates import Candidates, build_candidates
from .mentions import MentionTables, find_mentions, text_hash
from .names import build_dictionary, load_aliases

log = logging.getLogger(__name__)

MENTION_FILES = ("mentions", "spans", "figures", "nodes")


def relations_config(cfg: Config) -> RelationsConfig:
    if cfg.relations is None:
        raise SystemExit(f"{cfg.name}: 설정 파일에 relations: 절이 없습니다")
    return cfg.relations


def stage_mentions(cfg: Config) -> MentionTables:
    """분석 가능한 10-K 섹션에서 회사 이름 언급을 찾는다. 몇 초면 끝나므로 매번 새로 만든다."""
    rel = relations_config(cfg)
    universe = load_universe(cfg)
    docs = load_documents(cfg)
    allowed = {"ok"} if cfg.text.exclude_suspect else {"ok", "suspect"}
    docs = docs[docs["status"].isin(allowed)]

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

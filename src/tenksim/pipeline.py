"""파이프라인 단계와 산출물 입출력.

universe → ingest(EDGAR) → documents(정제·품질 판정) → methods(기업 벡터) → evaluate → report

각 단계 결과는 data/runs/{name}/ 아래에 저장되고, 다음 단계는 디스크에서 읽는다.
그래서 단계별로 따로 다시 돌릴 수 있다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config, MethodConfig
from .embed import DenseEmbedder, EmbeddingCache, TfidfMethod, build_method
from .evaluate import agreement, label_frame, label_metrics
from .filings import ingest
from .returns import (
    correlation_matrices,
    label_peers,
    load_prices,
    peer_correlation,
    random_baseline,
)
from .similarity import cosine_matrix, pair_percentiles, pool_chunks, top_k
from .text import assess, chunk_text, clean_section
from .universe import build_universe

log = logging.getLogger(__name__)

CENTER_SUFFIX = "+center"


# ---------------------------------------------------------------- universe / ingest / documents


def stage_universe(cfg: Config) -> pd.DataFrame:
    universe = build_universe(cfg.universe)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(cfg.run_dir / "universe.parquet", index=False)
    return universe


def load_universe(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(_require(cfg.run_dir / "universe.parquet", "ingest"))


def stage_ingest(cfg: Config, universe: pd.DataFrame, *, refresh: bool = False) -> pd.DataFrame:
    records = ingest(
        universe["cik"].tolist(),
        cfg.filings.year,
        cfg.filings.sections,
        cfg.sections_dir,
        workers=cfg.filings.workers,
        refresh=refresh,
    )
    return stage_documents(cfg, universe, records)


def stage_documents(cfg: Config, universe: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """원문을 정제하고 분석에 쓸 수 있는지 판정한다."""
    rows = []
    for rec in records.to_dict("records"):
        cleaned = None
        if rec["status"] == "fetched":
            cleaned = clean_section(
                rec["text_raw"],
                rec["section"],
                drop_page_markers=cfg.text.drop_page_markers,
                drop_table_rows=cfg.text.drop_table_rows,
            )
        row = {k: v for k, v in rec.items() if k != "text_raw"}
        row["status"] = assess(rec["status"], cleaned, cfg.text.min_chars)
        row["first_line"] = cleaned.first_line if cleaned else None
        row["n_chars_raw"] = cleaned.n_chars_raw if cleaned else 0
        row["n_chars"] = len(cleaned.text) if cleaned else 0
        row["table_char_share"] = cleaned.table_char_share if cleaned else np.nan
        row["text"] = cleaned.text if cleaned else ""
        rows.append(row)
    docs = pd.DataFrame(rows).merge(universe[["cik", "ticker", "name"]], on="cik", how="left")
    docs.to_parquet(cfg.run_dir / "documents.parquet", index=False)
    summary = docs.groupby(["section", "status"]).size().unstack(fill_value=0)
    log.info("Document status:\n%s", summary.to_string())
    return docs


def load_documents(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(_require(cfg.run_dir / "documents.parquet", "ingest"))


def usable_ciks(cfg: Config, universe: pd.DataFrame, docs: pd.DataFrame) -> list[int]:
    """모든 method가 쓰는 섹션이 분석 가능한 상태인 회사 (universe 순서 유지)."""
    allowed = {"ok"} if cfg.text.exclude_suspect else {"ok", "suspect"}
    needed = {m.section for m in cfg.methods}
    ok = docs[docs["status"].isin(allowed) & docs["section"].isin(needed)]
    n_sections = ok.groupby("cik")["section"].nunique()
    good = set(n_sections[n_sections == len(needed)].index)
    return [c for c in universe["cik"] if c in good]


# ---------------------------------------------------------------- methods


@dataclass
class MethodResult:
    name: str
    kind: str
    companies: pd.DataFrame
    """cik, ticker, name. 행 순서 = vectors 행 순서."""
    vectors: np.ndarray | sparse.csr_matrix
    chunks: pd.DataFrame | None = None
    """dense 방법만: owner(회사 행 번호), cik, chunk_idx, text, n_tokens."""
    chunk_vectors: np.ndarray | None = None

    def similarity_variants(self, center_variants: bool) -> dict[str, np.ndarray]:
        sims = {self.name: cosine_matrix(self.vectors)}
        if center_variants and self.kind != "tfidf":
            sims[self.name + CENTER_SUFFIX] = cosine_matrix(self.vectors, center=True)
        return sims

    def positions(self, ciks: list[int]) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.companies["cik"])}
        return np.array([lookup[c] for c in ciks], dtype=int)


def stage_methods(
    cfg: Config, universe: pd.DataFrame, docs: pd.DataFrame, only: list[str] | None = None
) -> dict[str, MethodResult]:
    ciks = usable_ciks(cfg, universe, docs)
    if len(ciks) < 3:
        raise RuntimeError(
            f"분석 가능한 회사가 {len(ciks)}개뿐입니다. reports의 품질 표를 확인하세요."
        )
    companies = universe.set_index("cik").loc[ciks, ["ticker", "name"]].reset_index()
    texts = docs.set_index(["cik", "section"])["text"]
    results = {}
    with EmbeddingCache(cfg.cache_path) as cache:
        for mcfg in cfg.methods:
            if only and mcfg.name not in only:
                continue
            log.info(
                "Method %s (%s, section=%s) on %d companies",
                mcfg.name,
                mcfg.kind,
                mcfg.section,
                len(ciks),
            )
            doc_texts = [texts[(c, mcfg.section)] for c in ciks]
            result = _run_method(mcfg, companies, doc_texts, cache)
            save_method(cfg, mcfg, result)
            results[mcfg.name] = result
    return results


def _run_method(
    mcfg: MethodConfig, companies: pd.DataFrame, doc_texts: list[str], cache: EmbeddingCache
) -> MethodResult:
    method = build_method(mcfg)
    if isinstance(method, TfidfMethod):
        return MethodResult(mcfg.name, mcfg.kind, companies, method.fit_transform(doc_texts))

    assert isinstance(method, DenseEmbedder)
    rows = []
    for owner, (cik, text) in enumerate(zip(companies["cik"], doc_texts, strict=True)):
        pieces = chunk_text(text, method.max_tokens, method.count_tokens)
        if mcfg.doc_tokens:
            pieces = _leading_chunks(pieces, mcfg.doc_tokens)
        for i, (chunk, n) in enumerate(pieces):
            rows.append((owner, cik, i, chunk, n))
    chunks = pd.DataFrame(rows, columns=["owner", "cik", "chunk_idx", "text", "n_tokens"])
    log.info("%s: %d chunks, %d tokens", mcfg.name, len(chunks), chunks["n_tokens"].sum())
    chunk_vectors = method.embed(chunks["text"].tolist(), cache)
    vectors = pool_chunks(
        chunk_vectors,
        chunks["owner"].to_numpy(),
        chunks["n_tokens"].to_numpy(dtype=np.float64),
        len(companies),
    )
    return MethodResult(mcfg.name, mcfg.kind, companies, vectors, chunks, chunk_vectors)


def _leading_chunks(chunks: list[tuple[str, int]], budget: int) -> list[tuple[str, int]]:
    """앞에서부터 누적 토큰이 budget을 넘지 않는 청크까지 (청크 단위라 근사치, 최소 1개)."""
    kept, total = [], 0
    for chunk, n in chunks:
        if kept and total + n > budget:
            break
        kept.append((chunk, n))
        total += n
    return kept


def save_method(cfg: Config, mcfg: MethodConfig, res: MethodResult) -> None:
    d = cfg.method_dir(res.name)
    d.mkdir(parents=True, exist_ok=True)
    res.companies.to_parquet(d / "companies.parquet", index=False)
    if sparse.issparse(res.vectors):
        sparse.save_npz(d / "vectors.npz", res.vectors)
    else:
        np.save(d / "vectors.npy", res.vectors)
    meta = {
        "config": mcfg.model_dump(mode="json"),
        "n_companies": len(res.companies),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if res.chunks is not None:
        res.chunks.to_parquet(d / "chunks.parquet", index=False)
        np.save(d / "chunk_vectors.npy", res.chunk_vectors)
        meta["n_chunks"] = len(res.chunks)
        meta["n_tokens"] = int(res.chunks["n_tokens"].sum())
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_method(cfg: Config, name: str) -> MethodResult:
    mcfg = cfg.method(name)
    d = cfg.method_dir(name)
    companies = pd.read_parquet(_require(d / "companies.parquet", "embed"))
    if (d / "vectors.npz").exists():
        vectors = sparse.load_npz(d / "vectors.npz").tocsr()
    else:
        vectors = np.load(d / "vectors.npy")
    chunks = chunk_vectors = None
    if (d / "chunks.parquet").exists():
        chunks = pd.read_parquet(d / "chunks.parquet")
        chunk_vectors = np.load(d / "chunk_vectors.npy")
    return MethodResult(name, mcfg.kind, companies, vectors, chunks, chunk_vectors)


def load_methods(cfg: Config) -> dict[str, MethodResult]:
    return {m.name: load_method(cfg, m.name) for m in cfg.methods}


# ---------------------------------------------------------------- neighbors / evaluate


def neighbors_table(res: MethodResult, sim: np.ndarray, k: int) -> pd.DataFrame:
    idx, scores = top_k(sim, k)
    pct = pair_percentiles(sim)
    comp = res.companies
    rows = []
    for i in range(len(comp)):
        for rank, (j, score) in enumerate(zip(idx[i], scores[i], strict=True), start=1):
            rows.append(
                {
                    "cik": comp.at[i, "cik"],
                    "ticker": comp.at[i, "ticker"],
                    "rank": rank,
                    "neighbor_cik": comp.at[j, "cik"],
                    "neighbor_ticker": comp.at[j, "ticker"],
                    "neighbor_name": comp.at[j, "name"],
                    "cosine": float(score),
                    "percentile": float(pct[i, j]),
                }
            )
    return pd.DataFrame(rows)


def stage_evaluate(
    cfg: Config, universe: pd.DataFrame, docs: pd.DataFrame, results: dict[str, MethodResult]
) -> dict:
    # 모든 방법을 같은 회사 집합에서 비교해야 공정하다
    common_set = set.intersection(*(set(r.companies["cik"]) for r in results.values()))
    common = [c for c in universe["cik"] if c in common_set]
    labels = label_frame(universe, docs).reindex(common)[list(cfg.evaluation.labels)]

    sims: dict[str, np.ndarray] = {}
    for res in results.values():
        pos = res.positions(common)
        for variant, sim in res.similarity_variants(cfg.center_variants).items():
            sims[variant] = sim[np.ix_(pos, pos)]
            suffix = "_center" if variant.endswith(CENTER_SUFFIX) else ""
            neighbors_table(res, sim, cfg.top_k).to_parquet(
                cfg.method_dir(res.name) / f"neighbors{suffix}.parquet", index=False
            )

    ks = cfg.evaluation.k
    metrics: dict = {
        "name": cfg.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_companies": len(common),
        "labels": {v: label_metrics(s, labels, ks) for v, s in sims.items()},
        "agreement": [],
        "returns": None,
    }
    names = list(sims)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            metrics["agreement"].append(
                {"a": names[a], "b": names[b], **agreement(sims[names[a]], sims[names[b]], max(ks))}
            )
    if cfg.evaluation.returns is not None:
        metrics["returns"] = _returns_metrics(cfg, universe, common, sims, labels)

    metrics = _nan_to_none(metrics)
    (cfg.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def load_metrics(cfg: Config) -> dict:
    path = _require(cfg.run_dir / "metrics.json", "evaluate")
    return json.loads(path.read_text(encoding="utf-8"))


def _returns_metrics(
    cfg: Config,
    universe: pd.DataFrame,
    common: list[int],
    sims: dict[str, np.ndarray],
    labels: pd.DataFrame,
) -> dict:
    rc = cfg.evaluation.returns
    tickers = universe.set_index("cik").loc[common, "ticker"].tolist()
    prices = load_prices(
        tickers, rc.market, rc.start, rc.end, cfg.run_dir / f"prices_{rc.start}_{rc.end}.parquet"
    )
    raw, resid, enough = correlation_matrices(prices, tickers, rc.market, rc.min_obs)
    idx = np.flatnonzero(enough)
    out: dict = {
        "n": int(len(idx)),
        "start": str(rc.start),
        "end": str(rc.end),
        "methods": {},
        "baselines": {},
    }
    for kind, corr in (("resid", resid), ("raw", raw)):
        c = corr[np.ix_(idx, idx)]
        for variant, sim in sims.items():
            s = sim[np.ix_(idx, idx)]
            for k in cfg.evaluation.k:
                nb, _ = top_k(s, k)
                out["methods"].setdefault(variant, {})[f"{kind}@{k}"] = peer_correlation(
                    c, list(nb)
                )
        for label in labels.columns:
            peers = label_peers(labels[label].to_numpy(dtype=object)[idx])
            out["baselines"].setdefault(label, {})[kind] = peer_correlation(c, peers)
        out["baselines"].setdefault("random", {})[kind] = random_baseline(c)
    return out


# ---------------------------------------------------------------- helpers


def _require(path, stage: str):
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. 먼저 'tenksim {stage}'를 실행하세요.")
    return path


def _nan_to_none(obj):
    """JSON 표준에는 NaN이 없으므로 null로 바꾼다."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    return obj

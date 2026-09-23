"""파이프라인 단계와 산출물 입출력.

universe → ingest(EDGAR) → documents(정제·품질 판정) → methods(기업 벡터) → evaluate → report

각 단계 결과는 data/runs/{name}/ 아래에 저장되고, 다음 단계는 디스크에서 읽는다.
그래서 단계별로 따로 다시 돌릴 수 있다.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config, MethodConfig
from .embed import DenseEmbedder, EmbeddingCache, TfidfMethod, build_method
from .evaluate import (
    agreement,
    bootstrap_means,
    label_frame,
    label_metrics,
    precision_per_firm,
)
from .filings import ingest
from .returns import (
    correlation_matrices,
    incremental_effect,
    label_peers,
    load_prices,
    peer_corr_per_firm,
    peer_correlation,
    random_baseline,
    random_per_firm,
    top_k_within,
)
from .similarity import cosine_matrix, ensemble_similarity, pair_percentiles, pool_chunks, top_k
from .text import CleanedSection, assess, chunk_text, clean_section, duplicate_share
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
    recs = records.to_dict("records")
    cleaned_by_cik: dict[int, dict[str, CleanedSection]] = defaultdict(dict)
    for rec in recs:
        if rec["status"] == "fetched":
            cleaned_by_cik[rec["cik"]][rec["section"]] = clean_section(
                rec["text_raw"],
                rec["section"],
                drop_page_markers=cfg.text.drop_page_markers,
                drop_table_rows=cfg.text.drop_table_rows,
            )
    rows = []
    for rec in recs:
        sections = cleaned_by_cik.get(rec["cik"], {})
        cleaned = sections.get(rec["section"])
        # 두 섹션이 같은 줄을 절반 넘게 공유하면 한쪽이 잘못 잘린 것이다.
        # 어느 쪽이 틀렸는지는 알 수 없으므로 둘 다 뺀다.
        duplicate_of = next(
            (
                name
                for name, other in sections.items()
                if cleaned is not None
                and name != rec["section"]
                and max(
                    duplicate_share(cleaned.text, other.text),
                    duplicate_share(other.text, cleaned.text),
                )
                > 0.5
            ),
            None,
        )
        row = {k: v for k, v in rec.items() if k != "text_raw"}
        row["status"], row["note"] = assess(
            rec["status"], cleaned, cfg.text.min_chars, duplicate_of
        )
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


def neighbors_table(companies: pd.DataFrame, sim: np.ndarray, k: int) -> pd.DataFrame:
    """회사마다 상위 k 이웃. score는 방법의 유사도(코사인, 앙상블이면 평균 백분위)."""
    idx, scores = top_k(sim, k)
    pct = pair_percentiles(sim)
    rows = []
    for i in range(len(companies)):
        for rank, (j, score) in enumerate(zip(idx[i], scores[i], strict=True), start=1):
            rows.append(
                {
                    "cik": companies.at[i, "cik"],
                    "ticker": companies.at[i, "ticker"],
                    "rank": rank,
                    "neighbor_cik": companies.at[j, "cik"],
                    "neighbor_ticker": companies.at[j, "ticker"],
                    "neighbor_name": companies.at[j, "name"],
                    "score": float(score),
                    "percentile": float(pct[i, j]),
                }
            )
    return pd.DataFrame(rows)


def neighbors_path(cfg: Config, variant: str) -> Path:
    if cfg.ensemble(variant) is not None:
        return cfg.run_dir / "ensembles" / variant / "neighbors.parquet"
    center = variant.endswith(CENTER_SUFFIX)
    name = variant.removesuffix(CENTER_SUFFIX)
    return cfg.method_dir(name) / ("neighbors_center.parquet" if center else "neighbors.parquet")


def similarity_for(cfg: Config, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """방법·변형·앙상블 이름 하나의 (회사 목록, 유사도 행렬)."""
    ens = cfg.ensemble(name)
    if ens is None:
        res = load_method(cfg, name.removesuffix(CENTER_SUFFIX))
        sims = res.similarity_variants(cfg.center_variants)
        if name not in sims:
            raise KeyError(f"{name}: 없는 변형입니다 (가능: {list(sims)})")
        return res.companies, sims[name]
    bases = list(dict.fromkeys(m.removesuffix(CENTER_SUFFIX) for m in ens.members))
    results = {b: load_method(cfg, b) for b in bases}
    common_set = set.intersection(*(set(r.companies["cik"]) for r in results.values()))
    first = results[bases[0]].companies
    companies = first[first["cik"].isin(common_set)].reset_index(drop=True)
    members = []
    for m in ens.members:
        res = results[m.removesuffix(CENTER_SUFFIX)]
        pos = res.positions(list(companies["cik"]))
        members.append(res.similarity_variants(cfg.center_variants)[m][np.ix_(pos, pos)])
    return companies, ensemble_similarity(members, ens.weights)


def stage_evaluate(
    cfg: Config, universe: pd.DataFrame, docs: pd.DataFrame, results: dict[str, MethodResult]
) -> dict:
    # 모든 방법을 같은 회사 집합에서 비교해야 공정하다
    common_set = set.intersection(*(set(r.companies["cik"]) for r in results.values()))
    common = [c for c in universe["cik"] if c in common_set]
    companies = universe.set_index("cik").loc[common, ["ticker", "name"]].reset_index()
    all_labels = label_frame(universe, docs).reindex(common)
    labels = all_labels[list(cfg.evaluation.labels)]

    sims: dict[str, np.ndarray] = {}
    for res in results.values():
        pos = res.positions(common)
        for variant, sim in res.similarity_variants(cfg.center_variants).items():
            sims[variant] = sim[np.ix_(pos, pos)]
            neighbors_table(res.companies, sim, cfg.top_k).to_parquet(
                neighbors_path(cfg, variant), index=False
            )
    for ens in cfg.ensembles:
        sims[ens.name] = ensemble_similarity([sims[m] for m in ens.members], ens.weights)
        path = neighbors_path(cfg, ens.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        neighbors_table(companies, sims[ens.name], cfg.top_k).to_parquet(path, index=False)

    ev = cfg.evaluation
    ks = ev.k
    metrics: dict = {
        "name": cfg.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_companies": len(common),
        "baseline": cfg.baseline_name,
        "main_k": ev.main_k,
        "labels": {v: label_metrics(s, labels, ks) for v, s in sims.items()},
        "labels_ci": _label_ci(cfg, sims, labels),
        "agreement": [],
        "returns": None,
        "beyond": None,
    }
    names = list(sims)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            metrics["agreement"].append(
                {"a": names[a], "b": names[b], **agreement(sims[names[a]], sims[names[b]], max(ks))}
            )
    if ev.returns is not None:
        metrics["returns"], metrics["beyond"] = _returns_metrics(
            cfg, universe, common, companies, sims, all_labels
        )

    metrics = _nan_to_none(metrics)
    (cfg.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def load_metrics(cfg: Config) -> dict:
    path = _require(cfg.run_dir / "metrics.json", "evaluate")
    return json.loads(path.read_text(encoding="utf-8"))


def _label_ci(cfg: Config, sims: dict[str, np.ndarray], labels: pd.DataFrame) -> dict:
    """주 레이블(서브산업이 있으면 서브산업)의 P@k 신뢰구간과 기준 방법 대비 차이."""
    name = "gics_sub_industry" if "gics_sub_industry" in labels.columns else labels.columns[0]
    values = labels[name].to_numpy(dtype=object)
    out: dict = {"label": name}
    for k in cfg.evaluation.k:
        per_firm = {v: precision_per_firm(s, values, k) for v, s in sims.items()}
        out[f"p@{k}"] = bootstrap_means(
            per_firm, n_boot=cfg.evaluation.n_boot, reference=cfg.baseline_name
        )
    return out


def _returns_metrics(
    cfg: Config,
    universe: pd.DataFrame,
    common: list[int],
    companies: pd.DataFrame,
    sims: dict[str, np.ndarray],
    all_labels: pd.DataFrame,
) -> tuple[dict, dict | None]:
    ev = cfg.evaluation
    rc = ev.returns
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
        "ci": {},
    }
    for kind, corr in (("resid", resid), ("raw", raw)):
        c = corr[np.ix_(idx, idx)]
        per_k: dict[int, dict[str, np.ndarray]] = {k: {} for k in ev.k}
        for variant, sim in sims.items():
            s = sim[np.ix_(idx, idx)]
            for k in ev.k:
                nb, _ = top_k(s, k)
                values = peer_corr_per_firm(c, list(nb))
                per_k[k][variant] = values
                out["methods"].setdefault(variant, {})[f"{kind}@{k}"] = float(np.nanmean(values))
        baselines = {}
        for label in ev.labels:
            peers = label_peers(all_labels[label].to_numpy(dtype=object)[idx])
            baselines[f"baseline:{label}"] = peer_corr_per_firm(c, peers)
            out["baselines"].setdefault(label, {})[kind] = peer_correlation(c, peers)
        baselines["baseline:random"] = random_per_firm(c)
        out["baselines"].setdefault("random", {})[kind] = random_baseline(c)
        for k in ev.k:
            out["ci"][f"{kind}@{k}"] = bootstrap_means(
                {**per_k[k], **baselines}, n_boot=ev.n_boot, reference=cfg.baseline_name
            )

    key = f"resid@{ev.main_k}"
    out["best"] = max(out["methods"], key=lambda v: out["methods"][v][key])
    sub_idx = companies.iloc[idx].reset_index(drop=True)
    beyond = _beyond_metrics(
        cfg, sims, idx, resid[np.ix_(idx, idx)], all_labels.iloc[idx], sub_idx, out["best"]
    )
    return out, beyond


def _beyond_metrics(
    cfg: Config,
    sims: dict[str, np.ndarray],
    idx: np.ndarray,
    corr: np.ndarray,
    labels: pd.DataFrame,
    companies: pd.DataFrame,
    best: str,
) -> dict | None:
    """GICS가 같은 회사로 묶지 않는 연결을 텍스트가 찾는지.

    1) 서브산업(또는 섹터) 밖에서만 이웃을 골라도 주가가 같이 움직이는가.
       분류 밖에서 무작위로 고른 회사, 그리고 '같은 섹터·다른 서브산업' 회사와 비교한다.
    2) 기업쌍 회귀로 GICS를 통제한 뒤에도 유사도가 동조성을 설명하는가.
    """
    if "gics_sub_industry" not in labels or "gics_sector" not in labels:
        return None
    sub = labels["gics_sub_industry"].to_numpy(dtype=object)
    sec = labels["gics_sector"].to_numpy(dtype=object)
    valid = np.array([isinstance(s, str) for s in sub]) & np.array(
        [isinstance(s, str) for s in sec]
    )
    if valid.sum() < 4:
        return None
    pair_ok = valid[:, None] & valid[None, :]
    same_sub = (sub[:, None] == sub[None, :]) & pair_ok
    same_sec = (sec[:, None] == sec[None, :]) & pair_ok
    outside_sub, outside_sec = pair_ok & ~same_sub, pair_ok & ~same_sec
    ev = cfg.evaluation
    k = ev.main_k

    random_outside_sub = random_per_firm(corr, outside_sub)
    random_sector_peer = random_per_firm(corr, same_sec & ~same_sub)
    random_outside_sec = random_per_firm(corr, outside_sec)
    texts_sub, lift_sub, lift_sector, texts_sec, lift_sec, regression = {}, {}, {}, {}, {}, {}
    for variant, full_sim in sims.items():
        sim = full_sim[np.ix_(idx, idx)]
        texts_sub[variant] = peer_corr_per_firm(corr, top_k_within(sim, outside_sub, k))
        texts_sec[variant] = peer_corr_per_firm(corr, top_k_within(sim, outside_sec, k))
        lift_sub[variant] = texts_sub[variant] - random_outside_sub
        lift_sector[variant] = texts_sub[variant] - random_sector_peer
        lift_sec[variant] = texts_sec[variant] - random_outside_sec
        regression[variant] = incremental_effect(
            sim, corr, same_sub, same_sec, n_boot=max(50, ev.n_boot // 5)
        )
    boot = partial(bootstrap_means, n_boot=ev.n_boot)
    out = {
        "k": k,
        "outside_sub": boot(texts_sub),
        "lift_outside_sub": boot(lift_sub),
        "lift_vs_sector_peer": boot(lift_sector),
        "outside_sector": boot(texts_sec),
        "lift_outside_sector": boot(lift_sec),
        "baselines": {
            "random_outside_sub": _mean(random_outside_sub),
            "random_sector_peer": _mean(random_sector_peer),
            "random_outside_sector": _mean(random_outside_sec),
        },
        "regression": regression,
        "example_variant": best,
        "examples": {},
    }
    # 가장 좋은 방법으로 찾은 '서브산업 밖' 이웃 예시
    tick = companies["ticker"].to_numpy()
    neighbors = top_k_within(sims[best][np.ix_(idx, idx)], outside_sub, 3)
    for t in _example_tickers(tick):
        i = int(np.flatnonzero(tick == t)[0])
        out["examples"][t] = {
            "sub_industry": sub[i],
            "neighbors": [
                {"ticker": tick[j], "sub_industry": sub[j], "resid_corr": float(corr[i, j])}
                for j in neighbors[i]
            ],
        }
    return out


EXAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "PFE", "AMZN", "KO", "NEE", "PLD"]


def _example_tickers(tickers: np.ndarray) -> list[str]:
    present = [t for t in EXAMPLE_TICKERS if t in set(tickers)]
    return present if len(present) >= 3 else list(tickers[:8])


# ---------------------------------------------------------------- helpers


def _require(path, stage: str):
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. 먼저 'tenksim {stage}'를 실행하세요.")
    return path


def _mean(values: np.ndarray) -> float:
    """NaN을 뺀 평균. 해당하는 회사가 하나도 없으면 NaN (경고 없이)."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


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

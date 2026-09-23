"""metrics.json과 품질 정보를 사람이 읽는 마크다운 리포트로 만든다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config

LABEL_TITLES = {
    "gics_sector": "GICS 섹터",
    "gics_sub_industry": "GICS 서브산업",
    "sic2": "SIC 2자리",
    "sic3": "SIC 3자리",
    "sic4": "SIC 4자리",
}
STATUSES = ["ok", "suspect", "too_short", "missing", "no_filing", "error"]
EXAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "PFE", "AMZN", "KO", "NEE", "PLD"]


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _str(v) -> str:
    return v if isinstance(v, str) else ""


def note_label(note: str) -> str:
    """text.assess가 남긴 사유 코드(';'로 여러 개)를 리포트용 문구로."""
    parts = []
    for code in filter(None, note.split(";")):
        kind, _, value = code.partition(":")
        labels = {
            "note": "재무제표 주석으로 시작",
            "wrong_item": f"다른 항목(Item {value.upper()})이 추출됨",
            "financial_statement": "재무제표로 시작",
            "toc": "목차·상호참조 색인 형식",
            "duplicate": f"{value} 섹션과 내용 중복",
            "chars": f"정제 후 {int(value):,}자" if value.isdigit() else code,
            "no_heading": "제목 줄 없음",
            "cut": f"Item {value.upper()} 제목에서 자름",
        }
        parts.append(labels.get(kind, code))
    return ", ".join(parts)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def best_variant(metrics: dict, cfg: Config) -> str:
    label = (
        "gics_sub_industry"
        if "gics_sub_industry" in cfg.evaluation.labels
        else cfg.evaluation.labels[0]
    )
    key = f"p@{max(cfg.evaluation.k)}"
    # 회사 수가 적으면 P@k가 자주 같아지므로 AUC로 순위를 가른다
    scores = {v: (m[label][key] or -1, m[label]["auc"] or -1) for v, m in metrics["labels"].items()}
    return max(scores, key=scores.get)


def build_report(cfg: Config, metrics: dict, docs: pd.DataFrame, universe: pd.DataFrame) -> str:
    ks = cfg.evaluation.k
    out = [
        f"# {cfg.name} 결과 리포트",
        "",
        f"- 생성: {metrics['created']}",
        f"- 10-K 제출 연도: {cfg.filings.year} · 수집 섹션: {', '.join(cfg.filings.sections)}",
        f"- 분석 대상: {metrics['n_companies']}개 기업 (universe {len(universe)}개 중)",
        "- 방법: " + ", ".join(f"`{m.name}` ({m.kind}, {m.section})" for m in cfg.methods),
    ]
    if cfg.center_variants:
        out.append("- `+center`: 전체 평균 벡터를 뺀 뒤 코사인 유사도를 잰 버전")
    out += [
        "",
        "지표 정의와 해석은 [docs/methodology.md](../docs/methodology.md)를 참고하세요.",
        "",
    ]

    # 1. 데이터 품질
    out += ["## 1. 데이터 품질", ""]
    counts = docs.groupby(["section", "status"]).size().unstack(fill_value=0)
    rows = [[sec] + [int(counts.loc[sec].get(s, 0)) for s in STATUSES] for sec in counts.index]
    out += [_table(["섹션", *STATUSES], rows), ""]
    bad = docs[docs["status"] != "ok"].sort_values(["status", "section", "ticker"])
    if len(bad):
        out += ["분석에서 제외된 문서:", ""]
        rows = [
            [
                r.ticker,
                _str(r.company) or _str(r.name),
                r.section,
                r.status,
                note_label(_str(getattr(r, "note", None))),
                (_str(r.error) or _str(r.first_line))[:80].replace("|", "/"),
            ]
            for r in bad.itertuples()
        ]
        out += [_table(["티커", "회사", "섹션", "상태", "사유", "첫 줄 / 오류"], rows), ""]
    if "note" in docs.columns:
        ok_notes = docs.loc[docs["status"] == "ok", "note"].fillna("")
        n_no_heading = int(ok_notes.str.contains("no_heading").sum())
        n_cut = int(ok_notes.str.contains("cut:").sum())
        if n_no_heading or n_cut:
            out += [
                "분석에 포함했지만 확인해 볼 만한 문서 (`documents.parquet`의 `note` 컬럼):",
                "",
                f"- 섹션 제목 줄을 못 찾았지만 다른 이상 신호가 없는 문서: {n_no_heading}건",
                f"- 본문 중간의 다음 항목 제목(예: Item 1A)에서 잘라낸 문서: {n_cut}건",
                "",
            ]

    # 2. 산업분류 재현
    out += [
        "## 2. 산업분류 재현",
        "",
        "텍스트로 찾은 이웃이 같은 산업에 속하는지 봅니다. "
        "AUC는 0.5가 무작위, P@k는 상위 k개 이웃 중 같은 산업 비율입니다.",
        "",
    ]
    headers = ["방법", "AUC", *[f"P@{k}" for k in ks]]
    for label in cfg.evaluation.labels:
        entries = {v: m[label] for v, m in metrics["labels"].items()}
        any_entry = next(iter(entries.values()))
        rows = [[v, e["auc"], *[e[f"p@{k}"] for k in ks]] for v, e in entries.items()]
        rows.append(["(무작위 기대값)", 0.5, *[any_entry[f"random@{k}"] for k in ks]])
        out += [
            f"### {LABEL_TITLES.get(label, label)} (n={any_entry['n']})",
            "",
            _table(headers, rows),
            "",
        ]

    # 3. 주가 동조성
    rm = metrics.get("returns")
    if rm:
        out += [
            f"## 3. 주가 동조성 ({rm['start']} ~ {rm['end']}, n={rm['n']})",
            "",
            "텍스트 이웃과의 일별 수익률 상관 평균입니다. 잔차는 시장(베타) 움직임을 뺀 값입니다. "
            "산업분류 기준선은 같은 분류의 회사 전체를 peer로 둔 값입니다.",
            "",
        ]
        headers = ["방법", *[f"잔차@{k}" for k in ks], *[f"원수익률@{k}" for k in ks]]
        rows = [
            [v, *[m[f"resid@{k}"] for k in ks], *[m[f"raw@{k}"] for k in ks]]
            for v, m in rm["methods"].items()
        ]
        for label, b in rm["baselines"].items():
            title = "무작위" if label == "random" else f"같은 {LABEL_TITLES.get(label, label)} 전체"
            rows.append([f"({title})", *[b["resid"]] * len(ks), *[b["raw"]] * len(ks)])
        out += [_table(headers, rows), ""]

    # 4. 방법 간 일치도
    if metrics["agreement"]:
        k = max(ks)
        out += [
            "## 4. 방법 간 일치도",
            "",
            f"Spearman은 전체 기업쌍 점수의 순위 상관, Jaccard@{k}는 상위 {k}개 이웃이 겹치는 정도입니다.",
            "",
        ]
        rows = [[a["a"], a["b"], a["spearman"], a[f"jaccard@{k}"]] for a in metrics["agreement"]]
        out += [_table(["A", "B", "Spearman", f"Jaccard@{k}"], rows), ""]

    # 5. 예시
    variant = best_variant(metrics, cfg)
    neighbors = _load_neighbors(cfg, variant)
    if neighbors is not None:
        present = [t for t in EXAMPLE_TICKERS if t in set(neighbors["ticker"])]
        if len(present) < 3:
            present = list(dict.fromkeys(neighbors["ticker"]))[:8]
        out += [
            f"## 5. 예시: 유사 기업 상위 5개 (`{variant}`)",
            "",
            "괄호 안은 전체 기업쌍 중 백분위(0~100)입니다.",
            "",
        ]
        rows = []
        for t in present:
            top = neighbors[neighbors["ticker"] == t].nsmallest(5, "rank")
            rows.append(
                [t, *[f"{r.neighbor_ticker} ({r.percentile:.0f})" for r in top.itertuples()]]
            )
        out += [_table(["회사", "1", "2", "3", "4", "5"], rows), ""]
    return "\n".join(out)


def _load_neighbors(cfg: Config, variant: str) -> pd.DataFrame | None:
    center = variant.endswith("+center")
    name = variant.removesuffix("+center")
    path = cfg.method_dir(name) / ("neighbors_center.parquet" if center else "neighbors.parquet")
    return pd.read_parquet(path) if path.exists() else None


def write_report(cfg: Config, metrics: dict, docs: pd.DataFrame, universe: pd.DataFrame) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(build_report(cfg, metrics, docs, universe) + "\n", encoding="utf-8")
    return cfg.report_path

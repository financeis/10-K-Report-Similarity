"""metrics.json과 품질 정보를 사람이 읽는 마크다운 리포트로 만든다.

순서는 중요도 순이다. 주가 동조성(경제적으로 의미 있는 연결을 찾는가)과
'GICS가 묶지 않는 연결'을 앞에 두고, 산업분류 재현은 최소 조건 확인용으로 뒤에 둔다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .pipeline import neighbors_path

LABEL_TITLES = {
    "gics_sector": "GICS 섹터",
    "gics_sub_industry": "GICS 서브산업",
    "sic2": "SIC 2자리",
    "sic3": "SIC 3자리",
    "sic4": "SIC 4자리",
}
STATUSES = ["ok", "suspect", "too_short", "missing", "no_filing", "error"]


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _str(v) -> str:
    return v if isinstance(v, str) else ""


def _ci(entry: dict | None) -> str:
    """'0.462 [0.431, 0.492]'"""
    if not entry or entry.get("mean") is None:
        return "–"
    return f"{entry['mean']:.3f} [{entry['lo']:.3f}, {entry['hi']:.3f}]"


def _diff(entry: dict | None, prefix: str = "diff") -> str:
    """짝지은 차이와 구간. 구간이 0을 포함하지 않으면 굵게 표시한다."""
    if not entry or entry.get(prefix) is None:
        return "–"
    d, lo, hi = entry[prefix], entry[f"{prefix}_lo"], entry[f"{prefix}_hi"]
    text = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"**{text}**" if lo > 0 or hi < 0 else text


def _lift(entry: dict | None) -> str:
    """차이 자체를 부트스트랩한 항목(mean/lo/hi)을 _diff 형식으로."""
    if not entry or entry.get("mean") is None:
        return "–"
    return _diff({"diff": entry["mean"], "diff_lo": entry["lo"], "diff_hi": entry["hi"]})


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
    """주가 동조성이 있으면 그 기준으로, 없으면 주 레이블의 P@k(동률이면 AUC)로 고른다."""
    if metrics.get("returns"):
        return metrics["returns"]["best"]
    label = metrics["labels_ci"]["label"]
    key = f"p@{cfg.evaluation.main_k}"
    scores = {v: (m[label][key] or -1, m[label]["auc"] or -1) for v, m in metrics["labels"].items()}
    return max(scores, key=scores.get)


def build_report(cfg: Config, metrics: dict, docs: pd.DataFrame, universe: pd.DataFrame) -> str:
    ks = cfg.evaluation.k
    k0 = cfg.evaluation.main_k
    base = metrics["baseline"]
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
    for ens in cfg.ensembles:
        out.append(f"- `{ens.name}`: {' + '.join(ens.members)}의 기업쌍 백분위 평균 (앙상블)")
    out += [
        f"- 괄호 안 구간은 기업 단위 부트스트랩 95% 신뢰구간({cfg.evaluation.n_boot}회)입니다. "
        f"'{base} 대비'는 같은 회사끼리 짝지은 차이이며, 구간이 0을 포함하지 않으면 **굵게** 표시합니다.",
        "",
        "지표 정의와 해석은 [docs/methodology.md](../docs/methodology.md)를 참고하세요.",
        "",
    ]
    out += _quality_section(docs)
    if metrics.get("returns"):
        out += _returns_section(metrics, ks, k0, base)
    if metrics.get("beyond"):
        out += _beyond_section(metrics["beyond"])
    out += _labels_section(cfg, metrics, ks, k0, base)
    out += _agreement_section(metrics, max(ks))
    out += _examples_section(cfg, metrics)
    return "\n".join(out)


def _quality_section(docs: pd.DataFrame) -> list[str]:
    out = ["## 1. 데이터 품질", ""]
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
    return out


def _returns_section(metrics: dict, ks: list[int], k0: int, base: str) -> list[str]:
    rm = metrics["returns"]
    ci = rm["ci"][f"resid@{k0}"]
    out = [
        f"## 2. 주가 동조성 ({rm['start']} ~ {rm['end']}, n={rm['n']})",
        "",
        "텍스트로 찾은 이웃과의 일별 수익률 상관 평균입니다. 이 프로젝트의 핵심 지표입니다. "
        "잔차는 시장 전체 움직임(베타)을 뺀 값이라 무작위 쌍이 0 근처가 됩니다. "
        "산업분류 기준선은 같은 분류의 회사 **전체**를 peer로 둔 값입니다.",
        "",
    ]
    headers = [
        "방법",
        *[f"잔차@{k}" if k != k0 else f"잔차@{k} [95% CI]" for k in ks],
        f"원수익률@{k0}",
        f"{base} 대비 잔차@{k0}",
    ]
    rows = []
    for v, m in rm["methods"].items():
        cells = [_ci(ci.get(v)) if k == k0 else m[f"resid@{k}"] for k in ks]
        rows.append([v, *cells, m[f"raw@{k0}"], _diff(ci.get(v))])
    for label, b in rm["baselines"].items():
        title = "무작위" if label == "random" else f"같은 {LABEL_TITLES.get(label, label)} 전체"
        cells = [_ci(ci.get(f"baseline:{label}")) if k == k0 else b["resid"] for k in ks]
        rows.append([f"({title})", *cells, b["raw"], ""])
    return out + [_table(headers, rows), ""]


def _beyond_section(bm: dict) -> list[str]:
    k = bm["k"]
    b = bm["baselines"]
    out = [
        "## 3. GICS가 묶지 않는 연결",
        "",
        "텍스트 유사도가 산업분류를 흉내 내는 데 그치는지, 분류표에 없는 경제적 연결"
        "(공급망, 고객, 업종을 넘는 경쟁)까지 찾는지를 봅니다.",
        "",
        f"### 3.1 분류 밖에서 고른 이웃 (상위 {k}개)",
        "",
        f"회사마다 **자기 GICS 서브산업(또는 섹터) 밖에서만** 가장 비슷한 {k}개를 골라 "
        "잔차 수익률 상관을 쟀습니다. 기준선은 다음 두 가지입니다.",
        "",
        f"- 분류 밖 무작위 회사: 서브산업 밖 {_fmt(b['random_outside_sub'])}, "
        f"섹터 밖 {_fmt(b['random_outside_sector'])}",
        f"- 같은 섹터·다른 서브산업 회사: {_fmt(b['random_sector_peer'])}",
        "",
        "무작위보다 높으면 텍스트가 분류 밖에서도 실제로 같이 움직이는 회사를 찾는다는 뜻입니다. "
        "'같은 섹터 대비'까지 높으면, 그 연결이 GICS의 상위 분류(섹터)로도 설명되지 않는다는 뜻입니다.",
        "",
    ]
    headers = [
        "방법",
        "서브산업 밖 이웃",
        "무작위 대비",
        "같은 섹터·다른 서브산업 대비",
        "섹터 밖 이웃",
        "무작위 대비",
    ]
    rows = [
        [
            v,
            _ci(bm["outside_sub"][v]),
            _lift(bm["lift_outside_sub"][v]),
            _lift(bm["lift_vs_sector_peer"][v]),
            _ci(bm["outside_sector"][v]),
            _lift(bm["lift_outside_sector"][v]),
        ]
        for v in bm["outside_sub"]
    ]
    out += [_table(headers, rows), ""]

    out += [
        "### 3.2 GICS를 통제한 기업쌍 회귀",
        "",
        "`잔차상관 = a + b1·같은_서브산업 + b2·같은_섹터 + β·유사도(표준화)`",
        "",
        "β는 같은 분류인지를 고정했을 때 유사도가 1 표준편차 높으면 잔차 상관이 얼마나 오르는지입니다. "
        "β > 0이면 텍스트가 분류표에 없는 연결 정보를 담고 있습니다. 구간은 회사 단위 재표집으로 구했습니다.",
        "",
    ]
    rows = [
        [
            v,
            _diff({"diff": r["beta"], "diff_lo": r["lo"], "diff_hi": r["hi"]}),
            r["r2_gics"],
            r["r2_full"],
        ]
        for v, r in bm["regression"].items()
    ]
    out += [_table(["방법", "β (유사도 1SD당)", "R² (GICS만)", "R² (GICS+텍스트)"], rows), ""]

    out += [
        f"### 3.3 예시: 서브산업 밖에서 가장 가까운 회사 (`{bm['example_variant']}`)",
        "",
        "괄호 안은 이웃의 GICS 서브산업과 2025년 잔차 수익률 상관(ρ)입니다.",
        "",
    ]
    rows = []
    for t, ex in bm["examples"].items():
        cells = [
            f"{n['ticker']} ({n['sub_industry']}, ρ={n['resid_corr']:.2f})" for n in ex["neighbors"]
        ]
        rows.append([f"{t} ({ex['sub_industry']})", *cells])
    out += [_table(["회사 (서브산업)", "1", "2", "3"], rows), ""]
    return out


def _labels_section(cfg: Config, metrics: dict, ks: list[int], k0: int, base: str) -> list[str]:
    lci = metrics["labels_ci"]
    main = lci["label"]
    out = [
        "## 4. 산업분류 재현 (최소 조건)",
        "",
        "텍스트가 사업 내용을 담고 있는지 확인하는 최소 조건입니다. 무작위보다 크게 높아야 하지만, "
        "분류를 완벽히 재현할수록 좋은 것은 아닙니다. 예를 들어 Apple의 이웃으로 부품 공급사 "
        "Skyworks를 찾으면 여기서는 '틀린 답'이 되지만, 3장에서는 가치 있는 연결입니다.",
        "",
        "AUC는 0.5가 무작위, P@k는 상위 k개 이웃 중 같은 분류 비율입니다.",
        "",
    ]
    entries = {v: m[main] for v, m in metrics["labels"].items()}
    any_entry = next(iter(entries.values()))
    headers = [
        "방법",
        "AUC",
        *[f"P@{k} [95% CI]" if k == k0 else f"P@{k}" for k in ks],
        f"{base} 대비 P@{k0}",
    ]
    rows = []
    for v, e in entries.items():
        cells = [_ci(lci[f"p@{k}"].get(v)) if k == k0 else e[f"p@{k}"] for k in ks]
        rows.append([v, e["auc"], *cells, _diff(lci[f"p@{k0}"].get(v))])
    rows.append(["(무작위 기대값)", 0.5, *[any_entry[f"random@{k}"] for k in ks], ""])
    out += [
        f"### {LABEL_TITLES.get(main, main)} (n={any_entry['n']})",
        "",
        _table(headers, rows),
        "",
    ]
    others = [lab for lab in cfg.evaluation.labels if lab != main]
    if others:
        headers = ["방법", *[f"{LABEL_TITLES.get(lab, lab)} AUC / P@{k0}" for lab in others]]
        rows = [
            [v, *[f"{m[lab]['auc']:.3f} / {m[lab][f'p@{k0}']:.3f}" for lab in others]]
            for v, m in metrics["labels"].items()
        ]
        first = next(iter(metrics["labels"].values()))
        rows.append(
            ["(무작위 기대값)", *[f"0.500 / {first[lab][f'random@{k0}']:.3f}" for lab in others]]
        )
        out += ["### 다른 분류 기준", "", _table(headers, rows), ""]
    return out


def _agreement_section(metrics: dict, k: int) -> list[str]:
    if not metrics["agreement"]:
        return []
    names = list(metrics["labels"])
    jac = {(a["a"], a["b"]): a[f"jaccard@{k}"] for a in metrics["agreement"]}
    rows = []
    for i, a in enumerate(names, start=1):
        cells = []
        for b in names:
            if a == b:
                cells.append("·")
            else:
                cells.append(f"{jac.get((a, b), jac.get((b, a))):.2f}")
        rows.append([f"{i}. {a}", *cells])
    return [
        "## 5. 방법 간 일치도",
        "",
        f"상위 {k}개 이웃이 겹치는 정도(Jaccard)입니다. 1이면 같은 이웃, 0이면 전혀 다른 이웃입니다. "
        "값이 낮은 두 방법은 서로 다른 정보를 보고 있으므로 앙상블할 여지가 있습니다.",
        "",
        _table(["방법", *[str(i) for i in range(1, len(names) + 1)]], rows),
        "",
    ]


def _examples_section(cfg: Config, metrics: dict) -> list[str]:
    variant = best_variant(metrics, cfg)
    path = neighbors_path(cfg, variant)
    if not path.exists():
        return []
    neighbors = pd.read_parquet(path)
    tickers = set(neighbors["ticker"])
    examples = metrics.get("beyond", {}) or {}
    present = list(examples.get("examples", {})) or [
        t for t in list(dict.fromkeys(neighbors["ticker"]))[:8] if t in tickers
    ]
    rows = []
    for t in present:
        top = neighbors[neighbors["ticker"] == t].nsmallest(5, "rank")
        rows.append([t, *[f"{r.neighbor_ticker} ({r.percentile:.0f})" for r in top.itertuples()]])
    return [
        f"## 6. 예시: 유사 기업 상위 5개 (`{variant}`)",
        "",
        "괄호 안은 전체 기업쌍 중 백분위(0~100)입니다.",
        "",
        _table(["회사", "1", "2", "3", "4", "5"], rows),
        "",
    ]


def write_report(cfg: Config, metrics: dict, docs: pd.DataFrame, universe: pd.DataFrame) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(build_report(cfg, metrics, docs, universe) + "\n", encoding="utf-8")
    return cfg.report_path

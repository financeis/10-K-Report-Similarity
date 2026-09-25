"""연도별 관계도 비교 (docs/relation-map-plan.md 10장, 단계 3-2).

두 해의 관계도(graph.db)를 같은 회사끼리 맞춰 놓고, 관계가 유지·새로 보임·보이지 않음 중 무엇인지 가른다.
관계도는 10-K Item 1·1A에 적힌 것만 담으므로 '보이지 않음'은 관계가 끝났다는 뜻이 아니다. 그래서 그해에
왜 안 보이는지 이유를 함께 적는다 (앞에 있는 이유가 우선).

- not_analyzed: 다른 해에 근거가 있던 10-K 섹션을 그해에는 쓸 수 없었음(제출 없음, 추출 품질 문제, 요약만
  추출되어 길이가 4분의 1 아래로 줄어듦). 비교가 불완전하다
- not_mentioned: 그해 두 회사 10-K의 Item 1·1A에 상대 회사를 가리키는 언급이 없음(제외 문맥 밖, 판정 모델이
  그 회사가 아니라고 본 언급은 셈에서 뺌). 공시 문구가 바뀐 경우다
- rejected_by_review: 사람이 검수에서 거절함
- uncertain: 언급은 있고 그 관계가 '불확실(검수 대기)'로 남음
- not_related: 언급은 있지만 판정 모델이 그 관계로 채택하지 않음

회사는 티커로 맞춘다. 지주회사 전환 등으로 CIK가 바뀐 회사(BlackRock)도 같은 회사로 본다. 외부 기업은
node_id로 맞춘다. 익명 고객('Customer A')은 해마다 같은 고객인지 알 수 없어 비교에서 빼고 따로 센다.
"""

from __future__ import annotations

import json
import logging
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..pipeline import similarity_for
from . import research
from .reviews import span_full_text
from .stages import open_graph, relations_config

log = logging.getLogger(__name__)

RELATIONS = research.RELATIONS
STATUS = {"kept": "유지", "new": "새로 보임", "gone": "보이지 않음"}
REASONS = {
    "not_analyzed": "그해 10-K 섹션을 분석하지 못함",
    "not_mentioned": "그해 10-K에 상대 이름 없음",
    "rejected_by_review": "검수에서 거절",
    "uncertain": "불확실(검수 대기)로 남음",
    "not_related": "언급은 있으나 관계로 채택 안 됨",
}
"""관계가 보이지 않는 해의 이유 (우선순위 순서)."""
SHRUNK = 0.25
"""근거가 있던 섹션이 다른 해의 이 비율 아래로 줄면 추출 문제(요약만 추출 등)로 본다."""
EXCERPT = 150
"""예시 문장에서 상대 이름 앞뒤로 보여줄 글자 수."""


@dataclass
class YearGraph:
    """한 해 관계도를 회사 키(티커 또는 node_id)로 맞춘 것."""

    year: int
    config: str
    edges: pd.DataFrame
    """key(정렬한 회사 키 쌍), relation, state(confirmed/accepted/uncertain/rejected), edge_id, src, dst,
    status(시점)."""
    mentioned: set
    """상대 회사를 가리키는 언급(판정 모델이 '그 회사 아님'이라고 한 것은 뺌)이 있는 회사 키 쌍."""
    analyzed: dict
    """회사 키 → 그해 10-K를 분석했는가 (분석 대상 회사만)."""
    names: dict
    """회사 키 → 표시 이름 ('Name (TICKER)')."""
    evidence: dict
    """edge_id → [근거 원문 (도입문 포함)]."""
    evidence_sections: dict = field(default_factory=dict)
    """edge_id → {(근거를 적은 회사 키, 섹션)}."""
    section_chars: dict = field(default_factory=dict)
    """(회사 키, 섹션) → 분석한 정제 본문 길이. 분석하지 못한 섹션은 없다."""
    unit_texts: dict = field(default_factory=dict)
    """회사 키 쌍 → 그해 판정한 문장 원문(도입문 포함) 집합."""
    excerpts: dict = field(default_factory=dict)
    """edge_id → 첫 근거 문장에서 상대 이름을 가운데 둔 발췌."""
    anonymous: int = 0
    """익명 고객과의 관계 수 (비교에서 뺌)."""
    settings: dict = field(default_factory=dict)
    """판정 설정 (모델, 질문 판, 임계값)."""


def node_keys(nodes: pd.DataFrame) -> dict[str, str]:
    """node_id → 회사 키. 분석 대상 회사(company)는 티커, 나머지는 node_id."""
    out = {}
    for node, kind, ticker in zip(nodes["node_id"], nodes["kind"], nodes["ticker"], strict=True):
        out[node] = ticker if kind == "company" and isinstance(ticker, str) and ticker else node
    return out


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _lead(v) -> str | None:
    return v if isinstance(v, str) and v else None  # 도입문이 없으면 NULL(NaN)


def _excerpt(text: str, pos: int | None) -> str:
    if pos is None or pos < 0:
        return text[: 2 * EXCERPT]
    a, b = max(pos - EXCERPT, 0), min(pos + EXCERPT, len(text))
    return ("…" if a else "") + text[a:b] + ("…" if b < len(text) else "")


def load_year(cfg: Config) -> YearGraph:
    with closing(open_graph(cfg)) as con:
        nodes = pd.read_sql("SELECT node_id, kind, ticker, name, analyzed FROM nodes", con)
        edges = research.shown_edges(cfg, con)
        status = pd.read_sql("SELECT edge_id, status, model_id, question_version FROM edges", con)
        units = pd.read_sql(
            "SELECT u.pair_key, u.d_is_entity, s.text, s.lead_text FROM unit_judgements u "
            "JOIN spans s USING (span_id)",
            con,
        )
        ev = pd.read_sql(
            "SELECT e.edge_id, e.target_node, s.doc_node, s.section, s.text, s.lead_text, "
            "s.char_start AS span_start, m.char_start AS mention_start "
            "FROM edge_evidence e JOIN spans s USING (span_id) "
            "LEFT JOIN mentions m ON m.span_id = e.span_id AND m.target_node = e.target_node "
            "ORDER BY e.edge_id, e.ord, m.char_start",
            con,
        ).drop_duplicates(["edge_id", "text", "doc_node"])
        docs = pd.read_sql(
            "SELECT f.node_id, d.section, LENGTH(d.text) AS chars FROM documents d "
            "JOIN filings f USING (accession)",
            con,
        )
    key = node_keys(nodes)
    edges = edges.merge(status, on="edge_id", how="left")
    edges = edges.assign(
        key=[pair_key(key[a], key[b]) for a, b in zip(edges["src"], edges["dst"], strict=True)]
    )
    anonymous = edges["key"].str.contains("anon:") & edges["state"].isin(research.SHOWN)
    n_anon = int(anonymous.sum())
    edges = edges[~edges["key"].str.contains("anon:")].reset_index(drop=True)

    def keyed(pk: str) -> str:
        a, b = pk.split("|")
        return pair_key(key[a], key[b])

    entity = units[units["d_is_entity"] != "no"]
    mentioned = {keyed(pk) for pk in entity["pair_key"]}
    unit_texts: dict[str, set] = {}
    for pk, text, lead in zip(units["pair_key"], units["text"], units["lead_text"], strict=True):
        unit_texts.setdefault(keyed(pk), set()).add(span_full_text(text, _lead(lead)))
    companies = nodes[nodes["kind"] == "company"]
    analyzed = {key[n]: bool(a) for n, a in zip(companies["node_id"], companies["analyzed"],
                                                 strict=True)}  # fmt: skip
    names = {
        key[n]: (f"{name} ({t})" if isinstance(t, str) and k == "company" else str(name))
        for n, k, t, name in zip(nodes["node_id"], nodes["kind"], nodes["ticker"], nodes["name"],
                                 strict=True)
    }  # fmt: skip
    evidence: dict[str, list[str]] = {}
    sections: dict[str, set] = {}
    excerpts: dict[str, str] = {}
    for r in ev.itertuples():
        evidence.setdefault(r.edge_id, []).append(span_full_text(r.text, _lead(r.lead_text)))
        sections.setdefault(r.edge_id, set()).add((key.get(r.doc_node, r.doc_node), r.section))
        if r.edge_id not in excerpts:
            pos = None if pd.isna(r.mention_start) else int(r.mention_start - r.span_start)
            excerpts[r.edge_id] = _excerpt(r.text, pos)
    section_chars = {
        (key.get(n, n), s): int(c) for n, s, c in zip(docs["node_id"], docs["section"], docs["chars"],
                                                      strict=True)
    }  # fmt: skip
    rel = relations_config(cfg).judge
    settings = {
        "model": rel.model,
        "context": rel.context,
        "accept": rel.accept,
        "reject": rel.reject,
        "thresholds": {
            str(k): {n: x for n, x in v.model_dump().items() if x is not None}
            for k, v in rel.thresholds.items()
        },
        "question_versions": sorted(status["question_version"].dropna().unique().tolist()),
        "models": sorted(status["model_id"].dropna().unique().tolist()),
    }
    return YearGraph(cfg.filings.year, cfg.name, edges, mentioned, analyzed, names, evidence,
                     sections, section_chars, unit_texts, excerpts, n_anon, settings)  # fmt: skip


def _state(g: YearGraph) -> dict[tuple[str, str], pd.Series]:
    """(회사 키 쌍, 관계) → 그해 관계 한 줄. 같은 키에 여러 줄이면(CIK가 둘) 보이는 것을 우선한다."""
    rank = {"confirmed": 0, "accepted": 1, "uncertain": 2, "rejected": 3}
    e = g.edges.assign(_r=g.edges["state"].map(rank)).sort_values("_r")
    return {(k, r): row for (k, r), row in e.groupby(["key", "relation"]).first().iterrows()}


def _section_missing(missing: YearGraph, other: YearGraph, edge_id: str | None) -> bool:
    """다른 해에 근거가 있던 (회사, 섹션)을 그해에 분석하지 못했거나 크게 줄었는가."""
    if edge_id is None:
        return False
    for sec in other.evidence_sections.get(edge_id, set()):
        if sec not in missing.section_chars:
            return True
        if missing.section_chars[sec] < SHRUNK * other.section_chars.get(sec, 0):
            return True
    return False


def _reason(missing: YearGraph, other: YearGraph, key: str, relation: str, state: dict,
            other_edge: str | None) -> str:  # fmt: skip
    a, b = key.split("|")
    if any(missing.analyzed.get(x) is False for x in (a, b)):
        return "not_analyzed"
    if _section_missing(missing, other, other_edge):
        return "not_analyzed"
    if key not in missing.mentioned:
        return "not_mentioned"
    row = state.get((key, relation))
    if row is not None and row["state"] == "rejected":
        return "rejected_by_review"
    if row is not None and row["state"] == "uncertain":
        return "uncertain"
    return "not_related"


def compare(base: YearGraph, cur: YearGraph) -> pd.DataFrame:
    """관계(회사 키 쌍 × 유형)마다 두 해의 상태. 어느 한 해라도 관계도에 보인 것만."""
    sb, sc = _state(base), _state(cur)
    shown = set(research.SHOWN)
    keys = {k for k, row in sb.items() if row["state"] in shown}
    keys |= {k for k, row in sc.items() if row["state"] in shown}
    rows = []
    for key, relation in sorted(keys):
        in_b = (key, relation) in sb and sb[(key, relation)]["state"] in shown
        in_c = (key, relation) in sc and sc[(key, relation)]["state"] in shown
        eb = sb[(key, relation)]["edge_id"] if (key, relation) in sb else None
        ec = sc[(key, relation)]["edge_id"] if (key, relation) in sc else None
        status = "kept" if in_b and in_c else ("new" if in_c else "gone")
        reason, same_other = None, None
        if status == "new":
            reason = _reason(base, cur, key, relation, sb, ec)
            same_other = bool(set(cur.evidence.get(ec, [])) & base.unit_texts.get(key, set()))
        elif status == "gone":
            reason = _reason(cur, base, key, relation, sc, eb)
            same_other = bool(set(base.evidence.get(eb, [])) & cur.unit_texts.get(key, set()))
        texts_b = set(base.evidence.get(eb, [])) if in_b else set()
        texts_c = set(cur.evidence.get(ec, [])) if in_c else set()
        shown_edge = ec if in_c else eb
        shown_state = sc if in_c else sb
        rows.append(
            {
                "key": key,
                "relation": relation,
                "status": status,
                "reason": reason,
                "base_edge": eb,
                "cur_edge": ec,
                "timing": shown_state[(key, relation)]["status"] if shown_edge else None,
                "same_text": bool(texts_b & texts_c) if status == "kept" else None,
                "same_text_other_year": same_other,
                "n_evidence_base": len(texts_b),
                "n_evidence_cur": len(texts_c),
            }
        )
    return pd.DataFrame(rows)


def similarity_stability(base_cfg: Config, cur_cfg: Config, k: int) -> dict:
    """두 해 텍스트 유사도(관계도 후보에 쓴 방법)의 상위 k 이웃이 얼마나 겹치나.

    두 해 모두 유사도가 있는 회사(티커)끼리만 보고, 상위 k도 그 회사들 안에서 다시 잰다."""
    rb, rc = relations_config(base_cfg).similarity, relations_config(cur_cfg).similarity
    cb, sb = similarity_for(base_cfg, rb)
    cc, sc = similarity_for(cur_cfg, rc)
    common = sorted(set(cb["ticker"]) & set(cc["ticker"]))
    ib = [cb.index[cb["ticker"] == t][0] for t in common]
    ic = [cc.index[cc["ticker"] == t][0] for t in common]
    a, b = sb[np.ix_(ib, ib)].astype(float), sc[np.ix_(ic, ic)].astype(float)
    np.fill_diagonal(a, -np.inf)
    np.fill_diagonal(b, -np.inf)
    ta, tb = np.argsort(-a, axis=1)[:, :k], np.argsort(-b, axis=1)[:, :k]
    overlap = np.array([len(set(x) & set(y)) / k for x, y in zip(ta, tb, strict=True)])
    top1 = float(np.mean(ta[:, 0] == tb[:, 0]))
    iu = np.triu_indices(len(common), 1)
    rank_corr = float(pd.Series(a[iu]).corr(pd.Series(b[iu]), method="spearman"))
    return {
        "similarity": rc,
        "k": k,
        "n_firms": len(common),
        "mean_overlap": float(overlap.mean()),
        "overlap_quartiles": [float(q) for q in np.percentile(overlap, [25, 50, 75])],
        "top1_same": top1,
        "pair_rank_corr": rank_corr,
    }


def filers(table: pd.DataFrame, g: YearGraph, status: str, n: int = 5) -> list[tuple[str, int]]:
    """'그해 10-K에 상대 이름 없음'으로 보이지 않게 된 관계가 많은 회사 (근거를 적었던 회사 기준)."""
    t = table[(table["status"] == status) & (table["reason"] == "not_mentioned")]
    col = "base_edge" if status == "gone" else "cur_edge"
    counts: dict[str, int] = {}
    for edge in t[col].dropna():
        for filer in {f for f, _ in g.evidence_sections.get(edge, set())}:
            counts[filer] = counts.get(filer, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]


def analyze(base_cfg: Config, cur_cfg: Config) -> dict:
    base, cur = load_year(base_cfg), load_year(cur_cfg)
    table = compare(base, cur)
    shown = set(research.SHOWN)
    by_type = {}
    for r in RELATIONS:
        t = table[table["relation"] == r]
        entry = {
            "base": int((base.edges["relation"].eq(r) & base.edges["state"].isin(shown)).sum()),
            "cur": int((cur.edges["relation"].eq(r) & cur.edges["state"].isin(shown)).sum()),
            "base_pairs": int(t["status"].isin(["kept", "gone"]).sum()),
            "cur_pairs": int(t["status"].isin(["kept", "new"]).sum()),
        }
        for s in STATUS:
            entry[s] = int((t["status"] == s).sum())
            entry[f"{s}_reasons"] = {
                k: int(v) for k, v in t.loc[t["status"] == s, "reason"].value_counts().items()
            }
        kept = t[t["status"] == "kept"]
        entry["kept_same_text"] = float(kept["same_text"].mean()) if len(kept) else None
        flips = t[t["reason"].isin(["uncertain", "not_related", "rejected_by_review"])]
        entry["flips"] = int(len(flips))
        entry["flips_same_text"] = int(flips["same_text_other_year"].fillna(False).sum())
        by_type[r] = entry
    both = {k for k, v in base.analyzed.items() if v and cur.analyzed.get(k)}
    return {
        "base": {"config": base.config, "year": base.year, "settings": base.settings},
        "cur": {"config": cur.config, "year": cur.year, "settings": cur.settings},
        "same_settings": _settings_key(base.settings) == _settings_key(cur.settings),
        "n_firms_analyzed": {
            "base": sum(base.analyzed.values()),
            "cur": sum(cur.analyzed.values()),
            "both": len(both),
        },
        "only_analyzed": {
            "base": sorted(k for k, v in base.analyzed.items() if v and not cur.analyzed.get(k)),
            "cur": sorted(k for k, v in cur.analyzed.items() if v and not base.analyzed.get(k)),
        },
        "anonymous": {"base": base.anonymous, "cur": cur.anonymous},
        "by_type": by_type,
        "dropped_names": [(base.names.get(f, f), c) for f, c in filers(table, base, "gone")],
        "added_names": [(cur.names.get(f, f), c) for f, c in filers(table, cur, "new")],
        "similarity": similarity_stability(base_cfg, cur_cfg, relations_config(cur_cfg).top_k),
        "examples": examples(table, base, cur),
        "table": table,
    }


def _settings_key(s: dict) -> dict:
    """두 해 판정 설정을 비교할 항목 (실제로 쓰인 모델 목록은 설정의 모델과 같아야 한다)."""
    return {k: s.get(k) for k in ("model", "context", "accept", "reject", "thresholds",
                                  "question_versions")}  # fmt: skip


def examples(table: pd.DataFrame, base: YearGraph, cur: YearGraph, n: int = 8) -> dict:
    """새로 보인 관계와 보이지 않게 된 관계의 예. 근거 문장이 많은(여러 번 적힌) 관계부터."""
    out = {}
    for status, g, col, count in (("new", cur, "cur_edge", "n_evidence_cur"),
                                  ("gone", base, "base_edge", "n_evidence_base")):  # fmt: skip
        for r in ("business", "competitor"):
            t = table[(table["status"] == status) & (table["relation"] == r)]
            t = t[t["reason"] != "not_analyzed"]
            t = t.sort_values([count, "key"], ascending=[False, True]).head(n)
            out[f"{status}_{r}"] = [
                {
                    "pair": " – ".join(g.names.get(x, x) for x in row.key.split("|")),
                    "reason": row.reason,
                    "timing": row.timing,
                    "text": g.excerpts.get(getattr(row, col), ""),
                }
                for row in t.itertuples()
            ]
    return out


# ---------------------------------------------------------------- 보고서

TIMING = {"current": "현재", "historical": "과거", "planned": "계획", "unclear": "불분명"}


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.0%}"


def _settings_line(s: dict) -> str:
    th = ", ".join(
        f"{q} {'/'.join(f'{k} {v}' for k, v in t.items())}" for q, t in s["thresholds"].items()
    )
    return (f"`{s['model']}`, 질문 {'·'.join(s['question_versions'])}, 입력 {s['context']}, "
            f"채택 {s['accept']} / 기각 {s['reject']} (질문별: {th})")  # fmt: skip


def format_report(result: dict) -> str:
    b, c = result["base"]["year"], result["cur"]["year"]
    bt, sim, nf = result["by_type"], result["similarity"], result["n_firms_analyzed"]
    only, anon = result["only_analyzed"], result["anonymous"]
    labels = {"competitor": "경쟁", "business": "공급·협력", "equity": "지분 (검증 전)"}
    same = result["same_settings"]
    lines = [
        f"# {result['cur']['config']} 관계도 연도 비교 ({b} → {c}년 제출 10-K)",
        "",
        f"- 생성: {datetime.now().isoformat(timespec='seconds')}",
        f"- 비교: `{result['base']['config']}`({b}년 제출 10-K)와 `{result['cur']['config']}`({c}년 제출 "
        "10-K)의 관계도입니다. 제출 연도 기준이라, 대부분은 한 해 전 회계연도의 10-K입니다.",
        f"- 판정 설정: {_settings_line(result['cur']['settings'])}."
        + (
            " 두 해가 같습니다."
            if same
            else f" **{b}년 설정과 다릅니다**: {_settings_line(result['base']['settings'])}."
        ),
        "- 관계도에 보이는 관계(모델 채택 + 검수로 확인)끼리 비교합니다. 회사는 티커로 맞춥니다(CIK가 바뀐 "
        "회사 포함).",
        f"- 분석한 회사: {b}년 {nf['base']}곳, {c}년 {nf['cur']}곳, 두 해 모두 {nf['both']}곳. "
        f"{b}년에만: {', '.join(only['base']) or '없음'}. {c}년에만: {', '.join(only['cur']) or '없음'}.",
        f"- 익명 고객('Customer A' 등)과의 관계는 해마다 같은 고객인지 알 수 없어 비교에서 뺐습니다({b}년 "
        f"{anon['base']}개, {c}년 {anon['cur']}개).",
        "- **'보이지 않음'은 관계가 끝났다는 뜻이 아닙니다.** 그해 10-K에 적히지 않았거나, 판정이 달라졌거나, "
        "그해 10-K를 분석하지 못한 경우입니다. 이유를 함께 셉니다.",
        "",
        "## 1. 유형별 변화",
        "",
    ]
    rows = []
    for r, e in bt.items():
        rows.append([labels[r], f"{e['base']:,}", f"{e['cur']:,}", f"{e['kept']:,}", f"{e['new']:,}",
                     f"{e['gone']:,}", _pct(e["kept"] / e["base_pairs"] if e["base_pairs"] else None),
                     _pct(e["kept_same_text"])])  # fmt: skip
    lines += research._table(
        ["관계", f"{b}년", f"{c}년", "유지", "새로 보임", "보이지 않음", f"{b}년 관계 중 유지",
         "유지된 관계 중 같은 문장"],
        rows,
    )  # fmt: skip
    lines += [
        "",
        f"- {b}년 수는 검수로 확인한 관계를 포함합니다.",
        "- '같은 문장'은 두 해 근거 문장 가운데 글자까지 같은 문장이 하나라도 있는 비율입니다. 10-K는 "
        "해마다 같은 문단을 되풀이하는 경우가 많습니다.",
        "",
        "## 2. 새로 보이거나 보이지 않게 된 이유",
        "",
        f"새로 보인 관계는 {b}년에, 보이지 않게 된 관계는 {c}년에 왜 없었는지입니다.",
        "",
    ]
    rows = []
    for r, e in bt.items():
        for s in ("new", "gone"):
            reasons = e[f"{s}_reasons"]
            rows.append([labels[r], STATUS[s], f"{e[s]:,}",
                         *(f"{reasons.get(k, 0):,}" for k in REASONS)])  # fmt: skip
    lines += research._table(["관계", "변화", "합계", *REASONS.values()], rows)
    dropped = ", ".join(f"{n} {k}개" for n, k in result["dropped_names"])
    added = ", ".join(f"{n} {k}개" for n, k in result["added_names"])
    flips = " · ".join(
        f"{labels[r]} {e['flips']}개 중 {e['flips_same_text']}개"
        for r, e in bt.items()
        if e["flips"]
    )
    lines += [
        "",
        "- **'그해 10-K에 상대 이름 없음'이 공시 문구가 바뀐 경우입니다.** 그해 두 회사 10-K의 Item 1·1A에 "
        "상대 회사를 가리키는 이름이 없습니다(제외 문맥의 언급과, 판정 모델이 그 회사가 아니라고 본 언급은 "
        "세지 않음).",
        f"  - 이렇게 보이지 않게 된 관계를 많이 적었던 회사({b}년 근거 기준): {dropped or '없음'}.",
        f"  - 새로 이름을 적은 회사({c}년 근거 기준): {added or '없음'}.",
        "- '분석하지 못함'에는 그해 10-K가 없거나 추출 품질 검사를 통과하지 못한 경우와, 근거가 있던 섹션이 "
        "요약만 추출되어 다른 해의 4분의 1 아래로 줄어든 경우가 들어갑니다.",
        "- '불확실'·'채택 안 됨'·'검수에서 거절'은 두 해 모두 이름은 나왔는데 한 해만 채택된 관계입니다. "
        f"두 해 문장이 글자까지 같은데도 판정이 달라진 것: {flips or '없음'}. 판정 입력의 앞뒤 문단이 달라지면 "
        "경계값 근처의 점수가 바뀝니다.",
        "",
        "## 3. 텍스트 유사도의 연도 안정성",
        "",
        f"관계도 후보에 쓰는 `{sim['similarity']}` 유사도(Item 1)로, 두 해 모두 유사도가 있는 {sim['n_firms']}개사의 "
        f"상위 {sim['k']} 이웃을 비교했습니다(이웃 순위는 이 회사들 안에서 다시 잼).",
        "",
        f"- 상위 {sim['k']} 이웃 가운데 이듬해에도 상위 {sim['k']}인 비율: 평균 {_pct(sim['mean_overlap'])} "
        f"(사분위 {', '.join(_pct(q) for q in sim['overlap_quartiles'])}).",
        f"- 가장 비슷한 회사(1위)가 같은 비율: {_pct(sim['top1_same'])}.",
        f"- 모든 기업쌍 유사도의 순위 상관(스피어만): {sim['pair_rank_corr']:.3f}.",
        "",
        "## 4. 예시",
        "",
        "분석하지 못한 10-K 때문에 생긴 변화는 뺐습니다. 근거 문장은 상대 회사 이름을 가운데 두고 앞뒤 "
        f"{EXCERPT}자씩 보여줍니다.",
        "",
    ]
    ex = result["examples"]
    for key, title in (
        ("new_business", f"{c}년에 새로 보인 공급·협력"),
        ("new_competitor", f"{c}년에 새로 보인 경쟁"),
        ("gone_business", f"{c}년에 보이지 않게 된 공급·협력"),
        ("gone_competitor", f"{c}년에 보이지 않게 된 경쟁"),
    ):
        lines += [f"### {title}", ""]
        items = ex.get(key) or []
        if not items:
            lines += ["해당하는 관계가 없습니다.", ""]
            continue
        year_of_text = c if key.startswith("new") else b
        rows = [[x["pair"], REASONS.get(x["reason"], "–"), TIMING.get(x["timing"], "–"),
                 x["text"].replace("|", "\\|").replace("\n", " ")]
                for x in items]  # fmt: skip
        lines += research._table(
            ["회사", f"{b if key.startswith('new') else c}년에 없던 이유", "시점",
             f"{year_of_text}년 근거 문장"],
            rows,
        )  # fmt: skip
        lines.append("")
    lines += [
        "## 5. 해석할 때 주의할 점",
        "",
        "- 관계도는 10-K Item 1·1A에 이름이 적힌 관계만 담습니다. 회사가 거래처·경쟁사 이름을 적는 방식은 "
        "해마다 바뀌므로, 공시 변화가 곧 거래 변화는 아닙니다.",
        "- 두 해 모두 같은 판정 모델과 임계값을 씁니다. 판정 정확도는 2024년 확인 표본으로만 검증했습니다.",
        "- 회사 목록은 두 해 모두 현재 S&P 500입니다(생존 편향).",
    ]
    return "\n".join(lines)


def report_path(cfg: Config) -> Path:
    return cfg.reports_dir / f"relations_changes_{cfg.name}.md"


def stage_changes(cfg: Config, base_cfg: Config) -> tuple[dict, Path]:
    """두 해 관계도를 비교해 relations/changes.parquet·changes.json과 보고서를 쓴다."""
    result = analyze(base_cfg, cfg)
    table = result.pop("table")
    out = cfg.relations_dir
    table.to_parquet(out / "changes.parquet", index=False)
    (out / "changes.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    path = report_path(cfg)
    path.write_text(format_report(result) + "\n", encoding="utf-8")
    return result, path

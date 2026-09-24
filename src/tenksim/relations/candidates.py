"""후보 쌍: (유사도 상위 K) ∪ (이름 언급 쌍). 판정 전 상태(pending)로 저장한다.

후보는 방향 없는 쌍 {A, B}이고 양 끝은 node_id 순서로 정렬한다(node_a < node_b).
유사도는 분석 대상 회사끼리만 있으므로 외부 기업·익명 고객과의 쌍은 언급으로만 들어온다.

쌍마다 판정에 넣을 근거 구간도 고른다. 한 회사의 10-K 쪽마다 최대 max_per_side개이고,
개수만 자르면 경쟁사 목록 문장이 공급 문장을 밀어낼 수 있어 단서 단어(경쟁·고객/공급·협력·지분)
종류별로 번갈아 고른다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..similarity import pair_percentiles
from .mentions import MentionTables
from .names import company_node

# 근거 구간을 고를 때만 쓰는 대략의 단서. 관계 판정이 아니다.
CUES: dict[str, re.Pattern] = {
    "competition": re.compile(r"\bcompet", re.I),
    "supply": re.compile(
        r"\bcustomers?\b|\bclients?\b|% of (?:our )?(?:total )?(?:net )?(?:revenue|sales)|"
        r"\bsales to\b|\bsuppl(?:y|ier|iers|ied|ies)\b|\bvendors?\b|\bfoundr|\bsole[- ]source|"
        r"\bsingle[- ]source|contract manufactur|\bpurchase[sd]? (?:from|of)\b|\bsource[sd]? from\b",
        re.I,
    ),
    "partnership": re.compile(
        r"\bpartner|\balliance|\bjoint venture|\bcollaborat|\blicens|\bagreement with\b|"
        r"\bco-develop|\bresell|\bdistribut",
        re.I,
    ),
    # "their own products"의 own은 지분이 아니므로 owns/owned만 본다
    "ownership": re.compile(
        r"\bequity (?:interest|stake|method)|\bowns\b|\bowned by\b|\b(?:stake|shares) in\b", re.I
    ),
}
_CUE_ORDER = [*CUES, "none"]

CANDIDATE_COLUMNS = [
    "pair_key", "node_a", "node_b", "source", "rank_ab", "rank_ba", "similarity_pct",
    "mentions_ab", "mentions_ba", "n_spans", "status",
]  # fmt: skip
"""rank_ab: A의 유사 기업 중 B의 순위(1 = 가장 비슷). mentions_ab: A의 10-K가 B를 언급한 횟수."""


def pair_key(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x}|{y}"


def span_cues(text: str) -> list[str]:
    return [name for name, rx in CUES.items() if rx.search(text)]


@dataclass
class Candidates:
    candidates: pd.DataFrame
    candidate_spans: pd.DataFrame
    """pair_key, span_id, doc_node(구간이 나온 10-K), order(판정 입력 순서), cues"""


def similarity_ranks(sim: np.ndarray) -> np.ndarray:
    """ranks[i, j] = i의 유사 기업 중 j의 순위(1부터). 대각선은 0."""
    s = np.array(sim, dtype=np.float64)
    np.fill_diagonal(s, np.inf)
    order = np.argsort(-s, axis=1, kind="stable")
    ranks = np.empty_like(order)
    n = len(s)
    for i in range(n):
        ranks[i, order[i]] = np.arange(n)
    return ranks


def build_candidates(
    tables: MentionTables,
    companies: pd.DataFrame | None,
    sim: np.ndarray | None,
    *,
    top_k: int,
    max_per_side: int,
) -> Candidates:
    """companies: 유사도 행렬의 행 순서대로 cik. 유사도를 안 쓰면 None."""
    rows: dict[str, dict] = {}

    def row(a: str, b: str) -> dict:
        key = pair_key(a, b)
        if key not in rows:
            x, y = key.split("|")
            rows[key] = {
                "pair_key": key, "node_a": x, "node_b": y, "from_similarity": False,
                "from_mention": False, "rank_ab": None, "rank_ba": None,
                "similarity_pct": None, "mentions_ab": 0, "mentions_ba": 0,
            }  # fmt: skip
        return rows[key]

    # 1) 유사도 상위 K (어느 한쪽 기준으로라도 K 안이면 후보)
    if sim is not None and companies is not None:
        nodes = [company_node(c) for c in companies["cik"]]
        ranks = similarity_ranks(sim)
        pct = pair_percentiles(sim)
        near = np.argwhere((ranks >= 1) & (ranks <= top_k))
        for i, j in near:
            row(nodes[i], nodes[j])["from_similarity"] = True
        pos = {n: k for k, n in enumerate(nodes)}

    # 2) 이름 언급 (제외 표시가 없는 것만)
    live = tables.mentions[tables.mentions["excluded"].isna()]
    for (doc, target), n in live.groupby(["doc_node", "target_node"]).size().items():
        r = row(doc, target)
        r["from_mention"] = True
        r["mentions_ab" if doc == r["node_a"] else "mentions_ba"] += int(n)

    # 순위와 백분위는 두 회사 모두 유사도 행렬에 있을 때만
    if sim is not None and companies is not None:
        for r in rows.values():
            i, j = pos.get(r["node_a"]), pos.get(r["node_b"])
            if i is not None and j is not None:
                r["rank_ab"], r["rank_ba"] = int(ranks[i, j]), int(ranks[j, i])
                r["similarity_pct"] = float(pct[i, j])

    spans = _select_spans(live, tables.spans, set(rows), max_per_side)
    n_spans = spans.groupby("pair_key").size()
    cand = pd.DataFrame(list(rows.values()))
    cand["source"] = np.select(
        [cand["from_similarity"] & cand["from_mention"], cand["from_mention"]],
        ["both", "mention"],
        "similarity",
    )
    cand["n_spans"] = cand["pair_key"].map(n_spans).fillna(0).astype(int)
    cand["status"] = "pending"
    for col in ("rank_ab", "rank_ba"):
        cand[col] = cand[col].astype("Int64")
    cand = cand[CANDIDATE_COLUMNS].sort_values("pair_key").reset_index(drop=True)
    return Candidates(cand, spans)


def _select_spans(
    live: pd.DataFrame, spans: pd.DataFrame, keys: set[str], max_per_side: int
) -> pd.DataFrame:
    """쌍마다, 10-K 쪽마다 근거 구간을 단서 종류별로 번갈아 max_per_side개까지 고른다."""
    text = spans.set_index("span_id")
    full = {
        sid: (lead + " " if isinstance(lead, str) else "") + t
        for sid, t, lead in zip(spans["span_id"], spans["text"], spans["lead_text"], strict=True)
    }
    by_side: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in live.drop_duplicates(["span_id", "target_node"]).itertuples():
        key = pair_key(m.doc_node, m.target_node)
        if key in keys:
            by_side[(key, m.doc_node)].append(m.span_id)
    out = []
    for (key, doc), span_ids in by_side.items():
        # 문서 안 순서대로 두고, 단서 종류별 대기열에서 하나씩 돌아가며 뽑는다
        span_ids = sorted(
            set(span_ids), key=lambda s: (text.at[s, "section"], text.at[s, "char_start"])
        )
        queues: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for sid in span_ids:
            cues = span_cues(full[sid])
            queues[cues[0] if cues else "none"].append((sid, cues))
        picked = []
        while len(picked) < max_per_side and any(queues.values()):
            for cue in _CUE_ORDER:
                if queues[cue] and len(picked) < max_per_side:
                    picked.append(queues[cue].pop(0))
        picked.sort(key=lambda p: span_ids.index(p[0]))
        for order, (sid, cues) in enumerate(picked):
            out.append(
                {
                    "pair_key": key,
                    "span_id": sid,
                    "doc_node": doc,
                    "order": order,
                    "cues": ",".join(cues),
                }  # fmt: skip
            )
    return pd.DataFrame(out, columns=["pair_key", "span_id", "doc_node", "order", "cues"])

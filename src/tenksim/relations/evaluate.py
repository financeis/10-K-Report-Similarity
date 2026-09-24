"""판정 채점: 표본 검수 라벨과 모델 판정 비교 (docs/relation-map-plan.md 7.6).

단위(근거 구간 × 언급된 회사)마다 라벨과 점수를 나란히 놓고 세 가지를 잰다.
- 질문별: 점수가 라벨을 얼마나 잘 가르는지(AUC), 임계값에서 채택·불확실·기각이 얼마나 맞는지
- 단위별 종합: 회사 식별과 관계 질문을 7.4의 규칙대로 합쳤을 때
- 쌍별 종합: 같은 쌍의 단위를 합친 관계 (화면에 선으로 그려질 것)
검수 라벨은 방향·세부 유형까지 적혀 있지만, 판정은 경쟁·공급·협력·지분 세 가지만 하므로
LABEL_GROUPS로 묶어 채점한다.
정밀도·재현율에는 Wilson 95% 구간을 붙인다. 표본이 작아 구간이 넓다는 것을 함께 봐야 한다.
"""

from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pandas as pd

from .judge import Judgement, decide
from .judge.questions import ENTITY_Q, RELATION_QUESTIONS, STATUS_Q, YES_NO_QUESTIONS
from .reviews import LATEST_LABELS

LABEL_GROUPS = {
    "competitor": ("competitor",),
    "business": ("doc_supplies_target", "target_supplies_doc", "partner"),
    "equity": ("doc_owns_target", "target_owns_doc"),
}
"""판정 관계 → 그에 해당하는 검수 라벨 코드."""
CALIBRATION_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


def load_labels(reviews: sqlite3.Connection, sample_id: str) -> pd.DataFrame:
    """표본의 검수 단위와 최신 라벨. 아직 검수하지 않은 단위는 labeled=False."""
    df = pd.read_sql(
        f"""
        SELECT u.ord, u.unit_id, u.span_id, u.doc_node, u.target_node, u.pair_key,
               l.label_id IS NOT NULL AS labeled, l.is_entity, l.relations, l.status,
               l.partner_type, COALESCE(l.skipped, 0) AS skipped, l.note, l.span_text
        FROM sample_units u
        LEFT JOIN ({LATEST_LABELS}) l ON l.unit_id = u.unit_id
        WHERE u.sample_id = ?
        ORDER BY u.ord
        """,
        reviews,
        params=(sample_id,),
    )
    if df.empty:
        raise ValueError(f"표본이 없거나 비어 있습니다: {sample_id}")
    df["labeled"] = df["labeled"].astype(bool)
    df["skipped"] = df["skipped"].astype(bool)
    df["relations"] = df["relations"].map(lambda r: json.loads(r) if isinstance(r, str) else [])
    return df


def unit_table(
    labels: pd.DataFrame, judgements: dict[str, Judgement], accept: float, reject: float
) -> pd.DataFrame:
    """라벨 + 질문별 점수·결정. 점수 열은 s_<질문>, 결정 열은 d_<질문>."""
    t = labels.copy()
    t["judged"] = t["unit_id"].isin(judgements.keys())
    for q in YES_NO_QUESTIONS:
        answers = [judgements[u].answers.get(q) if u in judgements else None for u in t["unit_id"]]
        t[f"s_{q}"] = [a.get("score") if a else np.nan for a in answers]
        t[f"d_{q}"] = [decide(a, accept, reject) if a else None for a in answers]
    t[f"c_{STATUS_Q}"] = [
        (judgements[u].answers.get(STATUS_Q) or {}).get("choice") if u in judgements else None
        for u in t["unit_id"]
    ]
    for r, codes in LABEL_GROUPS.items():
        t[f"y_{r}"] = t["relations"].map(lambda rels, codes=codes: any(c in rels for c in codes))
    records = t.to_dict("records")
    for r in RELATION_QUESTIONS:
        t[f"p_{r}"] = [_combine(row, r) for row in records]
    return t


def _combine(row: dict, relation: str) -> str | None:
    """7.4 규칙: 회사 식별이 기각되면 기각, 둘 다 채택이면 채택, 나머지는 불확실."""
    decisions = [row[f"d_{q}"] for q in (ENTITY_Q, relation)]
    if any(d is None for d in decisions):
        return None
    if "no" in decisions:
        return "no"
    if all(d == "yes" for d in decisions):
        return "yes"
    return "abstain"


# ---------------------------------------------------------------- 지표


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def _ratio(k: int, n: int) -> float | None:
    return round(k / n, 3) if n else None


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.all() or not y.any():
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, s)), 3)


def score_question(y: pd.Series, s: pd.Series, d: pd.Series) -> dict:
    """y: 라벨(bool), s: 점수, d: 결정(yes/no/abstain)."""
    y, s, d = y.to_numpy(bool), s.to_numpy(float), d.to_numpy(object)
    n, pos = len(y), int(y.sum())
    out = {"n": n, "n_pos": pos}
    if not n:
        return out
    half = s >= 0.5
    tp, fp = int((half & y).sum()), int((half & ~y).sum())
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, pos)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else None
    yes, no, abstain = d == "yes", d == "no", d == "abstain"
    k_yes = int((yes & y).sum())
    out |= {
        "auc": _auc(y, s),
        "brier": round(float(np.mean((s - y) ** 2)), 3),
        "at_0.5": {"precision": precision, "recall": recall, "f1": f1},
        "accepted": int(yes.sum()),
        "accepted_precision": _ratio(k_yes, int(yes.sum())),
        "accepted_precision_ci": wilson(k_yes, int(yes.sum())),
        "accepted_recall": _ratio(k_yes, pos),
        "accepted_recall_ci": wilson(k_yes, pos),
        "uncertain": int(abstain.sum()),
        "uncertain_pos": int((abstain & y).sum()),
        "rejected": int(no.sum()),
        "rejected_pos": int((no & y).sum()),
        "auto_rate": _ratio(int((yes | no).sum()), n),
    }
    return out


def score_decisions(y: pd.Series, p: pd.Series) -> dict:
    """종합 결정(yes/no/abstain)만 있을 때."""
    y, p = y.to_numpy(bool), p.to_numpy(object)
    yes, no, abstain = p == "yes", p == "no", p == "abstain"
    k_yes, pos = int((yes & y).sum()), int(y.sum())
    return {
        "n": len(y),
        "n_pos": pos,
        "accepted": int(yes.sum()),
        "accepted_precision": _ratio(k_yes, int(yes.sum())),
        "accepted_precision_ci": wilson(k_yes, int(yes.sum())),
        "accepted_recall": _ratio(k_yes, pos),
        "accepted_recall_ci": wilson(k_yes, pos),
        "uncertain": int(abstain.sum()),
        "uncertain_pos": int((abstain & y).sum()),
        "rejected": int(no.sum()),
        "rejected_pos": int((no & y).sum()),
    }


def pair_table(t: pd.DataFrame) -> pd.DataFrame:
    """쌍 × 관계마다 라벨(어느 단위든 있으면 참)과 종합 결정(채택 > 불확실 > 기각).
    두 회사 중 누구의 10-K에서 나온 문장이든 같은 쌍으로 합친다."""
    rows = [
        {"pair_key": u["pair_key"], "relation": r,
         "label": u["is_entity"] == "yes" and u[f"y_{r}"], "pred": u[f"p_{r}"]}
        for u in t.to_dict("records")
        for r in RELATION_QUESTIONS
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    rank = {"yes": 2, "abstain": 1, "no": 0}
    g = df.groupby(["pair_key", "relation"])
    out = g["label"].any().to_frame()
    out["pred"] = g["pred"].agg(lambda s: max(s, key=lambda v: rank.get(v, -1)))
    return out.reset_index()


def evaluate(t: pd.DataFrame) -> dict:
    labeled = t[t["labeled"] & ~t["skipped"] & t["judged"]]
    entity_known = labeled[labeled["is_entity"].isin(["yes", "no"])]
    entity_yes = labeled[labeled["is_entity"] == "yes"]
    out: dict = {
        "units": {
            "n": int(len(t)),
            "labeled": int(t["labeled"].sum()),
            "skipped": int((t["labeled"] & t["skipped"]).sum()),
            "judged": int(t["judged"].sum()),
            "scored": int(len(labeled)),
            "entity_unsure": int((labeled["is_entity"] == "unsure").sum()),
        },
        "questions": {},
    }
    q = out["questions"]
    q[ENTITY_Q] = score_question(
        entity_known["is_entity"] == "yes", entity_known[f"s_{ENTITY_Q}"],
        entity_known[f"d_{ENTITY_Q}"],
    )  # fmt: skip
    for r in RELATION_QUESTIONS:
        q[r] = score_question(entity_yes[f"y_{r}"], entity_yes[f"s_{r}"], entity_yes[f"d_{r}"])

    # 단위별 종합: 회사가 아니라고 라벨된 단위도 넣는다 (식별에서 걸러야 하므로)
    out["units_combined"] = {
        r: score_decisions(
            (entity_known["is_entity"] == "yes") & entity_known[f"y_{r}"], entity_known[f"p_{r}"]
        )
        for r in RELATION_QUESTIONS
    }

    pairs = pair_table(entity_known)
    out["pairs"] = {
        rel: score_decisions(g["label"], g["pred"]) for rel, g in pairs.groupby("relation")
    }
    out["pairs"]["all"] = score_decisions(pairs["label"], pairs["pred"])
    out["pairs"]["n_pairs"] = int(pairs["pair_key"].nunique())

    has_status = entity_yes[entity_yes["relations"].map(bool) & entity_yes["status"].notna()]
    out["status"] = _choice_accuracy(has_status["status"], has_status[f"c_{STATUS_Q}"])
    out["calibration"] = _calibration(entity_yes)
    return out


def _choice_accuracy(label: pd.Series, pred: pd.Series) -> dict:
    n = len(label)
    k = int((label.to_numpy() == pred.to_numpy()).sum())
    confusion = pd.crosstab(label, pred).to_dict() if n else {}
    return {"n": n, "accuracy": _ratio(k, n), "accuracy_ci": wilson(k, n), "confusion": confusion}


def _calibration(entity_yes: pd.DataFrame) -> list[dict]:
    """관계 질문을 모아 점수 구간별 실제 정답 비율."""
    s = np.concatenate([entity_yes[f"s_{r}"].to_numpy(float) for r in RELATION_QUESTIONS])
    y = np.concatenate([entity_yes[f"y_{r}"].to_numpy(bool) for r in RELATION_QUESTIONS])
    out = []
    edges = CALIBRATION_BINS
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        out.append(
            {"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
             "mean_score": round(float(s[m].mean()), 3) if m.any() else None,
             "observed": _ratio(int(y[m].sum()), int(m.sum()))}
        )  # fmt: skip
    return out


def disagreements(t: pd.DataFrame) -> pd.DataFrame:
    """라벨과 종합 결정이 어긋난 단위 (질문 문구를 고칠 때 보는 목록)."""
    rows = []
    for u in t[t["labeled"] & ~t["skipped"] & t["judged"]].to_dict("records"):
        wrong = []
        if u["is_entity"] in ("yes", "no"):
            d = u[f"d_{ENTITY_Q}"]
            if (u["is_entity"] == "yes") != (d == "yes") and d != "abstain":
                wrong.append(f"{ENTITY_Q}:{d}")
        for r in RELATION_QUESTIONS:
            truth = u["is_entity"] == "yes" and u[f"y_{r}"]
            p = u[f"p_{r}"]
            if (truth and p != "yes") or (not truth and p == "yes"):
                wrong.append(f"{r}:{p}")
        if wrong:
            rows.append(
                {"ord": u["ord"], "unit_id": u["unit_id"], "wrong": " ".join(wrong),
                 "label_entity": u["is_entity"], "label_relations": " ".join(u["relations"]),
                 "label_status": u["status"], "model_status": u[f"c_{STATUS_Q}"],
                 **{f"s_{q}": round(u[f"s_{q}"], 3) for q in YES_NO_QUESTIONS},
                 "note": u["note"], "text": u["span_text"]}
            )  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 보고서

QUESTION_LABEL = {
    ENTITY_Q: "회사 식별",
    "competitor": "경쟁",
    "business": "공급·협력",
    "equity": "지분",
}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _pct_ci(v: float | None, ci: list[float] | None) -> str:
    if v is None:
        return "-"
    return f"{_pct(v)} ({_pct(ci[0])}~{_pct(ci[1])})" if ci else _pct(v)


def _decision_row(name: str, m: dict) -> list:
    return [
        name,
        m["n_pos"],
        m["accepted"],
        _pct_ci(m["accepted_precision"], m["accepted_precision_ci"]),
        _pct_ci(m["accepted_recall"], m["accepted_recall_ci"]),
        f"{m['uncertain']} ({m['uncertain_pos']})",
        f"{m['rejected']} ({m['rejected_pos']})",
    ]


def format_report(metrics: dict, meta: dict) -> str:
    from tabulate import tabulate

    u = metrics["units"]
    lines = [
        f"# 판정 채점: 표본 {meta['sample']} · {meta['model']} · 질문 {meta['question_version']}",
        "",
        f"- 기준: {meta['labels']}",
        f"- 입력 방식 {meta.get('context', 'span')} · 채택 {meta['accept']} 이상, "
        f"기각 {meta['reject']} 미만, 그 사이는 불확실",
        f"- 검수 단위 {u['n']}개 중 라벨 {u['labeled']}개(보류 {u['skipped']}), 판정 {u['judged']}개, "
        f"채점 {u['scored']}개 (회사 식별 '모름' {u['entity_unsure']}개는 식별 채점에서 뺌)",
        "- 괄호 안 퍼센트는 Wilson 95% 구간. 불확실·기각 칸의 괄호는 그 안에 있던 정답 수",
        "",
        "## 질문별",
        "",
        "회사 식별은 '예/아니오' 라벨이 있는 단위, 나머지는 회사가 맞다고 라벨된 단위에서 잽니다.",
        "",
    ]
    head = ["질문", "정답", "AUC", "채택", "채택 정밀도", "채택 재현율", "불확실", "기각"]
    rows = []
    for q, m in metrics["questions"].items():
        if not m["n"]:
            continue
        r = _decision_row(QUESTION_LABEL[q], m)
        rows.append([r[0], f"{m['n_pos']}/{m['n']}", m["auc"] if m["auc"] is not None else "-",
                     *r[2:]])  # fmt: skip
    lines += [tabulate(rows, head, tablefmt="github"), ""]

    head = ["관계", "정답", "채택", "채택 정밀도", "채택 재현율", "불확실", "기각"]
    lines += [
        "## 단위별 종합 (회사 식별과 관계 질문을 함께 적용)",
        "",
        tabulate(
            [_decision_row(QUESTION_LABEL[r], m) for r, m in metrics["units_combined"].items()],
            head,
            tablefmt="github",
        ),  # fmt: skip
        "",
        f"## 쌍별 종합 ({metrics['pairs']['n_pairs']}쌍)",
        "",
        tabulate(
            [
                _decision_row(QUESTION_LABEL.get(k, "전체"), m)
                for k, m in metrics["pairs"].items()
                if isinstance(m, dict)
            ],
            head,
            tablefmt="github",
        ),  # fmt: skip
        "",
    ]
    m = metrics["status"]
    lines.append(f"- 시점 정확도: {_pct_ci(m['accuracy'], m['accuracy_ci'])} (n={m['n']})")
    lines += ["", "## 점수 보정 (관계 질문을 모아서)", ""]
    lines.append(
        tabulate(
            [
                [
                    c["bin"],
                    c["n"],
                    c["mean_score"] if c["mean_score"] is not None else "-",
                    _pct(c["observed"]),
                ]
                for c in metrics["calibration"]
            ],
            ["점수 구간", "개수", "평균 점수", "실제 정답 비율"],
            tablefmt="github",
        )  # fmt: skip
    )
    return "\n".join(lines) + "\n"

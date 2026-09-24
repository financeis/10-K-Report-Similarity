"""판정 채점: 표본 검수 라벨과 모델 판정 비교 (docs/relation-map-plan.md 7.6).

단위(근거 구간 × 언급된 회사)마다 라벨과 점수를 나란히 놓고 세 가지를 잰다.
- 질문별: 점수가 라벨을 얼마나 잘 가르는지(AUC), 임계값에서 채택·불확실·기각이 얼마나 맞는지
- 단위별 종합: 회사 식별과 관계 질문을 7.4의 규칙대로 합쳤을 때
- 쌍별 종합: 같은 쌍의 단위를 합친 관계 (화면에 선으로 그려질 것)
검수 라벨 v2는 판정과 같은 세 가지(경쟁 · 공급·협력 · 지분)다. v1 라벨(dev1)은 방향·세부
유형까지 적혀 있어 LABEL_GROUPS로 묶어 채점한다.
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
    "business": ("business", "doc_supplies_target", "target_supplies_doc", "partner"),
    "equity": ("equity", "doc_owns_target", "target_owns_doc"),
}
"""판정 관계 → 그에 해당하는 검수 라벨 코드 (v2 코드 + v1 코드)."""
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
    labels: pd.DataFrame,
    judgements: dict[str, Judgement],
    accept: float,
    reject: float,
    per_question: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """라벨 + 질문별 점수·결정. 점수 열은 s_<질문>, 결정 열은 d_<질문>.
    per_question: 질문별 (채택, 기각) 임계값. 없는 질문은 accept·reject."""
    per_question = per_question or {}
    t = labels.copy()
    t["judged"] = t["unit_id"].isin(judgements.keys())
    for q in YES_NO_QUESTIONS:
        a_q, r_q = per_question.get(q, (accept, reject))
        answers = [judgements[u].answers.get(q) if u in judgements else None for u in t["unit_id"]]
        t[f"s_{q}"] = [a.get("score") if a else np.nan for a in answers]
        t[f"d_{q}"] = [decide(a, a_q, r_q) if a else None for a in answers]
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
    out["sweep"] = {
        ENTITY_Q: threshold_sweep(
            entity_known["is_entity"] == "yes", entity_known[f"s_{ENTITY_Q}"]
        ),
        **{
            r: threshold_sweep(entity_yes[f"y_{r}"], entity_yes[f"s_{r}"])
            | relation_sweep(entity_known, r)
            for r in RELATION_QUESTIONS
        },
    }
    return out


ACCEPT_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
REJECT_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)


def threshold_sweep(
    y: pd.Series,
    s: pd.Series,
    min_precision: float = 0.95,
    max_missed: float = 0.02,
) -> dict:
    """임계값을 바꿔 가며 채택 정밀도·재현율과 기각 쪽에서 놓치는 정답을 센다.

    제안값(개발 표본에서만 참고):
    - 채택: 채택 정밀도가 min_precision 이상인 값 가운데 재현율이 가장 높은 값
    - 기각: 기각한 것 속 정답이 전체 정답의 max_missed 이하인 가장 높은 값
    """
    y, s = y.to_numpy(bool), s.to_numpy(float)
    pos = int(y.sum())
    accept = []
    for a in ACCEPT_GRID:
        m = s >= a
        k = int((m & y).sum())
        accept.append({"at": a, "n": int(m.sum()), "precision": _ratio(k, int(m.sum())),
                       "recall": _ratio(k, pos)})  # fmt: skip
    reject = []
    for r in REJECT_GRID:
        m = s < r
        missed = int((m & y).sum())
        reject.append({"at": r, "n": int(m.sum()), "missed": missed,
                       "missed_rate": _ratio(missed, pos)})  # fmt: skip
    ok_a = [x for x in accept if x["precision"] is not None and x["precision"] >= min_precision]
    # 재현율이 같으면 더 높은 값: 점수가 조금 흔들려도(같은 입력에 ±0.03) 정밀도를 지키게
    best_a = max(ok_a, key=lambda x: (x["recall"], x["at"]), default=None)
    ok_r = [x["at"] for x in reject if (x["missed_rate"] or 0) <= max_missed]
    return {
        "n": len(y),
        "n_pos": pos,
        "accept": accept,
        "reject": reject,
        "suggest_accept": best_a["at"] if best_a else None,
        "suggest_reject": max(ok_r) if ok_r else None,
    }


def relation_sweep(
    t: pd.DataFrame,
    relation: str,
    min_pair_precision: float = 0.95,
    min_unit_precision: float = 0.85,
) -> dict:
    """관계 질문의 채택 임계값별 결과를 근거 문장(단위)과 두 회사 관계(쌍) 기준으로 함께 센다.
    회사 식별은 지금 임계값의 결정을 그대로 쓴다.

    제안값: 쌍 정밀도 min_pair_precision 이상, 문장 정밀도 min_unit_precision 이상인 값 가운데
    쌍 재현율이 가장 높은 값 (같으면 문장 정밀도, 문장 재현율, 높은 임계값 순으로).
    화면의 선이 틀리는 것이 근거 문장 하나가 틀리는 것보다 나쁘므로 쌍 기준을 먼저 본다.
    """
    truth = ((t["is_entity"] == "yes") & t[f"y_{relation}"]).to_numpy(bool)
    entity_ok = (t[f"d_{ENTITY_Q}"] == "yes").to_numpy(bool)
    s = t[f"s_{relation}"].to_numpy(float)
    pairs = t["pair_key"].to_numpy()
    rows = []
    for a in ACCEPT_GRID:
        yes = entity_ok & (s >= a)
        k = int((yes & truth).sum())
        g = pd.DataFrame({"pair": pairs, "yes": yes, "truth": truth}).groupby("pair").any()
        kp = int((g["yes"] & g["truth"]).sum())
        rows.append(
            {"at": a, "units": int(yes.sum()), "unit_precision": _ratio(k, int(yes.sum())),
             "unit_recall": _ratio(k, int(truth.sum())), "pairs": int(g["yes"].sum()),
             "pair_precision": _ratio(kp, int(g["yes"].sum())),
             "pair_recall": _ratio(kp, int(g["truth"].sum()))}
        )  # fmt: skip
    ok = [
        x for x in rows
        if (x["pair_precision"] or 0) >= min_pair_precision
        and (x["unit_precision"] or 0) >= min_unit_precision
    ]  # fmt: skip
    best = max(
        ok,
        key=lambda x: (
            x["pair_recall"] or 0,
            x["unit_precision"] or 0,
            x["unit_recall"] or 0,
            x["at"],
        ),
        default=None,
    )
    return {"combined": rows, "suggest_accept": best["at"] if best else None}


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
        *(
            [
                f"- **주의: 확인 표본의 고정 설정과 다름({', '.join(meta['settings_changed'])}). "
                "이 결과는 합격 판단에 쓰지 않습니다.**"
            ]
            if meta.get("settings_changed")
            else ["- 확인 표본: 뽑을 때 고정한 판정 설정 그대로 채점"]
            if meta.get("frozen")
            else []
        ),  # fmt: skip
        f"- 입력 방식 {meta.get('context', 'span')} · 채택 {meta['accept']} 이상, "
        f"기각 {meta['reject']} 미만, 그 사이는 불확실"
        + "".join(
            f" · {QUESTION_LABEL[q]} {a}/{r}"
            for q, (a, r) in (meta.get("thresholds") or {}).items()
            if (a, r) != (meta["accept"], meta["reject"])
        ),
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
    if m["n"]:  # 시점은 v1 라벨(dev1)에만 있다
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
    if meta.get("purpose") == "confirm":
        # 확인 표본으로 임계값을 고르면 확인 표본이 개발 표본이 되어 버린다
        lines += ["", "(확인 표본이라 임계값별 표와 제안값은 싣지 않습니다.)"]
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "## 임계값별 (개발 표본에서 임계값을 고를 때만 봅니다)",
        "",
        "제안 채택값: 회사 식별은 정밀도 95% 이상에서 재현율이 가장 높은 값. 관계는 두 회사 관계(쌍) "
        "정밀도 95% 이상, 근거 문장 정밀도 85% 이상에서 쌍 재현율이 가장 높은 값.",
        "제안 기각값: 기각한 것 속에 놓친 정답이 2% 이하인 가장 높은 값.",
        "",
    ]
    for q, sw in metrics["sweep"].items():
        if "combined" in sw:
            cells = [
                f"{x['at']}: 문장 {x['units']}건 {_pct(x['unit_precision'])}/"
                f"{_pct(x['unit_recall'])}, 쌍 {x['pairs']}쌍 {_pct(x['pair_precision'])}/"
                f"{_pct(x['pair_recall'])}"
                for x in sw["combined"]
            ]
        else:
            cells = [
                f"{x['at']}: {x['n']}건 {_pct(x['precision'])}/{_pct(x['recall'])}"
                for x in sw["accept"]
            ]
        misses = [f"{x['at']}: 놓침 {x['missed']}" for x in sw["reject"]]
        lines += [
            f"- **{QUESTION_LABEL[q]}** (정답 {sw['n_pos']}/{sw['n']}) · 제안 채택 "
            f"{sw['suggest_accept'] or '-'}, 기각 {sw['suggest_reject'] or '-'}",
            *(f"  - 채택 {c} (정밀도/재현율)" for c in cells),
            f"  - 기각: {' · '.join(misses)}",
        ]
    return "\n".join(lines) + "\n"

"""10-K 본문의 회사 이름 언급 → 언급(mentions), 근거 구간(spans), 매출 비중 후보(figures).

모든 위치는 정제 텍스트(documents.parquet의 text) 기준이다. 문서마다 text_hash를 함께 남겨,
정제 규칙이 바뀌어 위치가 어긋났는지 나중에 알 수 있게 한다.

여기서는 '이름이 나왔다'만 기록한다. 그 이름이 정말 그 회사인지, 그 문장이 관계의 근거인지는
판정 단계(judge)가 정한다. 다만 관계와 무관한 게 확실한 문맥(임원 약력, 상장 거래소 표기 등)은
excluded에 이유를 적어 둔다. 지우지 않으므로 화면에서 확인할 수 있다.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import re
from dataclasses import dataclass

import pandas as pd

from .names import NameDictionary, company_node

log = logging.getLogger(__name__)

# 문장 경계: 마침표 등 뒤 공백 + 대문자/여는 괄호·따옴표, 또는 줄바꿈.
# 세미콜론으로는 나누지 않는다 ("such as A; B; and C" 같은 경쟁사 목록이 쪼개진다).
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"])|\n+")
_ABBREVIATIONS = {
    "Inc.", "Co.", "Corp.", "Ltd.", "Cos.", "Bros.", "Mr.", "Ms.", "Mrs.", "Dr.", "Jr.", "Sr.",
    "St.", "No.", "Nos.", "U.S.", "U.K.", "S.A.", "N.V.", "L.P.", "Inc.,", "vs.", "e.g.", "i.e.",
}  # fmt: skip
_INITIAL = re.compile(r"(?:[A-Z]\.)+")  # "J.P." "A." 같은 머리글자
_BULLETS = set("•●◦▪■-–*·\x95")
"""글머리표. \\x95와 \\uf0b7·\\uf0a7(Wingdings 글꼴의 글머리표가 문자로 남은 것)도 포함한다."""
_LETTER_BULLET = re.compile(r"[lo](?=[A-Z])")
"""Wingdings의 'l'·'o' 글머리표가 글자로 남은 줄 ('lPuget Sound Energy, Inc.')."""
_SENTENCE_END = set('.!?:;"”’)]')
"""줄 끝이 이 문자면 줄바꿈을 문장 경계로 본다."""
_FOOTNOTE = re.compile(r"^\(?\d{1,2}\)")
_ROW_MARKER = re.compile(r"(?:\(?\d{1,2}\)|\(?[a-z]\))\s")
"""줄 머리의 항목 표시 ('(1) ', 'b) ') — 이 앞의 줄바꿈은 잇지 않는다."""
_OPEN_END = re.compile(
    r"(?:,|\b(?:the|of|and|or|in|from|to|by|with|as|such|including|upon|than|between|for|at|on|"
    r"a|an|our|its|their))$"
)
"""문장이 끝나지 않은 줄 끝 (쉼표, 관사·전치사·접속사)."""
_WRAP_MIN_LINE = 60
"""이보다 긴 줄이 _OPEN_END로 끝날 때만 대문자로 시작하는 다음 줄과 잇는다 (표 행을 붙이지 않도록)."""
_LEAD_IN_MAX_DISTANCE = 6000
_PRONOUN_START = re.compile(r"^(?:It|They|These|This|Such|Its|Their|Both|Each of them)\b")

# 임원 약력 소제목. Item 1 끝에 붙는 경우가 많다. ("See Item 10 ..." 같은 참조 문장은 제외)
_EXEC_HEADING = re.compile(
    r"(?im)^(?!see\b)(?:supplementary item\.?\s*)?(?:information (?:about|regarding|concerning) )?"
    r"(?:the |our )?(?:company['’]s |registrant['’]s |[A-Z][\w&.]*['’]s )?"
    r"(?:current )?executive officers\b(?: of (?:the registrant|[A-Z][^\n]{0,60}))?[^\n]{0,250}$"
)
# 약력 줄: 나이, 직함, 경력 표현. 이런 줄이 이어지는 동안을 약력 부분으로 본다.
_BIO_LINE = re.compile(
    r"\(\d{2}\)|,\s*\d{2}[,.]|\bage[ds]?\b|\b(?:served|serves|serving|joined|appointed|named|elected|"
    r"president|chief|officer|director|chair|executive|treasurer|controller|secretary|counsel|"
    r"prior to|previously|career|position|positions)\b",
    re.I,
)
_EXEC_MAX_CHARS = 20_000

# 익명 고객·유통사 표기 ("Customer A", "Direct Customer B")
_ANONYMOUS = re.compile(r"\b((?:Direct |Indirect )?(?:Customer|Distributor|Partner) [A-H])\b")

# 매출 비중: "11.8% of our consolidated revenues", "10% or more of total net sales"
_FIGURE = re.compile(
    r"(?P<pre>approximately|about|nearly|over|more than|less than|at least)?\s*"
    r"(?P<value>\d{1,2}(?:\.\d+)?)\s?%\s*(?P<post>or more|or greater|or less)?\s*of\s+"
    r"(?:our |its |the company['’]s |the |total )*"
    r"(?P<den>(?:consolidated |total |net |worldwide |annual |segment )*"
    r"(?:revenues?|net sales|sales|net revenues?))",
    re.I,
)
_YEAR = re.compile(r"\b((?:19|20)\d\d)\b")
_PERCENT = re.compile(r"\d\s?%")
_COMBINED = re.compile(r"\b(?:together|combined|collectively|in the aggregate|aggregate)\b", re.I)
_EACH = re.compile(r"\beach\b", re.I)


SPAN_RULES = 3
"""근거 구간 규칙의 판 (3-2). 2: 문장 중간 줄바꿈(소문자로 이어지는 줄)을 잇고, 목록이 끝난 뒤 문장에는
도입문을 붙이지 않음. 3: 쉼표·전치사로 끝나는 긴 줄도 잇고, 항목 안의 둘째 문장에는 도입문을 붙이며,
Wingdings 글머리표를 알아봄."""


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _bullet_at(text: str, i: int) -> bool:
    """text[i]에서 글머리표로 시작하는가 (Wingdings 'l'·'o' 글머리표 포함)."""
    return i < len(text) and (text[i] in _BULLETS or bool(_LETTER_BULLET.match(text, i)))


def _wrapped(text: str, brk: int, nxt: int) -> bool:
    """text[brk:nxt]의 줄바꿈 하나가 문장 중간의 줄 맞춤인가."""
    before = text[brk - 1] if brk > 0 else "."
    if before in _SENTENCE_END or nxt >= len(text):
        return False
    if _bullet_at(text, nxt) or _ROW_MARKER.match(text, nxt):
        return False
    if text[nxt].islower():
        return True
    line_start = text.rfind("\n", 0, brk) + 1
    line = text[line_start:brk]
    return (
        len(line) > _WRAP_MIN_LINE
        and not _bullet_at(text, line_start)
        and bool(_OPEN_END.search(line))
    )


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """문장별 (시작, 끝). 앞뒤 공백은 뺀다. 'Inc.', 'Co.' 같은 약어 뒤에서는 나누지 않는다.

    줄바꿈은 문장 경계로 보되, 문장 중간에 들어간 줄바꿈(원문의 줄 맞춤, 지운 표·쪽번호 줄)은 잇는다.
    - 앞 줄이 문장 부호 없이 끝나고 다음 줄이 소문자로 시작할 때
    - 앞 줄이 길고(60자 넘음) 쉼표나 관사·전치사로 끝날 때 ("merchant silicon vendor,\\nBroadcom ...")
    다음 줄이 글머리표나 항목 표시('(1) ', 'b) ')로 시작하면 잇지 않는다.
    """
    out = []
    pos = 0
    for m in _SENTENCE_BREAK.finditer(text):
        if "\n" not in m.group(0):
            word = text[max(pos, m.start() - 12) : m.start()].rsplit(None, 1)[-1:]
            if word and (word[0] in _ABBREVIATIONS or _INITIAL.fullmatch(word[0])):
                continue
        elif m.group(0).count("\n") == 1 and _wrapped(text, m.start(), m.end()):
            continue
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    trimmed = []
    for s, e in out:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    return trimmed


def exec_bio_ranges(text: str) -> list[tuple[int, int]]:
    """임원 약력 부분의 (시작, 끝).

    소제목부터 약력처럼 보이는 줄(나이, 직함, 경력 표현)이 이어지는 동안이다. 이름만 있는
    짧은 줄은 다음 줄이 약력이면 이어지는 것으로 본다. 약력이 아닌 문단을 만나면 끝난다.
    """
    ranges = []
    for m in _EXEC_HEADING.finditer(text):
        pos = end = m.end()
        lines = []
        while pos < len(text) and pos - m.end() < _EXEC_MAX_CHARS:
            nl = text.find("\n", pos)
            nl = len(text) if nl < 0 else nl
            lines.append((pos, nl))
            pos = nl + 1
        for k, (a, b) in enumerate(lines):
            line = text[a:b].strip()
            if not line or _BIO_LINE.search(line):
                end = b
                continue
            nxt = next((text[c:d] for c, d in lines[k + 1 :] if text[c:d].strip()), "")
            if len(line) <= 80 and _BIO_LINE.search(nxt):
                end = b
                continue
            break
        ranges.append((m.start(), end))
    return ranges


def main_start(text: str, sents: list[tuple[int, int]], i: int, max_chars: int) -> int:
    """문장 i가 대명사로 시작하면("They also ...") 앞 문장까지 근거 본문에 넣는다."""
    s, e = sents[i]
    if i > 0 and _PRONOUN_START.match(text[s:e]) and e - sents[i - 1][0] <= max_chars:
        return sents[i - 1][0]
    return s


def _is_item(text: str, s: int, e: int) -> bool:
    """목록 항목처럼 보이는가: 글머리표로 시작하거나, 짧고 마침표로 끝나지 않는 줄."""
    return _bullet_at(text, s) or (e - s <= 200 and text[e - 1] != ".")


def _in_bullet_line(text: str, s: int) -> bool:
    """s가 글머리표로 시작하는 줄의 중간인가 (항목 안의 둘째 문장부터)."""
    line_start = text.rfind("\n", 0, s) + 1
    return line_start < s and _bullet_at(text, line_start)


def lead_in(text: str, sents: list[tuple[int, int]], i: int) -> tuple[int, int] | None:
    """문장 i가 목록 항목이면 그 목록의 도입문("Our competitors include:") 위치.

    목록 항목을 거슬러 올라가다 콜론으로 끝나는 문장을 만나면 그것이 도입문이다. 도입문은 본문과
    떨어져 있을 수 있어 근거 구간에 따로 저장한다. 문장 i가 목록 항목이 아닌 일반 문장이고 도입문과
    문장 i 사이에 글머리표 항목이 있으면, 목록이 이미 끝난 것이라 도입문을 붙이지 않는다. 한 항목 안의
    둘째 문장부터, 표가 풀린 행, 각주("(1) ...")는 목록의 일부라 도입문을 그대로 붙인다.
    """
    s, e = sents[i]
    j = i - 1
    while j >= 0 and s - sents[j][0] <= _LEAD_IN_MAX_DISTANCE:
        ps, pe = sents[j]
        if text[pe - 1] == ":":
            prose = not (
                _is_item(text, s, e) or _in_bullet_line(text, s) or _FOOTNOTE.match(text[s:e])
            )
            if prose and any(_bullet_at(text, sents[k][0]) for k in range(j + 1, i)):
                return None
            return ps, pe
        if not _is_item(text, ps, pe):
            return None
        j -= 1
    return None


@dataclass
class MentionTables:
    mentions: pd.DataFrame
    spans: pd.DataFrame
    figures: pd.DataFrame
    nodes: pd.DataFrame


def find_mentions(
    docs: pd.DataFrame,
    dictionary: NameDictionary,
    *,
    include_lead_in: bool = True,
    max_span_chars: int = 1500,
) -> MentionTables:
    """docs: documents.parquet 형식 (cik, section, accession_number, period_of_report, text ...)."""
    mentions: list[dict] = []
    anon_nodes: dict[str, str] = {}
    span_rows: dict[str, dict] = {}
    for doc in docs.itertuples():
        text = doc.text
        if not text:
            continue
        doc_node = company_node(doc.cik)
        accession = doc.accession_number
        year = _year(doc.period_of_report)
        sents = sentence_bounds(text)
        starts = [s for s, _ in sents]
        bios = exec_bio_ranges(text) if doc.section == "business" else []
        thash = text_hash(text)

        hits = []  # (start, end, matched, node_id, kind)
        for m in dictionary.regex.finditer(text):
            entry = dictionary.lookup(m.group(1), year)
            if entry is not None and entry.node_id != doc_node:
                hits.append((m.start(1), m.end(1), m.group(1), entry.node_id, entry.kind))
        for m in _ANONYMOUS.finditer(text):
            label = m.group(1)
            node = f"anon:{doc.cik}:{accession}:{_slug(label)}"
            anon_nodes[node] = label
            hits.append((m.start(1), m.end(1), label, node, "anonymous"))

        for start, end, matched, node, kind in sorted(hits):
            i = bisect.bisect_right(starts, start) - 1
            ss, se = sents[i]
            span_start = main_start(text, sents, i, max_span_chars) if include_lead_in else ss
            span_end = se
            # 표가 풀려 문장 경계가 없는 긴 덩어리는 언급 주변만 자른다
            if span_end - span_start > max_span_chars:
                span_start = max(ss, start - max_span_chars // 2)
                span_end = min(se, end + max_span_chars // 2)
            span_id = f"s:{accession}:{doc.section}:{span_start}-{span_end}"
            if span_id not in span_rows:
                lead = lead_in(text, sents, i) if include_lead_in else None
                if lead and lead[1] - lead[0] > max_span_chars:
                    lead = None
                span_rows[span_id] = {
                    "span_id": span_id,
                    "doc_node": doc_node,
                    "accession": accession,
                    "section": doc.section,
                    "text_hash": thash,
                    "char_start": span_start,
                    "char_end": span_end,
                    "lead_start": lead[0] if lead else None,
                    "lead_end": lead[1] if lead else None,
                    "text": text[span_start:span_end],
                    "lead_text": text[lead[0] : lead[1]] if lead else None,
                }
            excluded = None
            if any(a <= start < b for a, b in bios):
                excluded = "exec_bio"
            else:
                # 도입문도 함께 본다: "Ratings: … Moody's (3)"처럼 문맥이 도입문에만 있을 때
                row = span_rows[span_id]
                context = f"{row['lead_text'] or ''}\n{row['text']}"
                reason = dictionary.excluded_by(node, context, text, (start, end))
                if reason:
                    excluded = f"context:{reason}"
            mentions.append(
                {
                    "mention_id": f"m:{accession}:{doc.section}:{start}",
                    "doc_node": doc_node,
                    "target_node": node,
                    "matched_name": matched,
                    "name_kind": kind,
                    "accession": accession,
                    "section": doc.section,
                    "char_start": start,
                    "char_end": end,
                    "span_id": span_id,
                    "excluded": excluded,
                }
            )

    mentions_df = pd.DataFrame(mentions, columns=_MENTION_COLUMNS)
    spans_df = pd.DataFrame(list(span_rows.values()), columns=_SPAN_COLUMNS)
    figures_df = _figure_rows(mentions_df, spans_df)
    nodes = pd.concat(
        [
            dictionary.nodes,
            pd.DataFrame(
                {
                    "node_id": list(anon_nodes),
                    "kind": "anonymous",
                    "cik": pd.array([None] * len(anon_nodes), dtype="Int64"),
                    "ticker": None,
                    "name": list(anon_nodes.values()),
                }
            ),
        ],
        ignore_index=True,
    )
    return MentionTables(mentions_df, spans_df, figures_df, nodes)


_MENTION_COLUMNS = [
    "mention_id", "doc_node", "target_node", "matched_name", "name_kind", "accession",
    "section", "char_start", "char_end", "span_id", "excluded",
]  # fmt: skip
_SPAN_COLUMNS = [
    "span_id", "doc_node", "accession", "section", "text_hash", "char_start", "char_end",
    "lead_start", "lead_end", "text", "lead_text",
]  # fmt: skip
_FIGURE_COLUMNS = [
    "figure_id", "span_id", "target_node", "subject", "value", "operator", "period",
    "denominator", "raw",
]  # fmt: skip


def parse_figures(span_text: str, n_targets: int) -> list[dict]:
    """구간 하나의 매출 비중 후보. 귀속이 불분명하면 value를 비운다(원문은 raw에 남긴다)."""
    found = list(_FIGURE.finditer(span_text))
    if not found:
        return []
    years = sorted(set(_YEAR.findall(span_text)))
    if _COMBINED.search(span_text) and n_targets > 1:
        subject = "combined"
    elif n_targets > 1:
        subject = "each" if _EACH.search(span_text) else "unclear"
    else:
        subject = "single"
    # 한 문장에 수치나 연도가 여러 개면(연도별 나열 등) 어느 회사·연도의 값인지 코드로 정하지 않는다
    n_percent = len(_PERCENT.findall(span_text))
    clear = subject in ("single", "each") and n_percent == 1 and len(years) <= 1
    out = []
    for m in found:
        pre, post = (m.group("pre") or "").lower(), (m.group("post") or "").lower()
        if post in ("or more", "or greater") or pre in ("more than", "over", "at least"):
            op = ">="
        elif post == "or less" or pre == "less than":
            op = "<"
        elif pre in ("approximately", "about", "nearly"):
            op = "~"
        else:
            op = "="
        out.append(
            {
                "subject": subject if clear else "unclear",
                "value": float(m.group("value")) if clear else None,
                "operator": op if clear else None,
                "period": years[0] if clear and years else None,
                "denominator": m.group("den").lower() if clear else None,
                "raw": m.group(0).strip(),
            }
        )
    return out


def _figure_rows(mentions: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    live = mentions[mentions["excluded"].isna()]
    rows = []
    targets_by_span = live.groupby("span_id")["target_node"].unique()
    text_by_span = dict(zip(spans["span_id"], spans["text"], strict=True))
    for span_id, targets in targets_by_span.items():
        for n, fig in enumerate(parse_figures(text_by_span[span_id], len(targets))):
            for target in targets:
                rows.append(
                    {
                        "figure_id": f"f:{span_id}:{n}:{target}",
                        "span_id": span_id,
                        "target_node": target,
                        **fig,
                    }  # fmt: skip
                )
    return pd.DataFrame(rows, columns=_FIGURE_COLUMNS)


def _year(period: str | None) -> int | None:
    try:
        return int(str(period)[:4])
    except (TypeError, ValueError):
        return None


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")

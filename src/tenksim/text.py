"""섹션 원문 정제, 추출 품질 판정, 청킹.

edgartools가 준 텍스트는 문단이 줄바꿈으로 나뉘어 있고 HTML 엔티티도 풀려 있다.
남는 노이즈는 페이지 번호·머리글/꼬리말, 한 줄로 뭉개진 표, 섹션 제목, 그리고
경계를 놓쳐 뒤에 붙어 온 다음 항목이다. 대소문자와 구두점은 그대로 둔다.
요즘 임베딩 모델은 자연문으로 학습돼서 소문자화나 구두점 제거가 이득이 없고 정보만 잃는다.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

_TERMINAL_PUNCT = re.compile(r"[.!?:;\"”’)\]]$")
_NUMBER = re.compile(r"\d[\d,.]*")
_PAGE_NUMBER = re.compile(r"(page\s*)?[-–—]?\s*([a-z]{1,2}-)?\d{1,3}\s*[-–—]?", re.I)
_RUNNING_HEADER = re.compile(
    r"(tables?\s+of\s+contents?|index)(\s+to\s+financial\s+statements)?"
    r"(\s+index\s+to\s+financial\s+statements)?|part\s+[ivx]+\.?",
    re.I,
)
_FORM_FOOTER = re.compile(r"form\s+10-k", re.I)
_NOTE_HEADING = re.compile(r"note\s+\d+\b", re.I)
# 'Item 1.', 'ITEM I.', 'Item 1(a)', 'Item 1.A.', 'PART IItem 1A', 'Items 1 and 2' 같은 항목 제목
_ITEM_HEADING = re.compile(
    r"(?:part\s*[ivx]+\.?\s*)?items?\s*(\d{1,2}|[ivx]{1,4})"
    r"(?:\.?\s*\(\s*([a-c])\s*\)|\.?([a-c]))?(?![a-z0-9])",
    re.I,
)
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8"}
_EXPECTED_ITEM = {"business": "1", "risk_factors": "1a"}
_NAME_HEADINGS = {
    "business": re.compile(r"((our\s+)?business|description\s+of\s+business)\b", re.I),
    "risk_factors": re.compile(r"(risk\s*factors|risks\b)", re.I),
}
_FINANCIAL_STATEMENT = re.compile(
    r"(consolidated\s+)?(statements?\s+of\s+(income|operations|earnings|cash\s+flows|"
    r"comprehensive|financial\s+position|changes)|balance\s+sheets?)",
    re.I,
)
_TITLE_WORD = re.compile(r"[a-z]{4,}", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    text = html.unescape(text)
    # NFKC: 줄바꿈 없는 공백(\xa0)이나 전각 문자를 일반 문자로 바꾼다
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_page_marker(line: str) -> bool:
    """'12', 'K-3', 'Apple Inc. | 2024 Form 10-K | 5', 'Table of Contents' 같은 줄."""
    if _PAGE_NUMBER.fullmatch(line) or _RUNNING_HEADER.fullmatch(line):
        return True
    footer = len(line) < 120 and bool(_FORM_FOOTER.search(line))
    return footer and (line[-1].isdigit() or "|" in line)


def item_token(line: str) -> str | None:
    """항목 제목 줄이면 항목 번호('1', '1a', '7' 등)를, 아니면 None을 돌려준다."""
    m = _ITEM_HEADING.match(line)
    if not m or len(line) >= 100:  # 긴 줄은 제목이 아니라 본문 문장이다
        return None
    number = _ROMAN.get(m.group(1).lower(), m.group(1))
    return (number + (m.group(2) or m.group(3) or "")).lower()


def _has_title(line: str) -> bool:
    """'Item 1A. Risk Factors'처럼 항목 번호 뒤에 제목 단어가 붙어 있는지."""
    m = _ITEM_HEADING.match(line)
    return bool(m and _TITLE_WORD.search(line, m.end()))


def is_table_row(line: str) -> bool:
    """문장부호로 끝나지 않으면서 숫자 비중이 큰 줄. 예: 'North America186,751CCB141,640'."""
    if _TERMINAL_PUNCT.search(line):
        return False
    nonspace = sum(not c.isspace() for c in line)
    if nonspace == 0:
        return False
    digit_ratio = sum(c.isdigit() for c in line) / nonspace
    return (len(_NUMBER.findall(line)) >= 2 and digit_ratio >= 0.08) or digit_ratio >= 0.4


@dataclass(frozen=True)
class CleanedSection:
    text: str
    first_line: str
    """정제 후 첫 줄 (품질 리포트에서 눈으로 확인하는 용도)."""
    heading_ok: bool
    """앞 세 줄 안에서 이 섹션의 제목을 찾았는지. 못 찾아도 본문이 맞는 경우가 많아 참고용이다."""
    wrong_item: str | None
    """앞 세 줄에서 다른 항목 제목(예: Item 1C)이 먼저 나오면 그 번호."""
    starts_with_note: bool
    """'NOTE 2. ...'로 시작하면 재무제표 주석을 잘못 잘라 온 것이다."""
    starts_with_statement: bool
    """'Consolidated Statements of Income'처럼 재무제표로 시작하는 경우."""
    toc_like: bool
    """'Item N'으로 시작하는 짧은 줄이 5종류 이상이면 목차·상호참조 색인이다."""
    cut_at: str | None
    """본문 중간에 다음 항목 제목이 나와 거기서 잘랐다면 그 항목 번호."""
    n_chars_raw: int
    n_page_lines: int
    n_table_lines: int
    table_char_share: float


def clean_section(
    raw: str,
    section: str,
    *,
    drop_page_markers: bool = True,
    drop_table_rows: bool = True,
) -> CleanedSection:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in normalize(raw).split("\n")]
    lines = [ln for ln in lines if ln]
    total_chars = sum(map(len, lines))
    # 쪽마다 반복되는 짧은 줄(회사명 머리글 등)도 머리글로 본다
    counts = Counter(ln for ln in lines if len(ln) < 80)

    kept: list[str] = []
    n_page = n_table = table_chars = 0
    for line in lines:
        if is_page_marker(line) or counts.get(line, 0) >= 4:
            n_page += 1
            if drop_page_markers:
                continue
        elif is_table_row(line):
            n_table += 1
            table_chars += len(line)
            if drop_table_rows:
                continue
        kept.append(line)

    first = kept[0] if kept else ""
    expected = _EXPECTED_ITEM[section]
    heading_at, wrong_item = None, None
    for i, line in enumerate(kept[:3]):
        token = item_token(line)
        if token is not None:
            if token == expected:
                heading_at = i
            else:
                wrong_item = token
            break
        if len(line) < 100 and _NAME_HEADINGS[section].match(line):
            heading_at = i
            break
    # 모든 회사에 똑같이 들어가는 'Item 1. Business' 같은 제목 줄(과 그 앞 머리글)은 뺀다
    if heading_at is not None:
        kept = kept[heading_at + 1 :]
    # 다음 항목 제목이 나오면 거기서 자른다. edgartools가 경계를 놓쳐 Item 1 뒤에
    # Item 1A 전체를 붙여 오는 경우가 있다 (예: Bloom Energy, Rollins).
    cut_at = None
    if wrong_item is None:
        for i, line in enumerate(kept):
            token = item_token(line)
            # 'Item 8'이나 'Item 8 of'처럼 문장이 줄바꿈으로 끊긴 조각은 제목이 아니다.
            # 제목이면 뒤에 'Risk', 'Properties' 같은 단어가 붙는다.
            if token is not None and token != expected and _has_title(line):
                cut_at, kept = token, kept[:i]
                break
    # 색인 표의 줄은 쪽 번호 때문에 표로 지워지기도 하므로 정제 전 줄로 센다
    item_lines = {t for ln in lines if (t := item_token(ln)) is not None}
    return CleanedSection(
        text="\n".join(kept),
        first_line=first[:200],
        heading_ok=heading_at is not None,
        wrong_item=wrong_item,
        starts_with_note=bool(_NOTE_HEADING.match(first)),
        starts_with_statement=bool(_FINANCIAL_STATEMENT.match(first)),
        toc_like=len(item_lines) >= 5,
        cut_at=cut_at,
        n_chars_raw=total_chars,
        n_page_lines=n_page,
        n_table_lines=n_table,
        table_char_share=table_chars / total_chars if total_chars else 0.0,
    )


def duplicate_share(text: str, other: str, min_len: int = 40) -> float:
    """text의 (충분히 긴) 줄 중 other에도 그대로 있는 줄의 비율.

    edgartools가 다른 섹션의 일부를 이 섹션으로 잘라 오는 경우를 잡는다.
    예: Devon Energy의 Item 1 자리에 Item 1A의 일부가 들어온 사례.
    """
    lines = {ln for ln in text.split("\n") if len(ln) >= min_len}
    if not lines:
        return 0.0
    other_lines = {ln for ln in other.split("\n") if len(ln) >= min_len}
    return len(lines & other_lines) / len(lines)


def assess(
    fetch_status: str,
    cleaned: CleanedSection | None,
    min_chars: int,
    duplicate_of: str | None = None,
) -> tuple[str, str | None]:
    """분석에 쓸 수 있는지 판정한다. (status, 사유 코드)를 돌려준다.

    status: ok / suspect / too_short / missing / no_filing / error
    """
    if fetch_status != "fetched" or cleaned is None:
        return fetch_status, None
    if cleaned.starts_with_note:
        return "suspect", "note"
    if cleaned.wrong_item:
        return "suspect", f"wrong_item:{cleaned.wrong_item}"
    if cleaned.starts_with_statement:
        return "suspect", "financial_statement"
    if cleaned.toc_like:
        return "suspect", "toc"
    if duplicate_of:
        return "suspect", f"duplicate:{duplicate_of}"
    if len(cleaned.text) < min_chars:
        return "too_short", f"chars:{len(cleaned.text)}"
    notes = [] if cleaned.heading_ok else ["no_heading"]
    if cleaned.cut_at:
        notes.append(f"cut:{cleaned.cut_at}")
    return "ok", ";".join(notes) or None


def chunk_text(
    text: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[tuple[str, int]]:
    """문단 경계를 살려 max_tokens 이하 청크로 묶는다. (청크, 토큰 수) 목록을 돌려준다.

    긴 문단은 문장 단위로, 그래도 긴 문장은 단어 단위로 자른다.
    잘라낸 조각이 항상 1개 이상의 단어를 포함하므로 반드시 끝난다.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    pieces: list[tuple[str, int]] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        n = count_tokens(para)
        if n <= max_tokens:
            pieces.append((para, n))
            continue
        for sentence in _SENTENCE_END.split(para):
            n = count_tokens(sentence)
            if n <= max_tokens:
                pieces.append((sentence, n))
            else:
                pieces.extend(_split_words(sentence, max_tokens, count_tokens))

    chunks: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_tokens = 0
    for piece, n in pieces:
        # 조각 사이 구분자(줄바꿈)에 토큰 1개를 여유로 둔다
        if buf and buf_tokens + 1 + n > max_tokens:
            chunks.append(("\n".join(buf), buf_tokens))
            buf, buf_tokens = [], 0
        buf_tokens += n + (1 if buf else 0)
        buf.append(piece)
    if buf:
        chunks.append(("\n".join(buf), buf_tokens))
    return chunks


def _split_words(
    sentence: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[tuple[str, int]]:
    words = sentence.split()
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(words):
        # words[i:j]가 한도 안에 드는 가장 큰 j를 이분 탐색으로 찾는다
        lo, hi = i + 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(" ".join(words[i:mid])) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        piece = " ".join(words[i:lo])
        n = count_tokens(piece)
        if n > max_tokens:
            # 단어 하나가 한도를 넘는 극단적인 경우: 글자 단위로 자른다
            for s in range(0, len(piece), max_tokens):
                segment = piece[s : s + max_tokens]
                out.append((segment, count_tokens(segment)))
        else:
            out.append((piece, n))
        i = lo
    return out

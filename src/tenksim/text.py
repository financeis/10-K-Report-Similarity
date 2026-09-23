"""섹션 원문 정제, 추출 품질 판정, 청킹.

edgartools가 준 텍스트는 문단이 줄바꿈으로 나뉘어 있고 HTML 엔티티도 풀려 있다.
남는 노이즈는 페이지 번호·머리글/꼬리말, 한 줄로 뭉개진 표, 섹션 제목이다.
대소문자와 구두점은 그대로 둔다. 요즘 임베딩 모델은 자연문으로 학습돼서
소문자화나 구두점 제거가 이득이 없고 정보만 잃는다.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

_TERMINAL_PUNCT = re.compile(r"[.!?:;\"”’)\]]$")
_NUMBER = re.compile(r"\d[\d,.]*")
_PAGE_NUMBER = re.compile(r"(page\s*)?[-–—]?\s*([a-z]{1,2}-)?\d{1,3}\s*[-–—]?", re.I)
_RUNNING_HEADER = re.compile(r"(table of contents|index|part\s+[ivx]+\.?)", re.I)
_FORM_FOOTER = re.compile(r"form\s+10-k", re.I)
_NOTE_HEADING = re.compile(r"note\s+\d+\b", re.I)
_SECTION_HEADINGS = {
    "business": re.compile(r"(items?\s*1\b|business\b)", re.I),
    "risk_factors": re.compile(r"(item\s*1a\b|risk\s+factors\b)", re.I),
}
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
    return len(line) < 120 and bool(_FORM_FOOTER.search(line)) and line[-1].isdigit()


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
    """제목 판정에 쓴 첫 줄 (품질 리포트에서 눈으로 확인하는 용도)."""
    heading_ok: bool
    starts_with_note: bool
    """'NOTE 2. ...'로 시작하면 재무제표 주석을 잘못 잘라 온 것일 가능성이 크다."""
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

    kept: list[str] = []
    n_page = n_table = table_chars = 0
    for line in lines:
        if is_page_marker(line):
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
    heading_ok = bool(_SECTION_HEADINGS[section].match(first))
    # 모든 회사에 똑같이 들어가는 'Item 1. Business' 같은 제목 줄은 뺀다
    if heading_ok and len(first) < 100:
        kept = kept[1:]
    return CleanedSection(
        text="\n".join(kept),
        first_line=first[:200],
        heading_ok=heading_ok,
        starts_with_note=bool(_NOTE_HEADING.match(first)),
        n_chars_raw=total_chars,
        n_page_lines=n_page,
        n_table_lines=n_table,
        table_char_share=table_chars / total_chars if total_chars else 0.0,
    )


def assess(fetch_status: str, cleaned: CleanedSection | None, min_chars: int) -> str:
    """분석에 쓸 수 있는지 판정한다: ok / suspect / too_short / missing / no_filing / error."""
    if fetch_status != "fetched" or cleaned is None:
        return fetch_status
    if cleaned.starts_with_note or not cleaned.heading_ok:
        return "suspect"
    if len(cleaned.text) < min_chars:
        return "too_short"
    return "ok"


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

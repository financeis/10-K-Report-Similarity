"""회사명 사전: 기업 목록의 회사명 + 별칭 파일(configs/aliases.yaml) → 이름 하나로 본문을 훑는 정규식.

노드 ID
- 분석 대상 회사와 SEC에 공시하는 외부 회사: cik:<번호>
- SEC에 없는 외부 회사: ext:<이름>
- 익명 공시(Customer A 등): anon:<공시 회사 cik>:<accession>:<표기> (mentions.py에서 만든다)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

NameKind = Literal["base", "alias", "product"]

_SUFFIX = re.compile(
    r"(?:,?\s+(?:Inc\.?|Incorporated|Corporation|Corp\.?|Company|Co\.?|Companies|Holdings?|Group|"
    r"plc|PLC|Ltd\.?|Limited|N\.V\.|S\.A\.|L\.P\.|Worldwide|Global|Technologies|Technology|"
    r"Enterprises|Brands|Industries|Systems|Solutions|& Co\.?))+$"
)
_PARENS = re.compile(r"\s*\(.*?\)")


# ---------------------------------------------------------------- 별칭 파일


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductAlias(_Strict):
    name: str
    since: int | None = None
    """이 회계연도부터 모회사 소속 (비우면 제한 없음)."""
    until: int | None = None


class CompanyAliases(_Strict):
    names: list[str] = Field(default_factory=list)
    products: list[ProductAlias] = Field(default_factory=list)
    since: int | None = None
    """독립 회사가 된 첫 회계연도. 그 전에는 이 이름이 모회사의 사업부를 가리키므로
    (예: 분사 전 GE 10-K의 'GE Vernova') 이 회사의 모든 이름을 쓰지 않는다."""
    until: int | None = None


class ExternalCompany(_Strict):
    id: str
    name: str
    names: list[str]


class ExcludeContext(_Strict):
    reason: str
    tickers: list[str] = Field(default_factory=list)
    """이 회사를 가리키는 언급에만 적용. 비우면 모든 언급."""
    pattern: str


class AliasFile(_Strict):
    companies: dict[str, CompanyAliases] = Field(default_factory=dict)
    external: list[ExternalCompany] = Field(default_factory=list)
    generic_words: list[str] = Field(default_factory=list)
    exclude_contexts: list[ExcludeContext] = Field(default_factory=list)


def load_aliases(path: str | Path) -> AliasFile:
    with open(path, encoding="utf-8") as f:
        return AliasFile.model_validate(yaml.safe_load(f) or {})


# ---------------------------------------------------------------- 사전


def company_node(cik: int) -> str:
    return f"cik:{int(cik)}"


def strip_name(name: str) -> str:
    """'Apple Inc.' → 'Apple', 'Coca-Cola Company (The)' → 'Coca-Cola'."""
    name = _PARENS.sub("", name).strip()
    prev = None
    while prev != name:
        prev, name = name, _SUFFIX.sub("", name).strip().rstrip(",")
    return name


@dataclass(frozen=True)
class NameEntry:
    text: str
    node_id: str
    kind: NameKind
    since: int | None = None
    until: int | None = None

    def valid_in(self, year: int | None) -> bool:
        if year is None:
            return True
        return (self.since is None or year >= self.since) and (
            self.until is None or year <= self.until
        )


@dataclass
class ContextRule:
    reason: str
    nodes: frozenset[str]
    """비어 있으면 모든 언급에 적용."""
    pattern: re.Pattern

    def applies(self, node_id: str, text: str) -> bool:
        return (not self.nodes or node_id in self.nodes) and bool(self.pattern.search(text))


class NameDictionary:
    """이름 → 노드. 본문 검색용 정규식 하나와 문맥 제외 규칙을 함께 가진다."""

    def __init__(
        self, entries: list[NameEntry], nodes: pd.DataFrame, rules: list[ContextRule]
    ) -> None:
        self.entries: dict[str, list[NameEntry]] = {}
        for e in entries:
            self.entries.setdefault(e.text, []).append(e)
        self.nodes = nodes
        """node_id, kind(company/external), cik, ticker, name"""
        self.rules = rules
        self.regex = re.compile(r"(?<![\w&-])(" + trie_regex(self.entries) + r")(?![\w&-])")

    def lookup(self, text: str, year: int | None = None) -> NameEntry | None:
        for e in self.entries.get(text, []):
            if e.valid_in(year):
                return e
        return None

    def excluded_by(self, node_id: str, text: str) -> str | None:
        for rule in self.rules:
            if rule.applies(node_id, text):
                return rule.reason
        return None


def build_dictionary(universe: pd.DataFrame, aliases: AliasFile) -> NameDictionary:
    """universe: cik, ticker, name. 분석 대상이 아닌 회사도 '언급될' 수는 있으므로 전부 넣는다."""
    universe = universe.drop_duplicates("cik")
    ticker_node = {
        t: company_node(c) for t, c in zip(universe["ticker"], universe["cik"], strict=True)
    }
    generic = set(aliases.generic_words)

    unknown = sorted(set(aliases.companies) - set(ticker_node))
    if unknown:
        log.warning("aliases: 기업 목록에 없는 티커는 건너뜁니다: %s", unknown)

    explicit: list[NameEntry] = []
    for ticker, a in aliases.companies.items():
        node = ticker_node.get(ticker)
        if node is None:
            continue
        explicit += [NameEntry(n, node, "alias", a.since, a.until) for n in a.names]
        explicit += [
            NameEntry(p.name, node, "product", p.since or a.since, p.until or a.until)
            for p in a.products
        ]
    for ext in aliases.external:
        explicit += [NameEntry(n, ext.id, "alias") for n in ext.names]

    # 자동 이름: 여러 회사가 같은 이름을 가지면(예: Constellation) 어느 쪽인지 알 수 없어 뺀다.
    taken = {e.text for e in explicit}
    base: dict[str, set[str]] = {}
    period = {
        ticker_node[t]: (a.since, a.until) for t, a in aliases.companies.items() if t in ticker_node
    }
    for cik, name in zip(universe["cik"], universe["name"], strict=True):
        b = strip_name(name)
        single_word = len(b.split()) == 1
        if b in generic or b in taken or (single_word and len(b) < 4):
            continue
        base.setdefault(b, set()).add(company_node(cik))
    ambiguous = sorted(b for b, nodes in base.items() if len(nodes) > 1)
    if ambiguous:
        log.info("여러 회사가 공유해서 뺀 자동 이름: %s", ambiguous)
    entries = explicit + [
        NameEntry(b, node, "base", *period.get(node, (None, None)))
        for b, nodes in base.items()
        if len(nodes) == 1
        for node in nodes
    ]

    nodes = pd.concat(
        [
            pd.DataFrame(
                {
                    "node_id": [company_node(c) for c in universe["cik"]],
                    "kind": "company",
                    "cik": universe["cik"].astype("Int64"),
                    "ticker": universe["ticker"],
                    "name": universe["name"],
                }
            ),
            pd.DataFrame(
                {
                    "node_id": [e.id for e in aliases.external],
                    "kind": "external",
                    "cik": pd.array(
                        [
                            int(e.id[4:]) if e.id.startswith("cik:") else None
                            for e in aliases.external
                        ],
                        dtype="Int64",
                    ),
                    "ticker": None,
                    "name": [e.name for e in aliases.external],
                }
            ),
        ],
        ignore_index=True,
    )
    rules = [
        ContextRule(
            r.reason,
            frozenset(ticker_node[t] for t in r.tickers if t in ticker_node),
            re.compile(r.pattern, re.I),
        )
        for r in aliases.exclude_contexts
    ]
    return NameDictionary(entries, nodes, rules)


def trie_regex(words) -> str:
    """이름 수백 개를 단순 OR로 묶으면 매우 느리다. 글자 트라이로 묶어 분기를 줄인다."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = {}

    def build(node: dict) -> str:
        end = "" in node
        alts = [re.escape(ch) + build(sub) for ch, sub in sorted(node.items()) if ch != ""]
        if not alts:
            return ""
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return f"(?:{body})?" if end else body

    return build(trie)

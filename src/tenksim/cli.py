"""명령줄 인터페이스: tenksim <command> -c configs/<name>.yaml"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

import numpy as np
from dotenv import find_dotenv, load_dotenv

from . import pipeline
from .config import Config, load_config
from .report import write_report
from .similarity import explain_pair

log = logging.getLogger("tenksim")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 외부 라이브러리 로그는 경고 이상만 (edgartools는 INFO에 식별자를 찍기도 한다)
    for name in (
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "yfinance",
        "filelock",
        "hishel",
        "pyrate_limiter",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    # edgartools는 파싱 경고를 많이 낸다. 추출 품질은 우리 쪽 품질 판정(status)으로 확인한다.
    logging.getLogger("edgar").setLevel(logging.WARNING if verbose else logging.ERROR)


def cmd_ingest(cfg: Config, args) -> None:
    universe = pipeline.stage_universe(cfg)
    pipeline.stage_ingest(cfg, universe, refresh=args.refresh)


def cmd_embed(cfg: Config, args) -> None:
    universe = pipeline.load_universe(cfg)
    docs = pipeline.load_documents(cfg)
    pipeline.stage_methods(cfg, universe, docs, only=args.method)


def cmd_evaluate(cfg: Config, args) -> None:
    universe = pipeline.load_universe(cfg)
    docs = pipeline.load_documents(cfg)
    metrics = pipeline.stage_evaluate(cfg, universe, docs, pipeline.load_methods(cfg))
    path = write_report(cfg, metrics, docs, universe)
    log.info("Report written to %s", path)


def cmd_run(cfg: Config, args) -> None:
    universe = pipeline.stage_universe(cfg)
    docs = pipeline.stage_ingest(cfg, universe, refresh=args.refresh)
    results = pipeline.stage_methods(cfg, universe, docs)
    metrics = pipeline.stage_evaluate(cfg, universe, docs, results)
    path = write_report(cfg, metrics, docs, universe)
    log.info("Report written to %s", path)


def _find(res: pipeline.MethodResult, ticker: str) -> int:
    matches = np.flatnonzero(res.companies["ticker"].str.upper() == ticker.upper())
    if not len(matches):
        raise SystemExit(
            f"{ticker}: 이 method의 분석 대상에 없습니다 (품질 판정에서 빠졌을 수 있음)"
        )
    return int(matches[0])


def cmd_neighbors(cfg: Config, args) -> None:
    variant = args.method + (pipeline.CENTER_SUFFIX if args.center else "")
    try:
        companies, sim = pipeline.similarity_for(cfg, variant)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    table = pipeline.neighbors_table(companies, sim, args.k)
    matches = companies["ticker"].str.upper() == args.ticker.upper()
    if not matches.any():
        raise SystemExit(f"{args.ticker}: 분석 대상에 없습니다 (품질 판정에서 빠졌을 수 있음)")
    table = table[table["ticker"] == companies.loc[matches, "ticker"].iloc[0]]
    print(f"{args.ticker.upper()} 와 비슷한 기업 ({variant})")
    for r in table.itertuples():
        print(
            f"{r.rank:>3}. {r.neighbor_ticker:<6} {r.neighbor_name[:40]:<40} "
            f"score={r.score:.3f}  pct={r.percentile:5.1f}"
        )


def cmd_explain(cfg: Config, args) -> None:
    res = pipeline.load_method(cfg, args.method)
    if res.chunks is None:
        raise SystemExit("explain은 청크 임베딩 방법(sbert, openai)에서만 쓸 수 있습니다")
    a, b = _find(res, args.ticker_a), _find(res, args.ticker_b)
    chunks = res.chunks
    ia = np.flatnonzero(chunks["owner"].to_numpy() == a)
    ib = np.flatnonzero(chunks["owner"].to_numpy() == b)
    pairs = explain_pair(
        chunks["text"].iloc[ia].tolist(),
        res.chunk_vectors[ia],
        chunks["text"].iloc[ib].tolist(),
        res.chunk_vectors[ib],
        top=args.top,
    )
    for n, (score, ta, tb) in enumerate(pairs, start=1):
        print(f"\n[{n}] cosine={score:.3f}")
        for label, text in ((args.ticker_a.upper(), ta), (args.ticker_b.upper(), tb)):
            print(textwrap.indent(textwrap.shorten(text, 500, placeholder=" ..."), f"  {label}: "))


def cmd_mentions(cfg: Config, args) -> None:
    from .relations.stages import load_mentions, stage_mentions

    tables = load_mentions(cfg) if args.cached else stage_mentions(cfg)
    if args.ticker:
        _print_mentions(tables, args.ticker, args.all)


def _print_mentions(tables, ticker: str, show_excluded: bool) -> None:
    nodes = tables.nodes.set_index("node_id")
    match = nodes.index[nodes["ticker"].fillna("").str.upper() == ticker.upper()]
    if not len(match):
        raise SystemExit(f"{ticker}: 기업 목록에 없습니다")
    node = match[0]
    label = nodes["name"].to_dict()
    m = tables.mentions.merge(tables.spans[["span_id", "text", "lead_text"]], on="span_id")
    if not show_excluded:
        m = m[m["excluded"].isna()]
    for title, rows, other in (
        (f"{ticker.upper()}의 10-K가 언급한 회사", m[m["doc_node"] == node], "target_node"),
        (f"{ticker.upper()}를 언급한 회사", m[m["target_node"] == node], "doc_node"),
    ):
        print(f"\n== {title}: {rows[other].nunique()}곳, 언급 {len(rows)}건")
        for other_node, g in sorted(rows.groupby(other), key=lambda kv: -len(kv[1])):
            print(f"\n  {label.get(other_node, other_node)} ({other_node}) — {len(g)}건")
            for r in g.drop_duplicates("span_id").head(3).itertuples():
                flag = f" [제외: {r.excluded}]" if isinstance(r.excluded, str) else ""
                text = f"[{r.lead_text}] {r.text}" if isinstance(r.lead_text, str) else r.text
                snippet = textwrap.shorten(text, 300, placeholder=" ...")
                print(textwrap.indent(snippet, f"    {r.section[:4]}{flag} | "))


def main(argv: list[str] | None = None) -> None:
    # 설치 위치가 아니라 명령을 실행한 폴더에서 .env를 찾는다
    load_dotenv(find_dotenv(usecwd=True))
    # Windows 콘솔(cp949)에서 회사명의 특수문자 때문에 죽지 않도록
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="tenksim", description="10-K 텍스트 기반 기업 유사도")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, func) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("-c", "--config", required=True, help="configs/*.yaml")
        p.set_defaults(func=func)
        return p

    p = add("run", "전체 단계 실행 (ingest → embed → evaluate)", cmd_run)
    p.add_argument("--refresh", action="store_true", help="받아 둔 10-K도 다시 받는다")
    p = add("ingest", "기업 목록 구성 + 10-K 섹션 수집 + 정제/품질 판정", cmd_ingest)
    p.add_argument("--refresh", action="store_true", help="받아 둔 10-K도 다시 받는다")
    p = add("embed", "method별 기업 벡터 생성", cmd_embed)
    p.add_argument("--method", action="append", help="이 method만 (여러 번 지정 가능)")
    add("evaluate", "평가 지표 계산 + 리포트 작성", cmd_evaluate)
    p = add("neighbors", "한 회사의 유사 기업 목록", cmd_neighbors)
    p.add_argument("ticker")
    p.add_argument("--method", required=True)
    p.add_argument("--center", action="store_true", help="+center 변형 사용")
    p.add_argument("-k", type=int, default=10)
    p = add("explain", "두 회사가 비슷하다고 나온 근거 문단", cmd_explain)
    p.add_argument("ticker_a")
    p.add_argument("ticker_b")
    p.add_argument("--method", required=True)
    p.add_argument("--top", type=int, default=3)

    p = add("mentions", "관계도: 10-K 본문의 회사 이름 언급 찾기", cmd_mentions)
    p.add_argument("--ticker", help="이 회사의 언급을 출력")
    p.add_argument("--all", action="store_true", help="제외된 언급(임원 약력 등)도 출력")
    p.add_argument("--cached", action="store_true", help="다시 찾지 않고 저장된 결과만 출력")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(load_config(args.config), args)

"""명령줄 인터페이스: tenksim <command> -c configs/<name>.yaml"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

import numpy as np
import pandas as pd
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
        "typesafe_sdk",
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
    if cfg.relations is not None:
        from .relations.stages import stage_relations

        log.info("Graph database: %s", stage_relations(cfg))


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


def cmd_candidates(cfg: Config, args) -> None:
    from .relations.stages import load_candidates, load_mentions, stage_candidates

    cands = load_candidates(cfg) if args.cached else stage_candidates(cfg)
    if args.ticker:
        _print_candidates(cands, load_mentions(cfg), args.ticker, args.spans)


def _print_candidates(cands, tables, ticker: str, show_spans: bool) -> None:
    nodes = tables.nodes.set_index("node_id")
    match = nodes.index[nodes["ticker"].fillna("").str.upper() == ticker.upper()]
    if not len(match):
        raise SystemExit(f"{ticker}: 기업 목록에 없습니다")
    node = match[0]
    c = cands.candidates
    c = c[(c["node_a"] == node) | (c["node_b"] == node)].copy()
    a_side = c["node_a"] == node
    c["other"] = c["node_b"].where(a_side, c["node_a"])
    c["rank"] = c["rank_ab"].where(a_side, c["rank_ba"])  # 이 회사 기준 상대의 유사도 순위
    c["mentions_out"] = c["mentions_ab"].where(a_side, c["mentions_ba"])
    c["mentions_in"] = c["mentions_ba"].where(a_side, c["mentions_ab"])
    c = c.sort_values(["rank", "other"], na_position="last")
    label = nodes["name"].to_dict()
    span_text = tables.spans.set_index("span_id")["text"]
    print(f"{ticker.upper()} 후보 {len(c)}쌍 ({c['source'].value_counts().to_dict()})")
    print(f"{'상대':<40} {'출처':<10} {'순위':>4} {'언급(나→상대/상대→나)':>12} 구간")
    for r in c.itertuples():
        rank = "-" if pd.isna(r.rank) else str(int(r.rank))
        mentions = f"{r.mentions_out}/{r.mentions_in}"
        print(
            f"{label.get(r.other, r.other)[:40]:<40} {r.source:<10} {rank:>4} {mentions:>12} {r.n_spans}"
        )
        if show_spans:
            for s in cands.candidate_spans[
                cands.candidate_spans["pair_key"] == r.pair_key
            ].itertuples():
                who = "나" if s.doc_node == node else "상대"
                snippet = textwrap.shorten(span_text[s.span_id], 200, placeholder=" ...")
                print(f"    [{who} 10-K · {s.cues or '-'}] {snippet}")


def cmd_export(cfg: Config, args) -> None:
    from .relations.stages import stage_export, stage_relations

    path = stage_relations(cfg, judge=False) if args.all else stage_export(cfg)
    log.info("Graph database: %s", path)


def cmd_sample(cfg: Config, args) -> None:
    import sqlite3

    from .relations import reviews
    from .relations.stages import graph_path, judge_settings

    path = graph_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 먼저 `tenksim export --all -c ...`를 실행하세요")
    rev = reviews.connect(path.with_name("reviews.sqlite"))
    if not args.name:
        for s in reviews.sample_progress(rev):
            print(
                f"{s['sample_id']:<16} {s['purpose']:<8} 회사 {s['n_companies']:>3}  "
                f"검수 {s['n_labeled']:>4} / {s['n_units']:<4} (보류 {s['n_skipped'] or 0})  "
                f"{s['created_at']}"
            )
        return
    graph = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    graph.row_factory = sqlite3.Row
    try:
        res = reviews.create_sample(
            rev, graph, sample_id=args.name, purpose=args.purpose, n_companies=args.companies,
            per_company_cap=args.cap, seed=args.seed, config=cfg.name,
            settings=judge_settings(cfg) if args.purpose == "confirm" else None,
        )  # fmt: skip
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    names = dict(graph.execute("SELECT node_id, name FROM nodes"))
    print(
        f"표본 {res.sample_id} ({args.purpose}): 회사 {len(res.companies)}곳, 검수 단위 {res.n_units}개"
    )
    for c in res.companies:
        capped = f" (전체 {c['n_units_total']}개 중)" if c["n_units_total"] > c["n_units"] else ""
        print(f"  {names.get(c['node_id'], c['node_id'])}: {c['n_units']}개{capped}")
    minutes = res.n_units * 20 / 60
    print(
        f"건당 20초로 잡으면 약 {minutes:.0f}분입니다. `tenksim serve`에서 '표본 검수'로 들어가세요."
    )
    frozen = reviews.frozen_settings(rev, res.sample_id)
    if frozen:
        t = ", ".join(f"{q} {a}/{r}" for q, (a, r) in frozen["thresholds"].items())
        print(
            f"확인 표본이라 판정 설정을 고정했습니다: {frozen['model']}, 질문 "
            f"{frozen['question_version']}, 입력 {frozen['context']}, 채택/기각 {t}. "
            "검수를 마친 뒤 이 설정 그대로 eval-relations를 한 번 실행합니다."
        )


def cmd_judge(cfg: Config, args) -> None:
    import json

    from .relations.stages import make_judge, sample_requests, stage_judge, stage_judge_all

    if args.all:
        if args.dry_run or args.sample or args.limit:
            raise SystemExit("--all은 --sample, --limit, --dry-run과 함께 쓸 수 없습니다")
        log.info("Graph database: %s", stage_judge_all(cfg, args.model))
        return
    if not args.sample:
        raise SystemExit("--sample 이름이나 --all을 적어 주세요")
    if args.dry_run:
        judge = make_judge(cfg, args.model)
        requests = sample_requests(cfg, args.sample, args.limit, args.context)
        payload = judge.payload(requests[0])
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        chars = sum(len(json.dumps(judge.payload(r), ensure_ascii=False)) for r in requests)
        print(
            f"\n요청 {len(requests)}건, 입력 약 {chars / 4:,.0f}토큰 (글자 수 / 4로 어림). "
            "--dry-run을 빼면 실제로 판정합니다."
        )
        return
    stage_judge(cfg, args.sample, args.model, args.limit, context=args.context)


def cmd_eval_relations(cfg: Config, args) -> None:
    from .relations.stages import stage_eval

    _, report, path = stage_eval(
        cfg, args.sample, args.model, context=args.context, force=args.force
    )
    print(report)
    log.info("채점 결과: %s (라벨과 어긋난 단위는 같은 폴더의 *_disagreements.csv)", path)


def cmd_relations_report(cfg: Config, args) -> None:
    from .relations.research import stage_research

    _, path = stage_research(cfg, n_boot=args.n_boot)
    log.info("관계 유형별 주가 동조성 리포트: %s", path)


def cmd_relations_events(cfg: Config, args) -> None:
    from .relations.events import stage_events

    _, path = stage_events(cfg, n_boot=args.n_boot, refresh=args.refresh)
    log.info("실적 발표 이벤트 스터디 리포트: %s", path)


def cmd_serve(cfg: Config, args) -> None:
    try:
        import uvicorn

        from .app.server import STATIC_DIR, create_app
    except ImportError as exc:
        raise SystemExit("웹앱 의존성이 없습니다: uv sync --extra app") from exc
    from .relations.stages import graph_path

    path = graph_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 먼저 `tenksim export --all -c ...`를 실행하세요")
    if not (STATIC_DIR / "index.html").exists():
        log.warning(
            "화면 파일이 없습니다. web/ 폴더에서 `npm install && npm run build`를 실행하세요"
        )
    url = f"http://127.0.0.1:{args.port}"
    log.info("관계도 웹앱: %s (끝내려면 Ctrl+C)", url)
    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # 로컬 단일 사용자 앱이므로 이 컴퓨터에서만 접속할 수 있게 127.0.0.1로 고정한다
    uvicorn.run(create_app(path), host="127.0.0.1", port=args.port, log_level="warning")


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

    p = add(
        "run", "전체 단계 실행 (ingest → embed → evaluate, relations가 있으면 관계도까지)", cmd_run
    )
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
    p = add(
        "candidates", "관계도: 판정할 후보 쌍 만들기 (유사도 상위 K ∪ 이름 언급)", cmd_candidates
    )
    p.add_argument("--ticker", help="이 회사의 후보를 출력")
    p.add_argument("--spans", action="store_true", help="후보마다 판정에 넣을 근거 구간도 출력")
    p.add_argument("--cached", action="store_true", help="다시 만들지 않고 저장된 결과만 출력")
    p = add("export", "관계도: 웹앱이 읽는 graph.db 만들기", cmd_export)
    p.add_argument("--all", action="store_true", help="이름 언급과 후보도 새로 만든 뒤 내보낸다")
    p = add("sample", "관계도: 검수할 표본 회사 뽑기 (이름 없이 실행하면 표본 목록)", cmd_sample)
    p.add_argument("--name", help="새 표본 이름 (예: dev1)")
    p.add_argument("--purpose", choices=["dev", "confirm"], default="dev",
                   help="dev: 질문·기준 조정용, confirm: 합격 판단용(한 번만 씀)")  # fmt: skip
    p.add_argument("--companies", type=int, default=10, help="회사 수")
    p.add_argument("--cap", type=int, default=40, help="회사당 최대 검수 단위 수")
    p.add_argument("--seed", type=int, default=0)
    p = add(
        "judge",
        "관계도: 판정 모델(Jev)로 판정 (결과는 캐시에 쌓임). --all이면 관계까지 만든다",
        cmd_judge,
    )
    p.add_argument("--sample", help="표본 이름 (예: dev1)")
    p.add_argument("--all", action="store_true",
                   help="모든 이름 언급을 판정하고 관계를 합쳐 graph.db를 다시 쓴다")  # fmt: skip
    p.add_argument("--model", help="판정 모델 (기본: 설정의 relations.judge.model)")
    p.add_argument("--limit", type=int, help="앞에서부터 이 개수만 (시험용)")
    p.add_argument("--dry-run", action="store_true", help="보내지 않고 첫 요청과 어림 토큰만 출력")
    p.add_argument("--context", choices=["span", "nearby", "pair"],
                   help="판정 입력 (기본: 설정의 relations.judge.context)")  # fmt: skip
    p = add("eval-relations", "관계도: 판정을 표본 검수 라벨로 채점 (판정이 없으면 먼저 판정)",
            cmd_eval_relations)  # fmt: skip
    p.add_argument("--sample", required=True, help="표본 이름 (예: dev1)")
    p.add_argument("--model", help="판정 모델 (기본: 설정의 relations.judge.model)")
    p.add_argument("--context", choices=["span", "nearby", "pair"],
                   help="판정 입력 (기본: 설정의 relations.judge.context)")  # fmt: skip
    p.add_argument("--force", action="store_true",
                   help="확인 표본의 고정 설정과 달라도 채점 (합격 판단에는 쓰지 않음)")  # fmt: skip
    p = add("relations-report", "관계도: 관계 유형별 주가 동조성 분석과 리포트",
            cmd_relations_report)  # fmt: skip
    p.add_argument("--n-boot", type=int, help="부트스트랩 횟수 (기본: evaluation.n_boot)")
    p = add("relations-events", "관계도: 실적 발표 때 관계 상대의 주가 반응 (이벤트 스터디)",
            cmd_relations_events)  # fmt: skip
    p.add_argument("--n-boot", type=int, help="부트스트랩 횟수 (기본: evaluation.n_boot)")
    p.add_argument("--refresh", action="store_true", help="받아 둔 8-K 목록도 다시 받는다")
    p = add("serve", "관계도 웹앱 실행 (127.0.0.1)", cmd_serve)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(load_config(args.config), args)

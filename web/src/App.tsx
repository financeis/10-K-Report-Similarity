import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type CandidateRow,
  type EvidenceSpan,
  type Figure,
  type NodeDetail,
  type NodeSummary,
} from "./api";
import { Highlighted, type Range } from "./Highlighted";
import { href, useFetch, useRoute } from "./hooks";
import { CUE, KIND, OPERATOR, SECTION, SOURCE, excludedLabel, sectorColor } from "./labels";

export function App() {
  const route = useRoute();
  const meta = useFetch("meta", api.meta);
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          10-K 관계도
        </a>
        <SearchBox />
        {meta.data && (
          <span className="topbar-meta">
            {meta.data.filings_year}년 10-K · 후보 기준 {meta.data.similarity} 상위 {meta.data.top_k}
          </span>
        )}
      </header>
      {meta.error ? (
        <main className="page">
          <ErrorBox message={meta.error} />
        </main>
      ) : route.node ? (
        <NodePage nodeId={route.node} pairKey={route.pair} />
      ) : (
        <Home />
      )}
    </div>
  );
}

// ---------------------------------------------------------------- 검색

function SearchBox() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 150);
    return () => clearTimeout(t);
  }, [q]);
  const results = useFetch(debounced ? `search:${debounced}` : null, () => api.search(debounced));
  const go = (n: NodeSummary) => {
    window.location.hash = href(n.node_id);
    setQ("");
    setOpen(false);
  };
  return (
    <div className="search">
      <input
        type="search"
        placeholder="회사 이름이나 티커 (예: NVDA, Amazon)"
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && results.data?.length) go(results.data[0]);
          if (e.key === "Escape") setOpen(false);
        }}
        aria-label="회사 검색"
      />
      {open && debounced && results.data && (
        <ul className="search-results" role="listbox">
          {results.data.length === 0 && <li className="muted">찾는 회사가 없습니다</li>}
          {results.data.slice(0, 12).map((n) => (
            <li key={n.node_id} onMouseDown={() => go(n)} role="option">
              <NodeName name={n.name} ticker={n.ticker} sector={n.gics_sector} kind={n.kind} />
              <span className="muted small">후보 {n.n_candidates}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Home() {
  const list = useFetch("home", () => api.search(""));
  return (
    <main className="page">
      <section className="intro">
        <h1>10-K에서 찾은 기업 관계 후보</h1>
        <p>
          회사를 고르면 판정 전 후보 쌍(유사도 상위 기업 ∪ 10-K에 이름이 나온 기업)과 그 근거 문장을
          볼 수 있습니다. 아직 모델 판정 전이라, 관계의 종류는 표시하지 않습니다.
        </p>
      </section>
      <h2 className="section-title">이름 언급 후보가 많은 회사</h2>
      {list.error && <ErrorBox message={list.error} />}
      <div className="card-grid">
        {list.data?.map((n) => (
          <a key={n.node_id} className="company-card" href={href(n.node_id)}>
            <NodeName name={n.name} ticker={n.ticker} sector={n.gics_sector} kind={n.kind} />
            <span className="muted small">
              후보 {n.n_candidates} · 언급 {n.n_mention_candidates}
            </span>
          </a>
        ))}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------- 회사 화면

type Filter = "all" | "mention" | "similarity" | "cross";

const FILTERS: [Filter, string][] = [
  ["all", "전체"],
  ["mention", "이름 언급 있음"],
  ["similarity", "유사도만"],
  ["cross", "섹터가 다른 회사"],
];

function NodePage({ nodeId, pairKey }: { nodeId: string; pairKey: string | null }) {
  const node = useFetch(`node:${nodeId}`, () => api.node(nodeId));
  const cands = useFetch(`cands:${nodeId}`, () => api.candidates(nodeId));
  const [filter, setFilter] = useState<Filter>("all");

  const rows = useMemo(() => {
    const all = cands.data ?? [];
    const sector = node.data?.gics_sector;
    return all.filter((c) => {
      if (filter === "mention") return c.source !== "similarity";
      if (filter === "similarity") return c.source === "similarity";
      if (filter === "cross") return !!sector && c.other_sector !== sector;
      return true;
    });
  }, [cands.data, filter, node.data]);

  const counts = useMemo(() => {
    const all = cands.data ?? [];
    const sector = node.data?.gics_sector;
    return {
      all: all.length,
      mention: all.filter((c) => c.source !== "similarity").length,
      similarity: all.filter((c) => c.source === "similarity").length,
      cross: all.filter((c) => !!sector && c.other_sector !== sector).length,
    };
  }, [cands.data, node.data]);

  // 쌍을 고르지 않았으면 첫 언급 후보를 보여준다 (주소는 바꾸지 않음)
  const shownPair =
    pairKey ?? cands.data?.find((c) => c.source !== "similarity")?.pair_key ?? cands.data?.[0]?.pair_key ?? null;

  if (node.error) return <main className="page"><ErrorBox message={node.error} /></main>;
  return (
    <main className="split">
      <section className="left">
        {node.data && <NodeHeader node={node.data} />}
        <div className="chips" role="tablist">
          {FILTERS.map(([key, label]) => (
            <button
              key={key}
              role="tab"
              aria-selected={filter === key}
              className={filter === key ? "chip on" : "chip"}
              onClick={() => setFilter(key)}
            >
              {label} <span className="count">{counts[key]}</span>
            </button>
          ))}
        </div>
        {cands.error && <ErrorBox message={cands.error} />}
        <CandidateTable rows={rows} nodeId={nodeId} selected={shownPair} />
      </section>
      <section className="right">
        {shownPair ? (
          <PairPanel key={shownPair} pairKey={shownPair} viewer={nodeId} />
        ) : (
          !cands.loading && <p className="muted pad">후보가 없습니다.</p>
        )}
      </section>
    </main>
  );
}

function NodeHeader({ node }: { node: NodeDetail }) {
  return (
    <div className="node-header">
      <h1>
        <span className="dot big" style={{ background: sectorColor(node.gics_sector) }} />
        {node.name}
        {node.ticker && <span className="ticker">{node.ticker}</span>}
        {KIND[node.kind] && <span className="tag">{KIND[node.kind]}</span>}
      </h1>
      <p className="muted">
        {node.gics_sector ? `${node.gics_sector} · ${node.gics_sub_industry}` : "GICS 분류 없음"}
        {node.parent && (
          <>
            {" · 공시한 회사 "}
            <a href={href(node.parent.node_id)}>{node.parent.name}</a>
          </>
        )}
      </p>
      {node.filings.map((f) => (
        <p key={f.accession} className="small muted">
          <a href={f.filing_url} target="_blank" rel="noreferrer">
            {f.form} (제출 {f.filing_date}, 회계기간 {f.period_of_report}) ↗
          </a>
          {" · 분석한 항목: "}
          {(f.sections ?? "").split(",").map((s) => SECTION[s] ?? s).join(", ")}
        </p>
      ))}
      {!node.analyzed && node.kind === "company" && (
        <p className="small warn">이 회사의 10-K는 추출 품질 검사를 통과하지 못해, 다른 회사 10-K의 언급만 보입니다.</p>
      )}
    </div>
  );
}

function CandidateTable({
  rows,
  nodeId,
  selected,
}: {
  rows: CandidateRow[];
  nodeId: string;
  selected: string | null;
}) {
  return (
    <div className="table-wrap">
      <table className="cands">
        <thead>
          <tr>
            <th>상대 회사</th>
            <th>출처</th>
            <th title="이 회사 기준 상대의 유사도 순위 / 상대 기준 이 회사의 순위">유사도 순위</th>
            <th title="이 회사 10-K가 상대를 언급한 횟수 / 상대 10-K가 이 회사를 언급한 횟수">언급 나→ / →나</th>
            <th title="판정에 넣을 근거 문장 수">근거</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr
              key={c.pair_key}
              className={c.pair_key === selected ? "sel" : ""}
              onClick={() => (window.location.hash = href(nodeId, c.pair_key))}
            >
              <td>
                <NodeName name={c.other_name} ticker={c.other_ticker} sector={c.other_sector} kind={c.other_kind} />
              </td>
              <td>
                <span className={`badge src-${c.source}`}>{SOURCE[c.source]}</span>
              </td>
              <td className="num">
                {c.rank_mine ?? "–"} <span className="muted">/ {c.rank_theirs ?? "–"}</span>
              </td>
              <td className="num">
                {c.mentions_out} <span className="muted">/ {c.mentions_in}</span>
              </td>
              <td className="num">{c.n_spans || <span className="muted">0</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && <p className="muted pad">이 조건의 후보가 없습니다.</p>}
    </div>
  );
}

// ---------------------------------------------------------------- 후보 쌍 패널

function PairPanel({ pairKey, viewer }: { pairKey: string; viewer: string }) {
  const pair = useFetch(`pair:${pairKey}`, () => api.candidate(pairKey));
  const [contextSpan, setContextSpan] = useState<string | null>(null);
  if (pair.error) return <ErrorBox message={pair.error} />;
  if (!pair.data) return <p className="muted pad">불러오는 중…</p>;
  const d = pair.data;
  // 보고 있는 회사가 왼쪽에 오게
  const [me, other] = d.a.node_id === viewer ? [d.a, d.b] : [d.b, d.a];
  const rankMe = d.a.node_id === viewer ? d.rank_ab : d.rank_ba;
  const rankOther = d.a.node_id === viewer ? d.rank_ba : d.rank_ab;
  const names = { [d.a.node_id]: d.a.name, [d.b.node_id]: d.b.name };
  const live = d.spans.filter((s) => !s.excluded);
  const excluded = d.spans.filter((s) => s.excluded);

  return (
    <div className="pair">
      <h2 className="pair-title">
        <a href={href(me.node_id)}>{me.name}</a>
        <span className="muted"> ↔ </span>
        <a href={href(other.node_id)}>{other.name}</a>
      </h2>
      <dl className="stats">
        <div>
          <dt>후보가 된 이유</dt>
          <dd>
            <span className={`badge src-${d.source}`}>{SOURCE[d.source]}</span>
          </dd>
        </div>
        <div>
          <dt>유사도 순위</dt>
          <dd>
            {rankMe == null ? (
              "유사도 없음 (분석 대상 밖)"
            ) : (
              <>
                {me.ticker ?? me.name} 기준 {rankMe}위 · {other.ticker ?? other.name} 기준 {rankOther}위
              </>
            )}
          </dd>
        </div>
        {d.similarity_pct != null && (
          <div>
            <dt>전체 쌍 중 유사도</dt>
            <dd>상위 {Math.max(0.1, 100 - d.similarity_pct).toFixed(1)}%</dd>
          </div>
        )}
        <div>
          <dt>상태</dt>
          <dd>판정 전</dd>
        </div>
      </dl>

      {d.figures.length > 0 && <Figures figures={d.figures} spans={d.spans} names={names} />}

      {live.length === 0 ? (
        <p className="note">
          두 회사 10-K 어디에도 서로의 이름이 나오지 않습니다. 사업 설명이 비슷해서 후보가 되었고, 판정 단계에서
          두 회사의 사업 설명을 비교해 판단합니다.
        </p>
      ) : (
        [me, other].map((side) => {
          const spans = live.filter((s) => s.doc_node === side.node_id);
          if (!spans.length) return null;
          return (
            <section key={side.node_id} className="evidence-group">
              <h3>
                {side.name}의 10-K에서 <span className="muted">{spans.length}문장</span>
              </h3>
              {spans.map((s) => (
                <SpanCard key={s.span_id} span={s} onContext={() => setContextSpan(s.span_id)} />
              ))}
            </section>
          );
        })
      )}

      {excluded.length > 0 && (
        <details className="excluded">
          <summary>관계와 무관해 제외한 언급 {excluded.length}건</summary>
          {excluded.map((s) => (
            <SpanCard key={s.span_id} span={s} onContext={() => setContextSpan(s.span_id)} />
          ))}
        </details>
      )}
      {contextSpan && <ContextModal spanId={contextSpan} onClose={() => setContextSpan(null)} />}
    </div>
  );
}

function SpanCard({ span, onContext }: { span: EvidenceSpan; onContext: () => void }) {
  const ranges: Range[] = span.highlights.map((h) => ({ start: h.start, end: h.end, className: "name" }));
  return (
    <article className={span.excluded ? "span-card is-excluded" : "span-card"}>
      <div className="span-meta">
        <span>{SECTION[span.section] ?? span.section}</span>
        {span.excluded ? (
          <span className="tag warn">제외: {excludedLabel(span.excluded)}</span>
        ) : span.input_order != null ? (
          <span className="tag ok" title="판정 모델에 넣을 근거 문장">판정 입력 #{span.input_order + 1}</span>
        ) : (
          <span className="tag" title="한 회사 쪽 근거가 6개를 넘어 판정 입력에서 빠짐">입력 밖</span>
        )}
        {span.cues?.split(",").filter(Boolean).map((c) => (
          <span key={c} className="cue">
            {CUE[c] ?? c}
          </span>
        ))}
        <button className="link" onClick={onContext}>
          본문에서 보기
        </button>
      </div>
      {span.lead_text && <p className="lead">{span.lead_text} …</p>}
      <p className="span-text">
        <Highlighted text={span.text} ranges={ranges} />
      </p>
    </article>
  );
}

function Figures({
  figures,
  spans,
  names,
}: {
  figures: Figure[];
  spans: EvidenceSpan[];
  names: Record<string, string>;
}) {
  const docOf = Object.fromEntries(spans.map((s) => [s.span_id, s.doc_node]));
  // 같은 문장이 Item 1과 1A에 반복되면 같은 수치가 두 번 나온다
  const seen = new Set<string>();
  const unique = figures.filter((f) => {
    const key = `${docOf[f.span_id]}|${f.target_node}|${f.value}|${f.operator}|${f.period}|${f.raw}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return (
    <section className="figures">
      <h3>매출 비중 후보</h3>
      <ul>
        {unique.map((f) => (
          <li key={f.figure_id}>
            {f.value != null ? (
              <>
                <b>{names[docOf[f.span_id]]}</b> 매출에서 <b>{names[f.target_node]}</b>{" "}
                {f.subject === "each" ? "(각각) " : ""}
                <b>
                  {OPERATOR[f.operator ?? "="]}
                  {f.value}%
                </b>
                <span className="muted">
                  {" "}
                  ({[f.period, f.denominator].filter(Boolean).join(", ")})
                </span>
              </>
            ) : (
              <>
                <span className="muted">귀속이 불분명해 수치를 비움:</span> “{f.raw}”
              </>
            )}
          </li>
        ))}
      </ul>
      <p className="small muted">문장에서 코드로 뽑은 후보입니다. 판정 전이므로 원문으로 확인하세요.</p>
    </section>
  );
}

function ContextModal({ spanId, onClose }: { spanId: string; onClose: () => void }) {
  const ctx = useFetch(`ctx:${spanId}`, () => api.context(spanId));
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  useEffect(() => {
    body.current?.querySelector("#focus")?.scrollIntoView({ block: "center" });
  }, [ctx.data]);
  const ranges: Range[] = (ctx.data?.marks ?? []).map((m) => ({
    start: m.start,
    end: m.end,
    className: m.kind,
    id: m.kind === "span" ? "focus" : undefined,
  }));
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <header>
          {ctx.data && (
            <div>
              <b>{ctx.data.name}</b> {ctx.data.ticker && <span className="ticker">{ctx.data.ticker}</span>}
              <span className="muted">
                {" "}
                · {SECTION[ctx.data.section] ?? ctx.data.section} · 제출 {ctx.data.filing_date}
              </span>
              <div className="small muted">
                정제한 본문 {ctx.data.offset.toLocaleString()}번째 글자부터 · 표·쪽번호는 빠져 있습니다 ·{" "}
                <a href={ctx.data.filing_url} target="_blank" rel="noreferrer">
                  EDGAR 원문 ↗
                </a>
              </div>
            </div>
          )}
          <button className="close" onClick={onClose} aria-label="닫기">
            ×
          </button>
        </header>
        <div className="modal-body" ref={body}>
          {ctx.error && <ErrorBox message={ctx.error} />}
          {ctx.data && (
            <p className="doc">
              <Highlighted text={ctx.data.excerpt} ranges={ranges} />
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 공통

function NodeName({
  name,
  ticker,
  sector,
  kind,
}: {
  name: string;
  ticker: string | null;
  sector: string | null;
  kind: string;
}) {
  return (
    <span className="node-name">
      <span
        className={kind === "anonymous" ? "dot hollow" : "dot"}
        style={{ background: kind === "company" ? sectorColor(sector) : undefined }}
        title={sector ?? KIND[kind]}
      />
      <span className="name">{name}</span>
      {ticker && <span className="ticker">{ticker}</span>}
      {KIND[kind] && <span className="tag">{KIND[kind]}</span>}
    </span>
  );
}

function ErrorBox({ message }: { message: string }) {
  return <p className="error">불러오지 못했습니다: {message}</p>;
}

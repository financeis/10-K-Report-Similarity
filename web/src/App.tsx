import { useEffect, useMemo, useState } from "react";
import { api, type CandidateRow, type EvidenceSpan, type NodeDetail, type NodeSummary } from "./api";
import { ErrorBox, NodeName } from "./Common";
import { ContextModal } from "./ContextModal";
import { EdgeQueue } from "./EdgeQueue";
import { candidatesHref, href, queueHref, reviewHref, useFetch, useRoute, type NodeTab } from "./hooks";
import { Figures, NodeHeader, SpanText } from "./NodeParts";
import { RelationsView } from "./Relations";
import { ReviewPage } from "./Review";
import { CANDIDATE_STATUS, CUE, RELATION, SECTION, SOURCE, excludedLabel } from "./labels";

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
        <nav className="nav">
          <a href="#/" className={!route.review && !route.queue ? "on" : ""}>관계도</a>
          <a href={queueHref()} className={route.queue ? "on" : ""}>관계 검수</a>
          <a href={reviewHref()} className={route.review ? "on" : ""}>표본 검수</a>
        </nav>
        {meta.data && (
          <span className="topbar-meta">
            {meta.data.filings_year}년 10-K · 판정 {judgeModel(meta.data)} · 후보 {meta.data.similarity} 상위{" "}
            {meta.data.top_k}
          </span>
        )}
      </header>
      {meta.error ? (
        <main className="page">
          <ErrorBox message={meta.error} />
        </main>
      ) : route.review ? (
        <ReviewPage sample={route.sample} ord={route.ord} />
      ) : route.queue ? (
        <EdgeQueue edgeId={route.edge} />
      ) : route.node ? (
        <NodePage nodeId={route.node} tab={route.tab} pairKey={route.pair} edgeId={route.edge} />
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

function judgeModel(meta: Record<string, string>): string {
  try {
    return JSON.parse(meta.judge ?? "{}").model ?? "없음";
  } catch {
    return "없음";
  }
}

function Home() {
  const list = useFetch("home", () => api.search(""));
  return (
    <main className="page">
      <section className="intro">
        <h1>10-K에서 확인한 기업 관계</h1>
        <p>
          S&amp;P 500 회사의 10-K(Item 1·1A)에 적힌 문장에서 두 회사의 관계를 찾았습니다. 관계는 경쟁, 공급·협력,
          지분 세 가지이고 방향은 따지지 않습니다. 회사를 고르면 관계도와 근거 문장이 나옵니다. 지분 관계는 아직
          검증 전입니다.
        </p>
      </section>
      <h2 className="section-title">관계가 많은 회사</h2>
      {list.error && <ErrorBox message={list.error} />}
      <div className="card-grid">
        {list.data?.map((n) => (
          <a key={n.node_id} className="company-card" href={href(n.node_id)}>
            <NodeName name={n.name} ticker={n.ticker} sector={n.gics_sector} kind={n.kind} />
            <span className="muted small">
              관계 {n.n_relations} · 후보 {n.n_candidates}
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

function NodeTabs({ nodeId, tab }: { nodeId: string; tab: NodeTab }) {
  return (
    <div className="tabs" role="tablist">
      <a role="tab" aria-selected={tab === "relations"} className={tab === "relations" ? "on" : ""} href={href(nodeId)}>
        관계
      </a>
      <a
        role="tab"
        aria-selected={tab === "candidates"}
        className={tab === "candidates" ? "on" : ""}
        href={candidatesHref(nodeId)}
      >
        후보 전체
      </a>
    </div>
  );
}

function NodePage({
  nodeId,
  tab,
  pairKey,
  edgeId,
}: {
  nodeId: string;
  tab: NodeTab;
  pairKey: string | null;
  edgeId: string | null;
}) {
  const node = useFetch(`node:${nodeId}`, () => api.node(nodeId));
  if (node.error) return <main className="page"><ErrorBox message={node.error} /></main>;
  if (!node.data || node.data.node_id !== nodeId) return <main className="page"><p className="muted">불러오는 중…</p></main>;
  const header = (
    <>
      <NodeHeader node={node.data} />
      <NodeTabs nodeId={nodeId} tab={tab} />
    </>
  );
  return (
    <main className="split">
      {tab === "relations" ? (
        <RelationsView key={nodeId} node={node.data} edgeId={edgeId} header={header} />
      ) : (
        <CandidatesView key={nodeId} node={node.data} pairKey={pairKey} header={header} />
      )}
    </main>
  );
}

function CandidatesView({ node, pairKey, header }: { node: NodeDetail; pairKey: string | null; header: React.ReactNode }) {
  const nodeId = node.node_id;
  const cands = useFetch(`cands:${nodeId}`, () => api.candidates(nodeId));
  const [filter, setFilter] = useState<Filter>("all");

  const rows = useMemo(() => {
    const all = cands.data ?? [];
    const sector = node.gics_sector;
    return all.filter((c) => {
      if (filter === "mention") return c.source !== "similarity";
      if (filter === "similarity") return c.source === "similarity";
      if (filter === "cross") return !!sector && !!c.other_sector && c.other_sector !== sector;
      return true;
    });
  }, [cands.data, filter, node.gics_sector]);

  const counts = useMemo(() => {
    const all = cands.data ?? [];
    const sector = node.gics_sector;
    return {
      all: all.length,
      mention: all.filter((c) => c.source !== "similarity").length,
      similarity: all.filter((c) => c.source === "similarity").length,
      cross: all.filter((c) => !!sector && !!c.other_sector && c.other_sector !== sector).length,
    };
  }, [cands.data, node.gics_sector]);

  // 쌍을 고르지 않았으면 첫 언급 후보를 보여준다 (주소는 바꾸지 않음)
  const shownPair =
    pairKey ?? cands.data?.find((c) => c.source !== "similarity")?.pair_key ?? cands.data?.[0]?.pair_key ?? null;

  return (
    <>
      <section className="left">
        {header}
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
    </>
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
            <th>판정 결과</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr
              key={c.pair_key}
              className={c.pair_key === selected ? "sel" : ""}
              onClick={() => (window.location.hash = candidatesHref(nodeId, c.pair_key))}
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
              <td>
                <span className={`status st-${c.status}`}>{CANDIDATE_STATUS[c.status] ?? c.status}</span>
              </td>
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
  const docOf = Object.fromEntries(d.spans.map((s) => [s.span_id, s.doc_node]));
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
          <dt>판정 결과</dt>
          <dd>
            {CANDIDATE_STATUS[d.status] ?? d.status}
            {d.edges
              .filter((e) => e.state !== "rejected")
              .map((e) => (
                <span key={e.edge_id}>
                  {" · "}
                  <a className="small" href={href(viewer, e.edge_id)}>
                    {RELATION[e.relation]} 관계 보기 →
                  </a>
                </span>
              ))}
          </dd>
        </div>
      </dl>

      <Figures
        figures={d.figures}
        docOf={(f) => docOf[f.span_id]}
        names={names}
        note="문장에서 코드로 뽑은 후보입니다. 원문으로 확인하세요."
      />

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
      <SpanText text={span.text} lead={span.lead_text} highlights={span.highlights} />
    </article>
  );
}

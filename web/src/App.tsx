import { useEffect, useMemo, useRef, useState } from "react";
import { api, type CandidateRow, type EvidenceSpan, type NodeDetail, type NodeSummary } from "./api";
import { ErrorBox, NodeName } from "./Common";
import { ContextModal } from "./ContextModal";
import { EdgeQueue } from "./EdgeQueue";
import { EgoGraph } from "./EgoGraph";
import {
  candidatesHref,
  href,
  queueHref,
  reviewHref,
  setTheme,
  useFetch,
  useRoute,
  useTheme,
  type NodeTab,
} from "./hooks";
import { Figures, NodeHeader, SpanText } from "./NodeParts";
import { RelationsView } from "./Relations";
import { ReviewPage } from "./Review";
import {
  CANDIDATE_STATUS,
  CUE,
  RELATION,
  RELATION_COLOR,
  SECTION,
  SECTOR_KO,
  SOURCE,
  excludedLabel,
  sectorColor,
} from "./labels";

export function App() {
  const route = useRoute();
  const meta = useFetch("meta", api.meta);
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          <BrandMark />
          <span>
            10-K 관계도
            <small>S&amp;P 500 기업 관계 지도</small>
          </span>
        </a>
        <SearchBox />
        <nav className="nav">
          <a href="#/" className={!route.review && !route.queue ? "on" : ""}>관계도</a>
          <a href={queueHref()} className={route.queue ? "on" : ""}>관계 검수</a>
          <a href={reviewHref()} className={route.review ? "on" : ""}>표본 검수</a>
        </nav>
        {meta.data && (
          <span className="topbar-meta" title={`후보: ${meta.data.similarity} 상위 ${meta.data.top_k}`}>
            <span className="pill">{meta.data.filings_year}년 10-K</span>
            <span className="pill">판정 {judgeModel(meta.data)}</span>
          </span>
        )}
        <ThemeButton />
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
        <Home meta={meta.data} />
      )}
    </div>
  );
}

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="url(#brand-g)" />
      <defs>
        <linearGradient id="brand-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#3257e6" />
          <stop offset="1" stopColor="#7c5cf5" />
        </linearGradient>
      </defs>
      <g stroke="#fff" strokeWidth="1.8" opacity="0.7">
        <path d="M16 16 8.5 9.5M16 16l8.5-6M16 16l-7 7.5M16 16l7.5 7" />
      </g>
      <circle cx="16" cy="16" r="4.2" fill="#fff" />
      <circle cx="8.5" cy="9.5" r="2.4" fill="#ff9a9d" />
      <circle cx="24.5" cy="10" r="2.4" fill="#ff9a9d" />
      <circle cx="9" cy="23.5" r="2.4" fill="#6ee7d3" />
      <circle cx="23.5" cy="23" r="2.4" fill="#6ee7d3" />
    </svg>
  );
}

function ThemeButton() {
  const theme = useTheme();
  const dark = theme === "dark";
  return (
    <button
      className="icon-btn"
      onClick={() => setTheme(dark ? "light" : "dark")}
      title={dark ? "밝은 화면" : "어두운 화면"}
      aria-label={dark ? "밝은 화면으로" : "어두운 화면으로"}
    >
      <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {dark ? (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </>
        ) : (
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        )}
      </svg>
    </button>
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
  // '/'를 누르면 검색창으로 (입력 중일 때는 빼고)
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.key !== "/" || el.tagName === "INPUT" || el.tagName === "TEXTAREA") return;
      e.preventDefault();
      input.current?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="search">
      <svg className="search-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      {!q && <kbd className="search-kbd">/</kbd>}
      <input
        ref={input}
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

/** meta의 JSON 값 하나. 없거나 깨졌으면 빈 객체. */
function metaJson<T>(meta: Record<string, string> | undefined, key: string): Partial<T> {
  try {
    return JSON.parse(meta?.[key] ?? "{}");
  } catch {
    return {};
  }
}

type EdgeCounts = Record<string, { accepted?: number; uncertain?: number }>;

const PREVIEW_TICKER = "NVDA"; // 첫 화면 미리보기 회사 (없으면 관계가 가장 많은 회사)

const STEPS: [string, string][] = [
  ["10-K 수집", "EDGAR에서 S&P 500 회사의 10-K를 받아 Item 1(사업)과 1A(위험 요인)를 뽑습니다."],
  ["후보 찾기", "다른 회사 이름이 나온 문장과, 사업 설명이 비슷한 회사(TF-IDF + 임베딩) 상위 20곳을 후보로 둡니다."],
  ["관계 판정", "판정 모델(Jev)이 근거 문장마다 경쟁, 공급·협력, 지분 관계인지 점수로 답하고, 기준을 넘으면 채택합니다."],
  ["사람 검수", "애매한 관계는 검수 대기로 남겨 사람이 근거 문장을 읽고 정합니다."],
];

function Home({ meta }: { meta?: Record<string, string> }) {
  const list = useFetch("home", () => api.search(""));
  const edges = metaJson<EdgeCounts>(meta, "edges");
  const units = metaJson<{ total: number }>(meta, "units");
  const count = (r: string) => edges[r]?.accepted ?? 0;
  const uncertain = Object.values(edges).reduce((s, e) => s + (e?.uncertain ?? 0), 0);
  const preview = list.data?.find((n) => n.ticker === PREVIEW_TICKER) ?? list.data?.[0];
  const max = Math.max(1, ...(list.data ?? []).map((n) => n.n_relations));
  const fmt = (n: number) => n.toLocaleString("ko-KR");

  return (
    <main className="page home">
      <section className="hero">
        <div className="hero-text">
          <span className="eyebrow">S&amp;P 500 · {meta?.filings_year ?? ""}년 제출 10-K</span>
          <h1>
            공시 문장으로 확인한
            <br />
            <em>기업 관계 지도</em>
          </h1>
          <p className="lede">
            10-K(Item 1·1A)에 적힌 문장에서 두 회사의 관계를 찾았습니다. 관계는 <b>경쟁</b>, <b>공급·협력</b>,{" "}
            <b>지분</b> 세 가지이고, 모든 관계에는 근거 문장이 붙어 있습니다. 회사를 검색하거나 아래에서 고르세요.
          </p>
          <div className="stat-row">
            {(["competitor", "business", "equity"] as const).map((r) => (
              <div key={r} className="stat">
                <b>{fmt(count(r))}</b>
                <span>
                  <i style={{ background: RELATION_COLOR[r] }} />
                  {RELATION[r]}
                  {r === "equity" && " (검증 전)"}
                </span>
              </div>
            ))}
            <div className="stat">
              <b>{fmt(units.total ?? 0)}</b>
              <span>판정한 문장</span>
            </div>
          </div>
          <p className="small muted">
            판정 모델이 채택한 관계 수입니다. 검수 대기 {fmt(uncertain)}개는 따로 셉니다.
          </p>
        </div>
        <div className="hero-preview">
          {preview ? <Preview node={preview} /> : <div className="preview-empty" />}
        </div>
      </section>

      <section className="steps">
        {STEPS.map(([title, text], i) => (
          <div key={title} className="step">
            <span className="step-no">{i + 1}</span>
            <b>{title}</b>
            <p>{text}</p>
          </div>
        ))}
      </section>

      <h2 className="section-title">관계가 많은 회사</h2>
      {list.error && <ErrorBox message={list.error} />}
      <div className="card-grid co-grid">
        {list.data?.map((n) => (
          <a
            key={n.node_id}
            className="co-card"
            href={href(n.node_id)}
            style={{ "--c": sectorColor(n.gics_sector) } as React.CSSProperties}
          >
            <span className="co-top">
              <span className="co-name">
                <b>{n.name}</b>
                {n.ticker && <span className="ticker">{n.ticker}</span>}
              </span>
              <span className="co-count">
                {n.n_relations}
                <small>관계</small>
              </span>
            </span>
            <span className="co-sector">{SECTOR_KO[n.gics_sector ?? ""] ?? n.gics_sector ?? "분석 대상 밖"}</span>
            <span className="co-bar">
              <i style={{ width: `${(100 * n.n_relations) / max}%` }} />
            </span>
          </a>
        ))}
      </div>
    </main>
  );
}

/** 첫 화면의 관계도 미리보기 (채택·확인된 관계만). 점을 누르면 그 관계를 연다. */
function Preview({ node }: { node: NodeSummary }) {
  const id = node.node_id;
  const detail = useFetch(`node:${id}`, () => api.node(id));
  const rels = useFetch(`rels:${id}`, () => api.relations(id));
  const shown = useMemo(
    () => (rels.data ?? []).filter((r) => r.state === "accepted" || r.state === "confirmed"),
    [rels.data],
  );
  return (
    <>
      <header>
        <NodeName name={node.name} ticker={node.ticker} sector={node.gics_sector} kind={node.kind} />
        <span className="muted small">관계 {shown.length}</span>
        <a className="small" href={href(id)}>
          관계도 열기 →
        </a>
      </header>
      {detail.data && rels.data ? (
        <EgoGraph
          compact
          center={detail.data}
          relations={shown}
          selectedEdge={null}
          onSelectEdge={(e) => (window.location.hash = href(id, e))}
          onOpenNode={(n) => (window.location.hash = href(n))}
        />
      ) : (
        <div className="preview-empty" />
      )}
      <footer>
        {(["competitor", "business", "equity"] as const).map((r) => (
          <span key={r}>
            <i style={{ background: RELATION_COLOR[r] }} /> {RELATION[r]}
          </span>
        ))}
        <span className="muted">점 색은 GICS 섹터 · 점에 마우스를 올려 보세요</span>
      </footer>
    </>
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
        <span className="pair-link" role="img" aria-label="↔" />
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

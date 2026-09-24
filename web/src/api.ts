// FastAPI(src/tenksim/app/server.py)와 주고받는 형식. node_id·pair_key는 경로에 넣을 때 인코딩한다.

export type NodeKind = "company" | "external" | "anonymous";

export interface NodeSummary {
  node_id: string;
  kind: NodeKind;
  ticker: string | null;
  name: string;
  gics_sector: string | null;
  analyzed: number;
  n_candidates: number;
  n_mention_candidates: number;
}

export interface Filing {
  accession: string;
  form: string;
  filing_date: string;
  period_of_report: string;
  filing_url: string;
  sections: string | null;
}

export interface NodeDetail {
  node_id: string;
  kind: NodeKind;
  cik: number | null;
  ticker: string | null;
  name: string;
  gics_sector: string | null;
  gics_sub_industry: string | null;
  analyzed: number;
  parent_node: string | null;
  parent?: { node_id: string; ticker: string | null; name: string };
  filings: Filing[];
}

export type Source = "similarity" | "mention" | "both";

export interface CandidateRow {
  pair_key: string;
  source: Source;
  status: string;
  similarity_pct: number | null;
  n_spans: number;
  other_id: string;
  other_kind: NodeKind;
  other_ticker: string | null;
  other_name: string;
  other_sector: string | null;
  rank_mine: number | null;
  rank_theirs: number | null;
  mentions_out: number;
  mentions_in: number;
}

export interface Highlight {
  start: number;
  end: number;
  node_id: string;
  excluded: string | null;
}

export interface EvidenceSpan {
  span_id: string;
  doc_node: string;
  accession: string;
  section: string;
  text: string;
  lead_text: string | null;
  filing_date: string | null;
  filing_url: string | null;
  input_order: number | null;
  cues: string | null;
  excluded: string | null;
  highlights: Highlight[];
}

export interface Figure {
  figure_id: string;
  span_id: string;
  target_node: string;
  subject: string;
  value: number | null;
  operator: string | null;
  period: string | null;
  denominator: string | null;
  raw: string;
}

export interface PairNode {
  node_id: string;
  kind: NodeKind;
  ticker: string | null;
  name: string;
  gics_sector: string | null;
  gics_sub_industry: string | null;
}

export interface CandidateDetail {
  pair_key: string;
  node_a: string;
  node_b: string;
  source: Source;
  status: string;
  rank_ab: number | null;
  rank_ba: number | null;
  similarity_pct: number | null;
  mentions_ab: number;
  mentions_ba: number;
  n_spans: number;
  a: PairNode;
  b: PairNode;
  spans: EvidenceSpan[];
  figures: Figure[];
}

export interface Mark {
  start: number;
  end: number;
  kind: "span" | "lead" | "name";
}

export interface SpanContext {
  span_id: string;
  section: string;
  name: string;
  ticker: string | null;
  filing_url: string;
  filing_date: string;
  status: string;
  excerpt: string;
  offset: number;
  doc_length: number;
  marks: Mark[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* 본문이 JSON이 아니면 상태 문구만 */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

const enc = encodeURIComponent;

export const api = {
  meta: () => get<Record<string, string>>("/api/meta"),
  search: (q: string) => get<NodeSummary[]>(`/api/nodes?q=${enc(q)}&limit=60`),
  node: (id: string) => get<NodeDetail>(`/api/nodes/${enc(id)}`),
  candidates: (id: string) => get<CandidateRow[]>(`/api/nodes/${enc(id)}/candidates`),
  candidate: (key: string) => get<CandidateDetail>(`/api/candidates/${enc(key)}`),
  context: (spanId: string) => get<SpanContext>(`/api/spans/${enc(spanId)}/context`),
};

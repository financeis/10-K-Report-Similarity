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
  n_relations: number;
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
  doc_node?: string; // 관계 화면에서만: 수치가 나온 10-K의 회사
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
  edges: { edge_id: string; relation: string; state: string }[];
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

// ---------------------------------------------------------------- 관계 (2단계)

/** 방향 없는 관계 세 가지 (계획서 5장). */
export type RelationType = "competitor" | "business" | "equity";

/** 화면에 쓰는 최종 상태: 사람 확인 > 모델 채택 > 검수 대기, 사람이 거절한 것은 따로. */
export type EdgeState = "confirmed" | "accepted" | "uncertain" | "rejected";

export interface EdgeBase {
  edge_id: string;
  src: string;
  dst: string;
  relation: RelationType;
  decision: "accepted" | "uncertain";
  validated: number;
  status: "current" | "historical" | "planned" | "unclear" | null;
  score: number | null;
  n_evidence: number;
  similarity_pct: number | null;
  model_id: string | null;
  question_version: string | null;
  review_state: "accepted" | "rejected" | "needs_recheck" | null;
  review?: { verdict: string; note: string | null; reviewed_at: string };
  state: EdgeState;
}

export interface NodeRelation extends EdgeBase {
  other_id: string;
  other_kind: NodeKind;
  other_ticker: string | null;
  other_name: string;
  other_sector: string | null;
}

export interface EdgeEvidence {
  span_id: string;
  target_node: string;
  doc_node: string;
  section: string;
  text: string;
  lead_text: string | null;
  filing_date: string | null;
  filing_url: string | null;
  score: number | null;
  s_is_entity: number | null;
  unit_status: string | null;
  ord: number;
  highlights: Highlight[];
}

export interface EdgeDetail extends EdgeBase {
  a: PairNode;
  b: PairNode;
  evidence_hash: string;
  /** 질문별 [채택, 기각] 임계값 (graph.db를 만들 때 쓴 값) */
  thresholds: Record<string, [number, number]>;
  candidate: { rank_ab: number | null; rank_ba: number | null; source: Source } | null;
  evidence: EdgeEvidence[];
  figures: Figure[];
}

export interface QueueEdge extends EdgeBase {
  a_name: string;
  a_ticker: string | null;
  a_sector: string | null;
  a_kind: NodeKind;
  b_name: string;
  b_ticker: string | null;
  b_sector: string | null;
  b_kind: NodeKind;
}

export const api = {
  meta: () => get<Record<string, string>>("/api/meta"),
  search: (q: string) => get<NodeSummary[]>(`/api/nodes?q=${enc(q)}&limit=60`),
  node: (id: string) => get<NodeDetail>(`/api/nodes/${enc(id)}`),
  candidates: (id: string) => get<CandidateRow[]>(`/api/nodes/${enc(id)}/candidates`),
  candidate: (key: string) => get<CandidateDetail>(`/api/candidates/${enc(key)}`),
  context: (spanId: string) => get<SpanContext>(`/api/spans/${enc(spanId)}/context`),
  relations: (id: string) => get<NodeRelation[]>(`/api/nodes/${enc(id)}/relations`),
  edge: (id: string) => get<EdgeDetail>(`/api/edges/${enc(id)}`),
};

// ---------------------------------------------------------------- 표본 검수

export interface SampleProgress {
  sample_id: string;
  purpose: "dev" | "confirm";
  created_at: string;
  n_companies: number;
  n_units: number;
  n_labeled: number;
  n_skipped: number | null;
}

export interface ReviewNode {
  node_id: string;
  kind: NodeKind;
  ticker: string | null;
  name: string;
  gics_sector: string | null;
}

export interface SampleUnitRow {
  ord: number;
  unit_id: string;
  doc_node: string;
  target_node: string;
  pair_key: string;
  labeled: number;
  skipped: number;
}

export interface SampleDetail {
  sample: { sample_id: string; purpose: string; created_at: string; per_company_cap: number };
  companies: { node_id: string; n_units_total: number; node: ReviewNode | null }[];
  units: SampleUnitRow[];
  nodes: Record<string, ReviewNode>;
}

// 검수 라벨 v2: 방향·세부 유형 없이 세 가지 (계획서 5장)
export type Relation = "competitor" | "business" | "equity";

export interface SpanLabel {
  is_entity: "yes" | "no" | "unsure";
  relations: string[]; // v1 라벨(dev1)에는 방향이 있는 옛 코드가 들어 있다
  skipped: number;
  note: string | null;
  labeled_at: string;
}

export interface ReviewUnit {
  sample_id: string;
  ord: number;
  n_units: number;
  unit_id: string;
  doc: ReviewNode;
  target: ReviewNode;
  span_id: string;
  section: string;
  filing_date: string | null;
  filing_url: string | null;
  text: string;
  lead_text: string | null;
  highlights: { start: number; end: number }[];
  changed: boolean;
  label: SpanLabel | null;
}

export interface LabelIn {
  unit_id: string;
  sample_id: string;
  is_entity: string;
  relations: Relation[];
  skipped: boolean;
  note: string | null;
}

/** 서버가 거절한 요청. status로 종류를 가린다 (409: 그 사이 근거가 바뀜 등). */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* 본문이 JSON이 아니면 상태 문구만 */
    }
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return res.json() as Promise<T>;
}

export interface EdgeReviewIn {
  edge_id: string;
  verdict: "accept" | "reject";
  note: string | null;
  evidence_hash: string; // 검수자가 본 근거. 그 사이 근거가 바뀌었으면 서버가 409로 거절한다
}

export const reviewApi = {
  edgeQueue: (includeDone: boolean) => get<QueueEdge[]>(`/api/review/edges?include_done=${includeDone}`),
  saveEdge: (body: EdgeReviewIn) => post<{ review_id: number; edge: EdgeBase }>("/api/review/edges", body),
  samples: () => get<SampleProgress[]>("/api/review/samples"),
  sample: (id: string) => get<SampleDetail>(`/api/review/samples/${enc(id)}`),
  unit: (id: string, ord: number) => get<ReviewUnit>(`/api/review/samples/${enc(id)}/units/${ord}`),
  save: (label: LabelIn) => post<{ label_id: number; label: SpanLabel }>("/api/review/labels", label),
};

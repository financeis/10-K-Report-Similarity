// 화면에 쓰는 한국어 표기와 색.

export const SECTION: Record<string, string> = {
  business: "Item 1 사업",
  risk_factors: "Item 1A 위험 요인",
};

export const SOURCE: Record<string, string> = {
  similarity: "유사도",
  mention: "언급",
  both: "유사도+언급",
};

export const CUE: Record<string, string> = {
  competition: "경쟁",
  supply: "고객·공급",
  partnership: "협력",
  ownership: "지분",
};

export const EXCLUDED: Record<string, string> = {
  exec_bio: "임원 약력",
  "context:listing": "상장 거래소 표기",
  "context:index": "지수 언급",
  "context:credit_rating": "신용등급",
  "context:esg_rating": "ESG·지속가능성 평가",
  "context:trademark": "상표 문구",
  "context:other_meaning": "다른 뜻(제품명 등)",
  "context:other_company": "이름이 같은 다른 회사",
  "context:performance_graph": "주가 비교 그래프",
};

export const KIND: Record<string, string> = {
  company: "",
  external: "분석 대상 밖",
  anonymous: "익명 공시",
};

export const OPERATOR: Record<string, string> = { "=": "", ">=": "≥", "~": "약 ", "<": "<" };

// GICS 섹터 색. 섹터를 넘는 연결이 눈에 띄게 하려는 것 (계획서 8.2).
export const SECTOR_COLOR: Record<string, string> = {
  "Information Technology": "#4263eb",
  "Communication Services": "#8e5bd8",
  "Consumer Discretionary": "#f08c3a",
  "Consumer Staples": "#74b243",
  "Health Care": "#e0547a",
  Financials: "#1c9fb0",
  Industrials: "#7384a8",
  Energy: "#b8662b",
  Materials: "#b39b2e",
  Utilities: "#4fa3d9",
  "Real Estate": "#c05aa8",
};

export const SECTOR_KO: Record<string, string> = {
  "Information Technology": "정보기술",
  "Communication Services": "커뮤니케이션",
  "Consumer Discretionary": "경기소비재",
  "Consumer Staples": "필수소비재",
  "Health Care": "헬스케어",
  Financials: "금융",
  Industrials: "산업재",
  Energy: "에너지",
  Materials: "소재",
  Utilities: "유틸리티",
  "Real Estate": "부동산",
};

/** 분석 대상 밖 회사(섹터 모름)의 점 색. */
export const EXTERNAL_COLOR = "#9aa0a6";

export function sectorColor(sector: string | null): string {
  return (sector && SECTOR_COLOR[sector]) || "var(--muted-2)";
}

export function excludedLabel(reason: string | null): string {
  if (!reason) return "";
  return reason
    .split(",")
    .map((r) => EXCLUDED[r] ?? r)
    .join(", ");
}

// ---------------------------------------------------------------- 관계 (2단계)

export const RELATION: Record<string, string> = {
  competitor: "경쟁",
  business: "공급·협력",
  equity: "지분",
};

export const RELATION_HINT: Record<string, string> = {
  competitor: "서로 경쟁하거나, 한쪽이 다른 쪽을 경쟁사로 적음",
  business: "제품·서비스를 사고팔거나 제휴·합작·라이선스·유통 관계 (방향은 따지지 않음)",
  equity: "한쪽이 다른 쪽의 주식·지분을 가지고 있거나 가졌음 (모회사·분사 포함)",
};

// 관계 선 색. 섹터 색(점)과 겹치지 않게 채도가 다른 색을 쓴다.
export const RELATION_COLOR: Record<string, string> = {
  competitor: "#e5484d",
  business: "#12a594",
  equity: "#f5a524",
};

export const RELATIONS = ["competitor", "business", "equity"] as const;

export const STATUS: Record<string, string> = {
  current: "현재",
  historical: "과거",
  planned: "계획",
  unclear: "시점 불명",
};

export const STATE: Record<string, string> = {
  confirmed: "검수로 확인",
  accepted: "채택",
  uncertain: "검수 대기",
  rejected: "검수로 거절",
};

export const CANDIDATE_STATUS: Record<string, string> = {
  accepted: "관계 채택",
  uncertain: "검수 대기",
  rejected: "관계 없음 (판정)",
  rejected_by_review: "검수로 거절",
  similar: "유사도만 (이름 언급 없음)",
  pending: "판정 전",
};

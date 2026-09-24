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
  "Information Technology": "#3b6fd8",
  "Communication Services": "#8a5cd6",
  "Consumer Discretionary": "#d9822b",
  "Consumer Staples": "#6f9a3a",
  "Health Care": "#d0485f",
  Financials: "#2f9e8f",
  Industrials: "#5b6fa3",
  Energy: "#b5651d",
  Materials: "#a08a2c",
  Utilities: "#4f8fbf",
  "Real Estate": "#b3569f",
};

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

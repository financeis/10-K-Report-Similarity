// 회사 화면과 관계 화면이 함께 쓰는 조각: 회사 머리글, 근거 문장 카드, 매출 비중.

import type { Figure, NodeDetail } from "./api";
import { Highlighted, type Range } from "./Highlighted";
import { href } from "./hooks";
import { KIND, OPERATOR, SECTION, SECTOR_KO, sectorColor } from "./labels";

export function NodeHeader({ node }: { node: NodeDetail }) {
  const color = sectorColor(node.gics_sector);
  return (
    <div className="node-header">
      <h1>
        <span className="node-avatar" style={{ background: color }}>
          {(node.ticker ?? node.name.split(/\s+/).map((w) => w[0]).join("")).slice(0, 4).toUpperCase()}
        </span>
        <span className="node-title">
          {node.name}
          {node.ticker && <span className="ticker-chip">{node.ticker}</span>}
          {KIND[node.kind] && <span className="tag">{KIND[node.kind]}</span>}
        </span>
      </h1>
      <p className="node-sub">
        {node.gics_sector ? (
          <>
            <span className="sector-pill" style={{ "--c": color } as React.CSSProperties}>
              {SECTOR_KO[node.gics_sector] ?? node.gics_sector}
            </span>
            <span className="muted">{node.gics_sub_industry}</span>
          </>
        ) : (
          <span className="muted">GICS 분류 없음</span>
        )}
        {node.parent && (
          <span className="muted">
            {"공시한 회사 "}
            <a href={href(node.parent.node_id)}>{node.parent.name}</a>
          </span>
        )}
      </p>
      {node.filings.map((f) => (
        <p key={f.accession} className="small muted filing">
          <a href={f.filing_url} target="_blank" rel="noreferrer">
            {f.form} · 제출 {f.filing_date} · 회계기간 {f.period_of_report} ↗
          </a>
          <span>
            {"분석한 항목: "}
            {(f.sections ?? "").split(",").map((s) => SECTION[s] ?? s).join(", ")}
          </span>
        </p>
      ))}
      {!node.analyzed && node.kind === "company" && (
        <p className="small warn">이 회사의 10-K는 추출 품질 검사를 통과하지 못해, 다른 회사 10-K의 언급만 보입니다.</p>
      )}
    </div>
  );
}

/** 근거 문장 본문: 목록 도입문 + 문장, 회사 이름 강조. */
export function SpanText({
  text,
  lead,
  highlights,
}: {
  text: string;
  lead: string | null;
  highlights: { start: number; end: number }[];
}) {
  const ranges: Range[] = highlights.map((h) => ({ start: h.start, end: h.end, className: "name" }));
  return (
    <>
      {lead && <p className="lead">{lead} …</p>}
      <p className="span-text">
        <Highlighted text={text} ranges={ranges} />
      </p>
    </>
  );
}

export function Figures({
  figures,
  docOf,
  names,
  note,
}: {
  figures: Figure[];
  docOf: (f: Figure) => string | undefined;
  names: Record<string, string>;
  note: string;
}) {
  // 같은 문장이 Item 1과 1A에 반복되면 같은 수치가 두 번 나온다
  const seen = new Set<string>();
  const unique = figures.filter((f) => {
    const key = `${docOf(f)}|${f.target_node}|${f.value}|${f.operator}|${f.period}|${f.raw}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!unique.length) return null;
  return (
    <section className="figures">
      <h3>매출 비중 후보</h3>
      <ul>
        {unique.map((f) => {
          const doc = docOf(f);
          return (
            <li key={f.figure_id}>
              {f.value != null ? (
                <>
                  <b>{(doc && names[doc]) ?? "?"}</b> 매출에서 <b>{names[f.target_node] ?? f.target_node}</b>{" "}
                  {f.subject === "each" ? "(각각) " : ""}
                  <b>
                    {OPERATOR[f.operator ?? "="]}
                    {f.value}%
                  </b>
                  <span className="muted"> ({[f.period, f.denominator].filter(Boolean).join(", ")})</span>
                </>
              ) : (
                <>
                  <span className="muted">귀속이 불분명해 수치를 비움:</span> “{f.raw}”
                </>
              )}
            </li>
          );
        })}
      </ul>
      <p className="small muted">{note}</p>
    </section>
  );
}

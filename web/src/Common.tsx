import { KIND, sectorColor } from "./labels";

// ---------------------------------------------------------------- 공통

export function NodeName({
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

export function ErrorBox({ message }: { message: string }) {
  return <p className="error">불러오지 못했습니다: {message}</p>;
}

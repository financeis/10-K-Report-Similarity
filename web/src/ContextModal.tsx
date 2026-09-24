import { useEffect, useRef } from "react";
import { api } from "./api";
import { ErrorBox } from "./Common";
import { Highlighted, type Range } from "./Highlighted";
import { useFetch } from "./hooks";
import { SECTION } from "./labels";

/** 근거 구간을 정제 본문 앞뒤 문맥 안에서 강조해 보여준다. */
export function ContextModal({ spanId, onClose }: { spanId: string; onClose: () => void }) {
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

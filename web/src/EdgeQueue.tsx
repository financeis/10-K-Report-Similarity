// 관계 검수 (운영 검수, 계획서 8.1의 4): 판정 모델이 불확실로 남긴 관계를 사람이 맞음/아님으로 정한다.
// 표본 검수와 달리 모델 판정(점수)을 보여준다. 결과는 reviews.sqlite에 쌓이고 관계도에 바로 반영된다.

import { useEffect, useRef, useState } from "react";
import { reviewApi } from "./api";
import { ErrorBox, NodeName } from "./Common";
import { queueHref, useFetch } from "./hooks";
import { EdgePanel, EdgeTags, RelationBadge } from "./Relations";

export function EdgeQueue({ edgeId }: { edgeId: string | null }) {
  const [version, setVersion] = useState(0);
  const [showDone, setShowDone] = useState(false);
  const queue = useFetch(`queue:${showDone}:${version}`, () => reviewApi.edgeQueue(showDone));
  const list = queue.data ?? [];
  const current = edgeId && list.some((e) => e.edge_id === edgeId) ? edgeId : (list[0]?.edge_id ?? edgeId);
  const idx = list.findIndex((e) => e.edge_id === current);

  const go = (i: number) => {
    if (i >= 0 && i < list.length) window.location.hash = queueHref(list[i].edge_id);
  };

  // 키보드: A 맞음, R 아님, J 다음, K 이전.
  // - 키를 누르고 있어 반복 입력되면 무시한다 (다음 관계를 읽지도 않고 저장되지 않게)
  // - 본문 보기 창이 열려 있으면 쓰지 않는다 (창 뒤의 관계에 저장되지 않게)
  // - 화살표는 목록·본문을 스크롤하는 데 두고 빼앗지 않는다
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (document.querySelector(".modal-backdrop")) return;
      const k = e.key.toLowerCase();
      if (k === "a" || k === "r") {
        if (e.repeat) return;
        const verdict = k === "a" ? "accept" : "reject";
        document.querySelector<HTMLButtonElement>(`.edge-review button[data-verdict="${verdict}"]`)?.click();
      } else if (k === "j") go(idx + 1);
      else if (k === "k") go(idx - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // 고른 관계가 목록에서 보이게 스크롤한다
  const listRef = useRef<HTMLUListElement>(null);
  useEffect(() => {
    listRef.current?.querySelector("li.sel")?.scrollIntoView({ block: "nearest" });
  }, [current, list.length]);

  const onReviewed = () => {
    // 검수한 관계는 목록에서 빠지므로, 바로 다음 관계로 넘어간다
    const next = list[idx + 1] ?? list[idx - 1];
    setVersion((v) => v + 1);
    if (next && !showDone) window.location.hash = queueHref(next.edge_id);
  };

  return (
    <main className="split">
      <section className="left">
        <div className="node-header">
          <h1>관계 검수</h1>
          <p className="muted small">
            판정 모델이 채택도 기각도 하지 못한 관계입니다. 근거 문장을 읽고 맞음·아님을 고르면 관계도에 바로 반영됩니다.
            검수 기록은 reviews.sqlite에 쌓이고, 근거 문장이 바뀌면 다시 검수 대기열에 올라옵니다.
          </p>
        </div>
        <div className="chips">
          <span className="small muted">{showDone ? `불확실 관계 ${list.length}개` : `남은 관계 ${list.length}개`}</span>
          <button className={showDone ? "chip on" : "chip"} onClick={() => setShowDone((s) => !s)} aria-pressed={showDone}>
            검수한 것도 보기
          </button>
        </div>
        {queue.error && <ErrorBox message={queue.error} />}
        {queue.data && list.length === 0 && <p className="note pad-box">검수할 관계가 없습니다.</p>}
        <ul className="queue-list" ref={listRef}>
          {list.map((e) => (
            <li key={e.edge_id} className={e.edge_id === current ? "sel" : ""}>
              <a href={queueHref(e.edge_id)}>
                <span className="queue-pair">
                  <NodeName name={e.a_name} ticker={e.a_ticker} sector={e.a_sector} kind={e.a_kind} />
                  <span className="muted">↔</span>
                  <NodeName name={e.b_name} ticker={e.b_ticker} sector={e.b_sector} kind={e.b_kind} />
                </span>
                <span className="rel-meta">
                  <RelationBadge relation={e.relation} />
                  <EdgeTags edge={e} />
                  <span className="muted small">
                    점수 {e.score?.toFixed(2) ?? "–"} · 근거 {e.n_evidence}
                  </span>
                </span>
              </a>
            </li>
          ))}
        </ul>
      </section>
      <section className="right">
        {current ? (
          <>
            <p className="small muted">
              {idx >= 0 ? `${idx + 1} / ${list.length}` : ""} · 단축키: A 맞음 · R 아님 · J/K 다음·이전
            </p>
            <EdgePanel key={`${current}:${version}`} edgeId={current} reviewable onReviewed={onReviewed} />
          </>
        ) : (
          !queue.loading && <p className="muted pad">고른 관계가 없습니다.</p>
        )}
      </section>
    </main>
  );
}

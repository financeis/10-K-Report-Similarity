// 회사의 관계 화면 (2단계): 관계도 + 관계 목록 + 고른 관계의 근거 문장.
// 관계는 경쟁 · 공급·협력 · 지분 세 가지이고 방향이 없다 (계획서 5장).

import { useCallback, useMemo, useState } from "react";
import { ApiError, api, reviewApi, type EdgeDetail, type NodeDetail, type NodeRelation, type RelationType } from "./api";
import { ErrorBox, NodeName } from "./Common";
import { ContextModal } from "./ContextModal";
import { EgoGraph } from "./EgoGraph";
import { href, queueHref, useFetch } from "./hooks";
import { Figures, SpanText } from "./NodeParts";
import { RELATION, RELATION_COLOR, RELATION_HINT, RELATIONS, SECTION, STATE, STATUS } from "./labels";

// ---------------------------------------------------------------- 표시 조각

export function RelationBadge({ relation }: { relation: string }) {
  return (
    <span className="badge rel" style={{ color: RELATION_COLOR[relation] }} title={RELATION_HINT[relation]}>
      {RELATION[relation] ?? relation}
    </span>
  );
}

/** 관계에 붙는 표시: 검증 전(유형이 확인 표본에서 검증되지 않음), 과거, 검수 상태. */
export function EdgeTags({ edge }: { edge: Pick<NodeRelation, "validated" | "status" | "state" | "review_state"> }) {
  return (
    <>
      {!edge.validated && (
        <span className="tag warn" title="이 관계 유형은 확인 표본에서 아직 검증하지 못했습니다">
          검증 전
        </span>
      )}
      {edge.status && edge.status !== "current" && <span className="tag">{STATUS[edge.status]}</span>}
      {edge.state !== "accepted" && (
        <span className={edge.state === "confirmed" ? "tag ok" : edge.state === "rejected" ? "tag bad" : "tag warn"}>
          {STATE[edge.state]}
        </span>
      )}
      {edge.review_state === "needs_recheck" && (
        <span className="tag warn" title="검수한 뒤 근거 문장이 바뀌어 다시 확인해야 합니다">
          재검수 필요
        </span>
      )}
    </>
  );
}

// ---------------------------------------------------------------- 관계 화면

type Filters = { types: Set<RelationType>; uncertain: boolean; cross: boolean };

export function RelationsView({
  node,
  edgeId,
  header,
}: {
  node: NodeDetail;
  edgeId: string | null;
  header: React.ReactNode;
}) {
  const rels = useFetch(`rels:${node.node_id}`, () => api.relations(node.node_id));
  const [f, setF] = useState<Filters>({ types: new Set(RELATIONS), uncertain: true, cross: false });
  // 섹터를 아는 두 회사끼리만 비교한다 (분석 대상 밖·익명 회사는 섹터를 모름)
  const crossSector = useCallback(
    (r: NodeRelation) => !!node.gics_sector && !!r.other_sector && r.other_sector !== node.gics_sector,
    [node.gics_sector],
  );

  const all = useMemo(() => (rels.data ?? []).filter((r) => r.state !== "rejected"), [rels.data]);
  const shown = useMemo(
    () =>
      all.filter(
        (r) =>
          f.types.has(r.relation) &&
          (f.uncertain || r.state !== "uncertain") &&
          (!f.cross || crossSector(r)),
      ),
    [all, f, crossSector],
  );
  // 유형 칩의 숫자 = 다른 거르기(검수 대기·섹터)를 적용했을 때 그 유형의 관계 수 (아래 목록 머리글과 같다)
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const r of all)
      if ((f.uncertain || r.state !== "uncertain") && (!f.cross || crossSector(r))) c[r.relation] = (c[r.relation] ?? 0) + 1;
    return c;
  }, [all, f.uncertain, f.cross, crossSector]);
  const nUncertain = all.filter((r) => r.state === "uncertain" && f.types.has(r.relation) && (!f.cross || crossSector(r))).length;
  const selected = edgeId ?? shown.find((r) => r.state !== "uncertain")?.edge_id ?? shown[0]?.edge_id ?? null;

  const toggle = (t: RelationType) =>
    setF((s) => {
      const types = new Set(s.types);
      if (types.has(t)) types.delete(t);
      else types.add(t);
      return { ...s, types };
    });

  if (rels.error) return <ErrorBox message={rels.error} />;
  return (
    <>
      <section className="left">
        {header}
        <div className="chips" role="group" aria-label="관계 거르기">
          {RELATIONS.map((t) => (
            <button key={t} className={f.types.has(t) ? "chip on" : "chip"} onClick={() => toggle(t)} aria-pressed={f.types.has(t)}>
              <i className="swatch" style={{ background: RELATION_COLOR[t] }} /> {RELATION[t]}{" "}
              <span className="count">{counts[t] ?? 0}</span>
            </button>
          ))}
          <button className={f.uncertain ? "chip on" : "chip"} onClick={() => setF((s) => ({ ...s, uncertain: !s.uncertain }))} aria-pressed={f.uncertain}>
            검수 대기 포함 <span className="count">{nUncertain}</span>
          </button>
          <button className={f.cross ? "chip on" : "chip"} onClick={() => setF((s) => ({ ...s, cross: !s.cross }))} aria-pressed={f.cross}>
            섹터가 다른 회사만
          </button>
        </div>
        {rels.loading && !rels.data ? (
          <p className="muted pad">불러오는 중…</p>
        ) : all.length === 0 ? (
          <p className="note pad-box">
            10-K 문장에서 확인한 관계가 없습니다. 사업이 비슷한 기업과 판정 근거는 '후보 전체' 탭에서 볼 수 있습니다.
          </p>
        ) : (
          <>
            <EgoGraph
              center={node}
              relations={shown}
              selectedEdge={selected}
              onSelectEdge={(id) => (window.location.hash = href(node.node_id, id))}
              onOpenNode={(id) => (window.location.hash = href(id))}
            />
            <RelationList relations={shown} nodeId={node.node_id} selected={selected} />
          </>
        )}
      </section>
      <section className="right">
        {selected ? (
          <EdgePanel key={selected} edgeId={selected} viewer={node.node_id} />
        ) : (
          !rels.loading && <p className="muted pad">조건에 맞는 관계가 없습니다.</p>
        )}
      </section>
    </>
  );
}

function RelationList({ relations, nodeId, selected }: { relations: NodeRelation[]; nodeId: string; selected: string | null }) {
  return (
    <div className="rel-list">
      {RELATIONS.map((t) => {
        const rows = relations.filter((r) => r.relation === t);
        if (!rows.length) return null;
        return (
          <section key={t}>
            <h3>
              <RelationBadge relation={t} /> <span className="muted small">{rows.length}곳</span>
            </h3>
            <ul>
              {rows.map((r) => (
                <li key={r.edge_id} className={r.edge_id === selected ? "sel" : ""}>
                  <a href={href(nodeId, r.edge_id)}>
                    <NodeName name={r.other_name} ticker={r.other_ticker} sector={r.other_sector} kind={r.other_kind} />
                    <span className="rel-meta">
                      <EdgeTags edge={r} />
                      <span className="muted small">근거 {r.n_evidence}</span>
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- 관계 하나

export function EdgePanel({
  edgeId,
  viewer,
  reviewable = false,
  onReviewed,
}: {
  edgeId: string;
  viewer?: string;
  reviewable?: boolean;
  onReviewed?: (verdict: "accept" | "reject") => void;
}) {
  const [version, setVersion] = useState(0);
  const edge = useFetch(`edge:${edgeId}:${version}`, () => api.edge(edgeId));
  const [contextSpan, setContextSpan] = useState<string | null>(null);
  if (edge.error) return <ErrorBox message={edge.error} />;
  if (!edge.data) return <p className="muted pad">불러오는 중…</p>;
  const d = edge.data;
  const [me, other] = viewer && d.b.node_id === viewer ? [d.b, d.a] : [d.a, d.b];
  const names = { [d.a.node_id]: d.a.name, [d.b.node_id]: d.b.name };
  // 회사 식별이 불확실한 쪽은 문장이 언급한 회사다 (문장이 나온 10-K의 회사가 아니라)
  const doubt = entityDoubt(d);
  const doubtName = doubt ? (names[doubt.target_node] ?? doubt.target_node) : other.name;

  return (
    <div className="pair">
      <h2 className="pair-title">
        <a href={href(me.node_id)}>{me.name}</a>
        <span className="muted"> ↔ </span>
        <a href={href(other.node_id)}>{other.name}</a>
      </h2>
      <div className="edge-head">
        <RelationBadge relation={d.relation} />
        <EdgeTags edge={d} />
        <span className="muted small">{RELATION_HINT[d.relation]}</span>
      </div>
      {!d.validated && (
        <p className="note small">
          지분 관계는 확인 표본에 정답이 없어 판정 정확도를 아직 확인하지 못했습니다. 근거 문장으로 직접 확인해 주세요.
        </p>
      )}
      <dl className="stats">
        <div>
          <dt>판정</dt>
          <dd>
            {STATE[d.state]}
            {d.review && <span className="muted small"> · 검수 {d.review.reviewed_at.slice(0, 10)}</span>}
          </dd>
        </div>
        <div>
          <dt title="근거 문장 가운데 가장 높은 판정 점수 (Jev)">판정 점수</dt>
          <dd>{d.score != null ? d.score.toFixed(2) : "–"}</dd>
        </div>
        <div>
          <dt>시점</dt>
          <dd>{d.status ? STATUS[d.status] : "–"}</dd>
        </div>
        {d.similarity_pct != null && (
          <div>
            <dt>사업 유사도</dt>
            <dd>전체 쌍 중 상위 {Math.max(0.1, 100 - d.similarity_pct).toFixed(1)}%</dd>
          </div>
        )}
      </dl>

      {d.state === "uncertain" && <p className="note small">{uncertainReason(d, doubtName)}</p>}
      {reviewable && (
        <ReviewButtons
          edge={d}
          company={doubtName}
          entityDoubt={!!doubt}
          onSaved={(v) => (setVersion((x) => x + 1), onReviewed?.(v))}
          onStale={() => setVersion((x) => x + 1)}
        />
      )}
      {!reviewable && d.state === "uncertain" && (
        <p className="small">
          <a href={queueHref(d.edge_id)}>관계 검수에서 판단하기 →</a>
        </p>
      )}

      <Figures
        figures={d.figures}
        docOf={(f) => f.doc_node}
        names={names}
        note="문장에서 코드로 뽑은 후보입니다. 원문으로 확인하세요."
      />

      <section className="evidence-group">
        <h3>
          근거 문장 <span className="muted">{d.evidence.length}개 · 점수 높은 순</span>
        </h3>
        {d.evidence.map((ev) => (
          <article key={`${ev.span_id}|${ev.target_node}`} className="span-card">
            <div className="span-meta">
              <b className="doc-name">{names[ev.doc_node] ?? ev.doc_node}</b>
              <span>의 10-K · {SECTION[ev.section] ?? ev.section}</span>
              {ev.filing_date && <span>· 제출 {ev.filing_date}</span>}
              {ev.score != null && (
                <span className="tag" title="이 문장의 관계 점수 (Jev)">
                  점수 {ev.score.toFixed(2)}
                </span>
              )}
              {ev.s_is_entity != null && (
                <span
                  className={ev.s_is_entity < accept(d, "is_entity") ? "tag warn" : "tag"}
                  title="문장 속 강조한 이름이 그 회사가 맞다는 점수 (Jev)"
                >
                  회사 식별 {ev.s_is_entity.toFixed(2)}
                  {ev.s_is_entity < accept(d, "is_entity") ? " · 불확실" : ""}
                </span>
              )}
              {ev.unit_status && ev.unit_status !== "current" && <span className="tag">{STATUS[ev.unit_status]}</span>}
              <button className="link" onClick={() => setContextSpan(ev.span_id)}>
                본문에서 보기
              </button>
              {ev.filing_url && (
                <a className="small" href={ev.filing_url} target="_blank" rel="noreferrer">
                  EDGAR ↗
                </a>
              )}
            </div>
            <SpanText text={ev.text} lead={ev.lead_text} highlights={ev.highlights} />
          </article>
        ))}
      </section>
      <p className="small muted">
        판정: {d.model_id} · 질문 {d.question_version}. 관계는 근거 문장마다 판정한 뒤, 하나라도 채택된 문장이 있으면
        채택합니다.
      </p>
      {contextSpan && <ContextModal spanId={contextSpan} onClose={() => setContextSpan(null)} />}
    </div>
  );
}

/** 관계 점수·회사 식별 점수의 채택 기준. graph.db에 기록된 값이 없으면 기본값 0.8. */
function accept(d: EdgeDetail, question: string): number {
  return d.thresholds?.[question]?.[0] ?? 0.8;
}

/** 관계 점수는 채택 기준을 넘었는데 회사 식별이 불확실한 근거 (없으면 undefined). */
function entityDoubt(d: EdgeDetail): EdgeDetail["evidence"][number] | undefined {
  return d.evidence.find(
    (ev) => (ev.score ?? 0) >= accept(d, d.relation) && ev.s_is_entity != null && ev.s_is_entity < accept(d, "is_entity"),
  );
}

function uncertainReason(d: EdgeDetail, company: string): string {
  if (entityDoubt(d))
    return `관계 점수는 채택 기준(${accept(d, d.relation)})을 넘지만, 문장 속 이름이 정말 ${company}인지(회사 식별 점수)가 불확실해 검수 대기로 남았습니다.`;
  return `가장 높은 관계 점수(${d.score?.toFixed(2) ?? "–"})가 채택 기준(${accept(d, d.relation)})에 못 미쳐 검수 대기로 남았습니다.`;
}

function ReviewButtons({
  edge,
  company,
  entityDoubt,
  onSaved,
  onStale,
}: {
  edge: EdgeDetail;
  company: string;
  entityDoubt: boolean;
  onSaved: (verdict: "accept" | "reject") => void;
  onStale: () => void;
}) {
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (verdict: "accept" | "reject") => {
    setSaving(true);
    setError(null);
    try {
      await reviewApi.saveEdge({
        edge_id: edge.edge_id,
        verdict,
        note: note.trim() || null,
        evidence_hash: edge.evidence_hash,
      });
      onSaved(verdict);
    } catch (e) {
      setError((e as Error).message);
      if (e instanceof ApiError && e.status === 409) setTimeout(onStale, 1500); // 새 근거를 다시 불러온다
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="edge-review">
      <p className="small">
        {entityDoubt ? (
          <>
            문장 속 이름이 <b>{company}</b>가 맞고, 두 회사의 <b>{RELATION[edge.relation]}</b> 관계를 말하나요?
          </>
        ) : (
          <>
            근거 문장이 두 회사의 <b>{RELATION[edge.relation]}</b> 관계를 말하나요?
          </>
        )}
      </p>
      <div className="review-actions">
        <button className="btn primary" disabled={saving} onClick={() => save("accept")} data-verdict="accept">
          맞음 <kbd>A</kbd>
        </button>
        <button className="btn" disabled={saving} onClick={() => save("reject")} data-verdict="reject">
          아님 <kbd>R</kbd>
        </button>
        <input
          className="note-input"
          placeholder="메모 (선택)"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

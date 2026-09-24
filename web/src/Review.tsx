// 표본 가림 검수 (계획서 7.6). 모델 판정·유사도·단서 단어는 보여주지 않는다.
// 검수자는 근거 문장 하나마다 "이 문장이 X와 Y의 어떤 관계를 말하나"만 고른다.
// X = 문장이 나온 10-K를 낸 회사, Y = 문장이 언급한 회사.

import { useCallback, useEffect, useState } from "react";
import { reviewApi, type Relation, type ReviewNode, type SampleDetail } from "./api";
import { ErrorBox } from "./Common";
import { ContextModal } from "./ContextModal";
import { Highlighted } from "./Highlighted";
import { reviewHref, useFetch } from "./hooks";
import { SECTION } from "./labels";

const PURPOSE: Record<string, string> = {
  dev: "개발 표본 (질문·기준 조정용)",
  confirm: "확인 표본 (합격 판단용)",
};

export function ReviewPage({ sample, ord }: { sample: string | null; ord: number | null }) {
  if (!sample) return <SampleList />;
  return <SampleReview sampleId={sample} ord={ord} />;
}

function SampleList() {
  const list = useFetch("samples", reviewApi.samples);
  return (
    <main className="page">
      <section className="intro">
        <h1>표본 검수</h1>
        <p>
          표본 회사의 근거 문장마다 두 회사의 관계를 고릅니다. 모델 판정은 가려져 있고, 여기서 고른 결과가
          판정 모델을 채점하는 정답이 됩니다. 표본은 터미널에서 만듭니다:
        </p>
        <pre className="cmd">uv run tenksim sample -c configs/sp500_2024.yaml --name dev1 --purpose dev</pre>
      </section>
      {list.error && <ErrorBox message={list.error} />}
      {list.data?.length === 0 && <p className="muted">아직 표본이 없습니다.</p>}
      <div className="card-grid">
        {list.data?.map((s) => (
          <a key={s.sample_id} className="company-card" href={reviewHref(s.sample_id)}>
            <b>{s.sample_id}</b>
            <span className="small muted">{PURPOSE[s.purpose] ?? s.purpose}</span>
            <Progress done={s.n_labeled} total={s.n_units} />
            <span className="small muted">
              회사 {s.n_companies}곳 · 검수 {s.n_labeled}/{s.n_units}
              {s.n_skipped ? ` · 보류 ${s.n_skipped}` : ""}
            </span>
          </a>
        ))}
      </div>
    </main>
  );
}

function Progress({ done, total }: { done: number; total: number }) {
  const pct = total ? (100 * done) / total : 0;
  return (
    <div className="progress" role="progressbar" aria-valuenow={done} aria-valuemax={total}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

function SampleReview({ sampleId, ord }: { sampleId: string; ord: number | null }) {
  const [version, setVersion] = useState(0); // 저장할 때마다 진행 상황을 다시 불러온다
  const detail = useFetch(`sample:${sampleId}:${version}`, () => reviewApi.sample(sampleId));

  // 순서를 안 정했으면 처음 검수 안 한 단위로
  useEffect(() => {
    if (ord == null && detail.data) {
      const next = detail.data.units.find((u) => !u.labeled) ?? detail.data.units[0];
      if (next) window.location.replace(reviewHref(sampleId, next.ord));
    }
  }, [ord, detail.data, sampleId]);

  if (detail.error) return <main className="page"><ErrorBox message={detail.error} /></main>;
  if (!detail.data) return <main className="page"><p className="muted">불러오는 중…</p></main>;
  const d = detail.data;
  const done = d.units.filter((u) => u.labeled).length;
  return (
    <main className="review">
      <aside className="review-side">
        <a href={reviewHref()} className="small">
          ← 표본 목록
        </a>
        <h2>{d.sample.sample_id}</h2>
        <p className="small muted">{PURPOSE[d.sample.purpose] ?? d.sample.purpose}</p>
        <Progress done={done} total={d.units.length} />
        <p className="small muted">
          검수 {done} / {d.units.length}
        </p>
        <UnitGrid detail={d} current={ord} />
      </aside>
      {ord != null && (
        <UnitReview
          key={`${sampleId}:${ord}`}
          sampleId={sampleId}
          ord={ord}
          nextUnlabeled={d.units.find((u) => !u.labeled && u.ord > ord)?.ord ?? null}
          onSaved={() => setVersion((v) => v + 1)}
        />
      )}
    </main>
  );
}

function UnitGrid({ detail, current }: { detail: SampleDetail; current: number | null }) {
  // 회사별로 끊어서 보여준다 (표본은 회사 → 쌍 → 문서 순서로 정렬돼 있음)
  let lastPair = "";
  return (
    <div className="unit-grid">
      {detail.units.map((u) => {
        const newPair = u.pair_key !== lastPair;
        lastPair = u.pair_key;
        const cls = [
          "unit",
          u.labeled ? (u.skipped ? "skipped" : "done") : "",
          u.ord === current ? "current" : "",
          newPair ? "new-pair" : "",
        ].join(" ");
        const doc = detail.nodes[u.doc_node]?.name ?? u.doc_node;
        const target = detail.nodes[u.target_node]?.name ?? u.target_node;
        return (
          <a key={u.ord} className={cls} href={reviewHref(detail.sample.sample_id, u.ord)} title={`${doc} → ${target}`}>
            {u.ord + 1}
          </a>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- 단위 하나

type Entity = "yes" | "no" | "unsure";

interface Draft {
  isEntity: Entity | null;
  relations: Relation[] | null; // null = 아직 안 고름, [] = 관계를 말하지 않음
  note: string;
}

// 방향과 세부 유형은 묻지 않는다 (계획서 5장, 2026-09-25 결정)
function relationOptions(x: ReviewNode, y: ReviewNode): [Relation, string, string][] {
  const X = x.ticker ?? x.name;
  const Y = y.ticker ?? y.name;
  return [
    ["competitor", "경쟁", `${X}와 ${Y}가 경쟁한다 (경쟁사 목록 포함)`],
    [
      "business",
      "공급·협력",
      "한쪽이 다른 쪽에 제품·서비스를 팔거나 제공한다(고객·공급사·임대·금융 포함), 또는 제휴·합작·라이선스·유통. 방향은 따지지 않음",
    ],
    ["equity", "지분", "한쪽이 다른 쪽의 주식·지분을 가지고 있거나 가졌다(모회사·분사 포함). 방향은 따지지 않음"],
  ];
}

// v1 라벨(dev1)의 방향 있는 코드를 지금 선택지로 바꿔 보여준다
const LEGACY: Record<string, Relation> = {
  competitor: "competitor",
  doc_supplies_target: "business",
  target_supplies_doc: "business",
  partner: "business",
  doc_owns_target: "equity",
  target_owns_doc: "equity",
};

function currentRelations(codes: string[]): Relation[] {
  return [...new Set(codes.map((c) => LEGACY[c] ?? (c as Relation)))];
}

function UnitReview({
  sampleId,
  ord,
  nextUnlabeled,
  onSaved,
}: {
  sampleId: string;
  ord: number;
  nextUnlabeled: number | null;
  onSaved: () => void;
}) {
  const unit = useFetch(`unit:${sampleId}:${ord}`, () => reviewApi.unit(sampleId, ord));
  const [draft, setDraft] = useState<Draft>({ isEntity: null, relations: null, note: "" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showContext, setShowContext] = useState(false);

  // 이미 검수한 단위면 그 결과를 채워 둔다
  useEffect(() => {
    const u = unit.data;
    if (!u) return;
    const l = u.label;
    const anonymous = u.target.kind === "anonymous";
    setDraft(
      l && !l.skipped
        ? { isEntity: l.is_entity, relations: currentRelations(l.relations), note: l.note ?? "" }
        : { isEntity: anonymous ? "yes" : null, relations: null, note: l?.note ?? "" },
    );
  }, [unit.data]);

  const go = useCallback(
    (to: number) => {
      if (unit.data && to >= 0 && to < unit.data.n_units) window.location.hash = reviewHref(sampleId, to);
    },
    [sampleId, unit.data],
  );

  const save = useCallback(
    async (skipped: boolean) => {
      const u = unit.data;
      if (!u || saving) return;
      if (!skipped) {
        if (!draft.isEntity) return setError("먼저 이 이름이 그 회사가 맞는지 골라 주세요.");
        if (draft.isEntity === "yes" && draft.relations === null)
          return setError("관계를 고르거나, '관계를 말하지 않음'을 골라 주세요.");
      }
      setSaving(true);
      setError(null);
      try {
        const rels = draft.isEntity === "yes" ? (draft.relations ?? []) : [];
        await reviewApi.save({
          unit_id: u.unit_id,
          sample_id: sampleId,
          is_entity: skipped ? (draft.isEntity ?? "unsure") : draft.isEntity!,
          relations: skipped ? [] : rels,
          skipped,
          note: draft.note.trim() || null,
        });
        onSaved();
        const next = nextUnlabeled ?? ord + 1;
        if (next < u.n_units) go(next);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setSaving(false);
      }
    },
    [unit.data, saving, draft, sampleId, onSaved, nextUnlabeled, ord, go],
  );

  const toggle = useCallback((r: Relation) => {
    setDraft((d) => {
      const cur = d.relations ?? [];
      const relations = cur.includes(r) ? cur.filter((x) => x !== r) : [...cur, r];
      return { ...d, isEntity: d.isEntity ?? "yes", relations };
    });
  }, []);

  // 키보드: Y/N/U 회사 확인, 1~3 관계, 0 관계 없음, Enter 저장, S 보류, ←/→ 이동
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.tagName === "TEXTAREA" || el.tagName === "INPUT" || showContext) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const u = unit.data;
      if (!u) return;
      const k = e.key.toLowerCase();
      const opts = relationOptions(u.doc, u.target);
      if (k === "y") setDraft((d) => ({ ...d, isEntity: "yes" }));
      else if (k === "n") setDraft((d) => ({ ...d, isEntity: "no", relations: [] }));
      else if (k === "u") setDraft((d) => ({ ...d, isEntity: "unsure", relations: [] }));
      else if (k >= "1" && k <= String(opts.length)) toggle(opts[Number(k) - 1][0]);
      else if (k === "0") setDraft((d) => ({ ...d, isEntity: d.isEntity ?? "yes", relations: [] }));
      else if (k === "enter") save(false);
      else if (k === "s") save(true);
      else if (k === "arrowleft") go(ord - 1);
      else if (k === "arrowright") go(ord + 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [unit.data, toggle, save, go, ord, showContext]);

  if (unit.error) return <section className="review-main"><ErrorBox message={unit.error} /></section>;
  if (!unit.data) return <section className="review-main"><p className="muted">불러오는 중…</p></section>;
  const u = unit.data;
  const X = u.doc;
  const Y = u.target;
  const anonymous = Y.kind === "anonymous";
  const opts = relationOptions(X, Y);
  const entityOk = draft.isEntity === "yes";
  const matched = u.highlights.map((h) => u.text.slice(h.start, h.end));

  return (
    <section className="review-main">
      <div className="review-head">
        <span className="muted">
          {ord + 1} / {u.n_units}
        </span>
        {u.label && (
          <span className={u.label.skipped ? "tag warn" : "tag ok"}>
            {u.label.skipped ? "보류함" : "검수함"} · {u.label.labeled_at.replace("T", " ")}
          </span>
        )}
        {u.changed && (
          <span className="tag warn" title="표본을 뽑은 뒤 graph.db를 다시 만들어 이 문장이 바뀌었습니다">
            원문이 바뀜
          </span>
        )}
        <span className="spacer" />
        <button className="btn" onClick={() => go(ord - 1)} disabled={ord === 0}>
          ← 이전
        </button>
        <button className="btn" onClick={() => go(ord + 1)} disabled={ord + 1 >= u.n_units}>
          다음 →
        </button>
      </div>

      <article className="review-card">
        <div className="span-meta">
          <b className="doc-name">{X.name}</b>
          <span>의 10-K</span>
          <span>· {SECTION[u.section] ?? u.section}</span>
          {u.filing_date && <span>· 제출 {u.filing_date}</span>}
          <button className="link" onClick={() => setShowContext(true)}>
            본문에서 보기
          </button>
        </div>
        {u.lead_text && <p className="lead">{u.lead_text} …</p>}
        <p className="review-text">
          <Highlighted text={u.text} ranges={u.highlights.map((h) => ({ ...h, className: "name" }))} />
        </p>
      </article>

      {!anonymous && (
        <fieldset className="q">
          <legend>
            1. 강조한 이름 {matched.length ? `“${[...new Set(matched)].join("”, “")}”` : ""}이{" "}
            <b>
              {Y.name}
              {Y.ticker ? ` (${Y.ticker})` : ""}
            </b>
            를 가리키나요?
          </legend>
          <div className="opts">
            <Choice on={draft.isEntity === "yes"} k="Y" onClick={() => setDraft((d) => ({ ...d, isEntity: "yes" }))}>
              맞음
            </Choice>
            <Choice
              on={draft.isEntity === "no"}
              k="N"
              onClick={() => setDraft((d) => ({ ...d, isEntity: "no", relations: [] }))}
            >
              아님 (다른 회사, 지명, 제품명 등)
            </Choice>
            <Choice
              on={draft.isEntity === "unsure"}
              k="U"
              onClick={() => setDraft((d) => ({ ...d, isEntity: "unsure", relations: [] }))}
            >
              모르겠음
            </Choice>
          </div>
        </fieldset>
      )}

      <fieldset className="q" disabled={!entityOk}>
        <legend>
          {anonymous ? "1" : "2"}. 이 문장이 <b>{X.name}</b>와 <b>{Y.name}</b>의 어떤 관계를 말하나요?{" "}
          <span className="muted small">여러 개 고를 수 있음</span>
        </legend>
        <div className="opts col">
          {opts.map(([code, label, hint], i) => (
            <Choice key={code} on={!!draft.relations?.includes(code)} k={String(i + 1)} onClick={() => toggle(code)} hint={hint}>
              {label}
            </Choice>
          ))}
          <Choice
            on={draft.relations !== null && draft.relations.length === 0}
            k="0"
            onClick={() => setDraft((d) => ({ ...d, relations: [] }))}
            hint="이름만 나열, 임원 경력, 업계 일반 설명 등. 이미 끝난 관계도 관계로 고릅니다"
          >
            관계를 말하지 않음
          </Choice>
        </div>
      </fieldset>

      <label className="note">
        <span className="small muted">메모 (선택)</span>
        <textarea
          rows={2}
          value={draft.note}
          onChange={(e) => setDraft((d) => ({ ...d, note: e.target.value }))}
          placeholder="판단이 애매했던 이유 등"
        />
      </label>

      {error && <p className="error">{error}</p>}
      <div className="review-actions">
        <button className="btn primary" onClick={() => save(false)} disabled={saving}>
          저장하고 다음 <kbd>Enter</kbd>
        </button>
        <button className="btn" onClick={() => save(true)} disabled={saving}>
          보류 <kbd>S</kbd>
        </button>
        <span className="small muted">←/→ 이동 · 1~3 관계 · 0 관계 없음</span>
      </div>
      {showContext && <ContextModal spanId={u.span_id} onClose={() => setShowContext(false)} />}
    </section>
  );
}

function Choice({
  on,
  k,
  hint,
  onClick,
  children,
}: {
  on: boolean;
  k?: string;
  hint?: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button type="button" className={on ? "choice on" : "choice"} onClick={onClick} aria-pressed={on}>
      {k && <kbd>{k}</kbd>}
      <span>
        {children}
        {hint && <span className="hint">{hint}</span>}
      </span>
    </button>
  );
}

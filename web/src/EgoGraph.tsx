// 한 회사를 가운데 둔 관계도 (계획서 8.2). Cytoscape.js의 좌표 지정(preset) 배치.
// - 관계에 방향이 없으므로: 경쟁사는 위, 공급·협력은 아래, 지분만 있는 회사는 오른쪽 열.
// - 위·아래 호는 같은 궤도(점선 원) 위에 두고 양옆을 비워, 두 구역이 갈라져 보이게 한다.
// - 여러 관계인 회사는 점 하나로 그리고(선은 관계마다 하나), 위치는 공급·협력 > 경쟁 > 지분 순으로 정한다.
// - 선 색 = 관계 유형(가운데 쪽으로 옅어짐), 점선 = 검수 대기, 흐린 선 = 과거 관계.
// - 점 색 = GICS 섹터, 회색 = 분석 대상 밖, 빈 원 = 익명 공시, 점 크기 = 근거 문장 수.
// - 점을 누르면 그 관계의 근거가 오른쪽에 나오고, 두 번 누르면 그 회사의 관계도로 간다.

import cytoscape, { type Core, type ElementDefinition, type NodeSingular } from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";
import type { NodeDetail, NodeKind, NodeRelation, RelationType } from "./api";
import { useTheme } from "./hooks";
import { EXTERNAL_COLOR, RELATION, RELATION_COLOR, SECTOR_COLOR, SECTOR_KO, STATE, STATUS } from "./labels";

const PRIORITY: RelationType[] = ["business", "competitor", "equity"];
const GAP = 50; // 같은 줄에서 이웃한 두 점 사이 거리
const ROW = 74; // 두 줄로 놓을 때 줄 간격
const MAX_SPREAD = Math.PI * 0.8; // 한 호가 차지하는 최대 각도 (양옆을 36°씩 비운다)
const ROW_LIMITS = [16, 36]; // 한 호에 16곳이 넘으면 두 줄, 36곳이 넘으면 세 줄
const EQUITY_STEP = 46;
const ZONE = 48; // 구역 이름과 호 바깥 줄 사이
const FONT = '"Pretendard Variable", Pretendard, system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif';

interface Other {
  id: string;
  name: string;
  ticker: string | null;
  sector: string | null;
  kind: NodeKind;
  rels: NodeRelation[]; // PRIORITY 순. 첫 관계가 점의 위치와 누를 때 여는 관계를 정한다
  evidence: number;
}

/** 이름표를 놓는 쪽: 궤도 바깥쪽. 호의 양 끝(옆으로 누운 곳)은 좌우에 둔다. */
type Side = "top" | "bottom" | "left" | "right";

interface Spot {
  x: number;
  y: number;
  side: Side;
}

interface Layout {
  others: (Other & Spot)[];
  rings: { w: number; h: number }[];
  zones: { relation: RelationType; x: number; y: number }[];
  /** 그림의 세로 길이 (모형 좌표) */
  height: number;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function colorOf(o: Pick<Other, "kind" | "sector">): string {
  return o.kind === "company" ? (SECTOR_COLOR[o.sector ?? ""] ?? EXTERNAL_COLOR) : EXTERNAL_COLOR;
}

function shortName(name: string): string {
  return name.length > 16 ? `${name.slice(0, 15)}…` : name;
}

/** 상대 회사별로 관계를 묶고, 첫 관계 유형으로 구역을 나눈다. 구역 안에서는 같은 섹터끼리 모이게 정렬. */
function group(relations: NodeRelation[]): Record<RelationType, Other[]> {
  const byOther = new Map<string, NodeRelation[]>();
  for (const r of relations) byOther.set(r.other_id, [...(byOther.get(r.other_id) ?? []), r]);
  const groups: Record<RelationType, Other[]> = { business: [], competitor: [], equity: [] };
  for (const rs of byOther.values()) {
    rs.sort((a, b) => PRIORITY.indexOf(a.relation) - PRIORITY.indexOf(b.relation));
    const r = rs[0];
    groups[r.relation].push({
      id: r.other_id,
      name: r.other_name,
      ticker: r.other_ticker,
      sector: r.other_sector,
      kind: r.other_kind,
      rels: rs,
      evidence: rs.reduce((s, x) => s + x.n_evidence, 0),
    });
  }
  for (const g of Object.values(groups))
    g.sort((a, b) => (a.sector ?? "~").localeCompare(b.sector ?? "~") || a.name.localeCompare(b.name));
  return groups;
}

const rowsFor = (n: number) => 1 + ROW_LIMITS.filter((k) => n > k).length;

/** 위·아래 호가 모두 MAX_SPREAD 안에 들어가는 궤도 반지름(세로). */
function orbit(...ns: number[]): number {
  return Math.max(150, ...ns.map((n) => (GAP * (n - 1)) / (rowsFor(n) * MAX_SPREAD)));
}

/** 궤도를 옆으로 늘리는 비율. 그림 칸이 가로로 길고 이름도 가로로 길어서 늘리되, 큰 그림은 칸이 네모에 가까워 덜 늘린다. */
function wideFor(R: number): number {
  return Math.min(1.35, Math.max(1.05, 1.35 - (R - 200) / 400));
}

/** 궤도(가로로 늘린 타원) 위 호에 n개를 고르게 놓는다. 여러 줄이면 바깥 줄(R + ROW, R + 2·ROW)과 번갈아 놓는다. */
function arc(n: number, center: number, R: number, wide: number, side: "top" | "bottom"): Spot[] {
  const rows = rowsFor(n);
  const spread = Math.min(MAX_SPREAD, (GAP * (n - 1)) / (rows * R));
  return Array.from({ length: n }, (_, i) => {
    const t = n === 1 ? center : center - spread / 2 + (spread * i) / (n - 1);
    const r = R + (i % rows) * ROW;
    const c = Math.cos(t);
    return { x: wide * r * c, y: r * Math.sin(t), side: Math.abs(c) > 0.72 ? (c > 0 ? "right" : "left") : side };
  });
}

function layout(groups: Record<RelationType, Other[]>): Layout {
  const nc = groups.competitor.length;
  const nb = groups.business.length;
  const ne = groups.equity.length;
  const R = orbit(nc, nb);
  const wide = wideFor(R);
  const outerC = R + (rowsFor(nc) - 1) * ROW;
  const outerB = R + (rowsFor(nb) - 1) * ROW;
  const column = (nc || nb ? wide * Math.max(outerC, outerB) : 60) + 120;
  const spots: Record<RelationType, Spot[]> = {
    competitor: arc(nc, -Math.PI / 2, R, wide, "top"),
    business: arc(nb, Math.PI / 2, R, wide, "bottom"),
    equity: groups.equity.map((_, i) => ({ x: column, y: (i - (ne - 1) / 2) * EQUITY_STEP, side: "right" })),
  };
  const zones: Layout["zones"] = [];
  if (nc) zones.push({ relation: "competitor", x: 0, y: -(outerC + ZONE) });
  if (nb) zones.push({ relation: "business", x: 0, y: outerB + ZONE });
  if (ne) zones.push({ relation: "equity", x: column + 14, y: -((ne - 1) / 2) * EQUITY_STEP - 40 });
  const ring = (r: number) => ({ w: 2 * wide * r, h: 2 * r });
  const nRings = nc || nb ? Math.max(rowsFor(nc), rowsFor(nb)) : 0;
  const rings = Array.from({ length: nRings }, (_, i) => ring(R + i * ROW));
  const top = nc ? outerC + ZONE : 80;
  const bottom = nb ? outerB + ZONE : 80;
  const equity = ne ? ((ne - 1) / 2) * EQUITY_STEP + 40 : 0;
  return {
    others: PRIORITY.flatMap((p) => groups[p].map((o, i) => ({ ...o, ...spots[p][i] }))),
    rings,
    zones,
    height: Math.max(top, equity) + Math.max(bottom, equity) + 30,
  };
}

/** 근거 문장 수에 따른 점 크기: 1개 20px, 2개 25, 4개 30, 8개 이상 35. */
function sizeFor(evidence: number): number {
  return 20 + 5 * Math.min(3, Math.log2(Math.max(1, evidence)));
}

interface Tip {
  id: string;
  x: number;
  y: number;
  below: boolean;
}

export function EgoGraph({
  center,
  relations,
  selectedEdge,
  onSelectEdge,
  onOpenNode,
  compact = false,
}: {
  center: NodeDetail;
  relations: NodeRelation[];
  selectedEdge: string | null;
  onSelectEdge: (edgeId: string) => void;
  onOpenNode: (nodeId: string) => void;
  /** 첫 화면 미리보기: 도구·범례 없이 */
  compact?: boolean;
}) {
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const handlers = useRef({ onSelectEdge, onOpenNode });
  handlers.current = { onSelectEdge, onOpenNode };
  const theme = useTheme();
  const [tip, setTip] = useState<Tip | null>(null);
  const [full, setFull] = useState(false);

  const groups = useMemo(() => group(relations), [relations]);
  const placed = useMemo(() => layout(groups), [groups]);
  const byId = useMemo(() => new Map(placed.others.map((o) => [o.id, o])), [placed]);

  useEffect(() => {
    if (!box.current) return;
    const text = cssVar("--text", "#1c1f24");
    const muted = cssVar("--muted", "#5f6878");
    const panel = cssVar("--graph-bg", "#ffffff");
    const line = cssVar("--graph-ring", "#d9dde5");
    const centerColor = center.kind === "company" ? (SECTOR_COLOR[center.gics_sector ?? ""] ?? EXTERNAL_COLOR) : EXTERNAL_COLOR;
    const centerLabel = center.ticker ?? center.name;

    const elements: ElementDefinition[] = [
      ...placed.rings.map((r, i) => ({
        data: { id: `ring:${i}`, w: r.w, h: r.h },
        position: { x: 0, y: 0 },
        classes: "deco ring",
        grabbable: false,
        selectable: false,
      })),
      ...placed.zones.map((z) => ({
        data: { id: `zone:${z.relation}`, label: RELATION[z.relation], color: RELATION_COLOR[z.relation] },
        position: { x: z.x, y: z.y },
        classes: "deco zone",
        grabbable: false,
        selectable: false,
      })),
      {
        data: { id: center.node_id, label: centerLabel, color: centerColor },
        position: { x: 0, y: 0 },
        classes: centerLabel.length > 6 ? "center long" : "center",
        grabbable: false,
      },
      ...placed.others.map((o) => ({
        data: {
          id: o.id,
          label: o.ticker ?? shortName(o.name),
          color: colorOf(o),
          size: sizeFor(o.evidence),
        },
        position: { x: o.x, y: o.y },
        classes: ["other", `side-${o.side}`, o.kind === "anonymous" ? "hollow" : ""].join(" "),
      })),
      ...relations.map((r) => {
        const color = RELATION_COLOR[r.relation];
        return {
          data: {
            id: r.edge_id,
            source: center.node_id,
            target: r.other_id,
            color,
            grad: [alpha(color, 0.12), color],
          },
          classes: [r.state === "uncertain" ? "uncertain" : "", r.status === "historical" ? "past" : ""].join(" "),
        };
      }),
    ];

    const instance = cytoscape({
      container: box.current,
      elements,
      layout: { name: "preset", fit: true, padding: compact ? 16 : 28 },
      minZoom: 0.25,
      maxZoom: 3,
      wheelSensitivity: 0.25,
      autounselectify: true,
      boxSelectionEnabled: false,
      userZoomingEnabled: !compact,
      userPanningEnabled: !compact,
      style: [
        { selector: "node, edge", style: { "z-index-compare": "manual", "overlay-opacity": 0 } },
        {
          selector: "node",
          style: {
            "font-family": FONT,
            "z-index": 3,
            "transition-property": "opacity",
            "transition-duration": 180,
          },
        },
        {
          selector: "node.other",
          style: {
            width: "data(size)",
            height: "data(size)",
            "background-color": "data(color)",
            "border-width": 2.5,
            "border-color": panel,
            label: "data(label)",
            color: text,
            "font-size": 11,
            "font-weight": 600,
            "text-background-color": panel,
            "text-background-opacity": 0.82,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "2px",
            "min-zoomed-font-size": 6,
          },
        },
        { selector: "node.side-top", style: { "text-valign": "top", "text-margin-y": -4 } },
        { selector: "node.side-bottom", style: { "text-valign": "bottom", "text-margin-y": 4 } },
        { selector: "node.side-right", style: { "text-halign": "right", "text-valign": "center", "text-margin-x": 6 } },
        { selector: "node.side-left", style: { "text-halign": "left", "text-valign": "center", "text-margin-x": -6 } },
        {
          selector: "node.hollow",
          style: { "background-color": panel, "border-color": muted, "border-style": "dashed", "border-width": 2 },
        },
        {
          selector: "node.center",
          style: {
            width: 68,
            height: 68,
            "background-color": "data(color)",
            label: "data(label)",
            color: "#ffffff",
            "font-size": 15,
            "font-weight": 800,
            "text-valign": "center",
            "text-halign": "center",
            "border-width": 4,
            "border-color": panel,
            "underlay-color": "data(color)",
            "underlay-padding": 14,
            "underlay-opacity": 0.16,
            "underlay-shape": "ellipse",
            "z-index": 4,
          },
        },
        {
          selector: "node.center.long",
          style: {
            color: text,
            "font-size": 13,
            "text-valign": "bottom",
            "text-margin-y": 10,
            "text-background-color": panel,
            "text-background-opacity": 0.85,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "3px",
          },
        },
        {
          selector: "node.ring",
          style: {
            width: "data(w)",
            height: "data(h)",
            "background-opacity": 0,
            "border-width": 1.2,
            "border-style": "dashed",
            "border-color": line,
            events: "no",
            "z-index": 0,
          },
        },
        {
          selector: "node.zone",
          style: {
            width: 1,
            height: 1,
            "background-opacity": 0,
            "border-width": 0,
            label: "data(label)",
            color: "data(color)",
            "font-size": 14,
            "font-weight": 700,
            "text-valign": "center",
            "text-halign": "center",
            "text-background-color": "data(color)",
            "text-background-opacity": 0.14,
            "text-background-shape": "roundrectangle",
            "text-background-padding": "6px",
            events: "no",
            "z-index": 0,
          },
        },
        {
          selector: "edge",
          style: {
            width: 2.2,
            "curve-style": "bezier",
            "control-point-step-size": 26,
            "line-color": "data(color)",
            "line-fill": "linear-gradient",
            "line-gradient-stop-colors": (e: cytoscape.EdgeSingular) => e.data("grad"),
            "line-gradient-stop-positions": [8, 72],
            opacity: 0.9,
            "z-index": 1,
            "transition-property": "opacity, width",
            "transition-duration": 180,
          },
        },
        { selector: "edge.uncertain", style: { "line-style": "dashed", "line-dash-pattern": [6, 4] } },
        { selector: "edge.past", style: { opacity: 0.35 } },
        { selector: "edge.hover", style: { width: 3.6, opacity: 1 } },
        {
          selector: "edge.selected",
          style: {
            width: 5,
            opacity: 1,
            "underlay-color": "data(color)",
            "underlay-padding": 5,
            "underlay-opacity": 0.18,
            "z-index": 2,
          },
        },
        {
          selector: "node.hover, node.picked",
          style: {
            "underlay-color": "data(color)",
            "underlay-padding": 7,
            "underlay-opacity": 0.28,
            "underlay-shape": "ellipse",
            "font-weight": 800,
          },
        },
        { selector: "node.dim", style: { opacity: 0.18 } },
        { selector: "edge.dim", style: { opacity: 0.07 } },
      ],
    });

    const hover = (n: NodeSingular | null) => {
      instance.batch(() => {
        instance.elements().removeClass("dim hover");
        if (!n) return;
        const keep = n.union(n.connectedEdges()).union(instance.getElementById(center.node_id));
        instance.elements().not(keep).not(".deco").addClass("dim");
        n.addClass("hover");
        n.connectedEdges().addClass("hover");
      });
      if (!n) return setTip(null);
      // 점 위에 띄우고, 위쪽 끝이면 아래에 띄운다. 좌우는 그림 밖으로 나가지 않게
      const p = n.renderedPosition();
      const half = n.renderedHeight() / 2;
      const below = p.y - half < 150;
      setTip({
        id: n.id(),
        x: Math.min(Math.max(p.x, 150), instance.width() - 150),
        y: below ? p.y + half : p.y - half,
        below,
      });
    };
    instance.on("mouseover", "node.other", (e) => hover(e.target));
    instance.on("mouseout", "node.other", () => hover(null));
    instance.on("mouseover", "edge", (e) => e.target.addClass("hover"));
    instance.on("mouseout", "edge", (e) => e.target.removeClass("hover"));
    instance.on("viewport drag", () => setTip(null));
    instance.on("tap", "edge", (e) => handlers.current.onSelectEdge(e.target.id()));
    instance.on("tap", "node.other", (e) => {
      const o = byId.get(e.target.id());
      if (o) handlers.current.onSelectEdge(o.rels[0].edge_id);
    });
    instance.on("dbltap", "node.other", (e) => handlers.current.onOpenNode(e.target.id()));
    instance.on("mouseover", "node.other, edge", () => (box.current!.style.cursor = "pointer"));
    instance.on("mouseout", "node.other, edge", () => (box.current!.style.cursor = ""));

    // 가운데에서 퍼져 나가는 첫 등장 (움직임 줄이기 설정이면 생략)
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      instance.nodes(".other").forEach((n, i) => {
        const to = { ...n.position() };
        n.position({ x: 0, y: 0 });
        n.style("opacity", 0);
        n.delay(Math.min(i * 12, 360)).animate(
          { position: to, style: { opacity: 1 } },
          { duration: 520, easing: "ease-out-cubic", complete: () => void n.removeStyle("opacity") },
        );
      });
      const deco = instance.nodes(".deco");
      deco.style("opacity", 0);
      deco.animate({ style: { opacity: 1 } }, { duration: 700, complete: () => void deco.removeStyle("opacity") });
    }
    // 웹 글꼴이 늦게 오면 글자 폭을 다시 잰다
    let alive = true;
    document.fonts?.ready.then(() => alive && instance.style().update());

    cy.current = instance;
    return () => {
      alive = false;
      instance.destroy();
      cy.current = null;
      setTip(null);
    };
  }, [center, relations, placed, byId, theme, compact]);

  useEffect(() => {
    const c = cy.current;
    if (!c) return;
    c.batch(() => {
      c.elements().removeClass("selected picked");
      if (!selectedEdge) return;
      const e = c.getElementById(selectedEdge);
      e.addClass("selected");
      e.connectedNodes(".other").addClass("picked");
    });
  }, [selectedEdge, relations, theme]);

  // 크게 보기: 창 크기가 바뀐 뒤 다시 맞춘다 (처음 그릴 때는 건너뛴다: 첫 등장 움직임과 겹치지 않게). Esc로 닫는다
  const lastFull = useRef(full);
  useEffect(() => {
    const c = cy.current;
    if (!c || lastFull.current === full) return;
    lastFull.current = full;
    const raf = requestAnimationFrame(() => {
      c.resize();
      c.animate({ fit: { eles: c.elements(), padding: 36 } }, { duration: 250 });
    });
    return () => cancelAnimationFrame(raf);
  }, [full]);
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setFull(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);

  const zoomBy = (k: number) => {
    const c = cy.current;
    if (!c) return;
    c.animate(
      { zoom: { level: c.zoom() * k, renderedPosition: { x: c.width() / 2, y: c.height() / 2 } } },
      { duration: 160 },
    );
  };
  const fit = () => cy.current?.animate({ fit: { eles: cy.current.elements(), padding: 36 } }, { duration: 250 });
  const savePng = () => {
    const c = cy.current;
    if (!c) return;
    const blob = c.png({ output: "blob", bg: cssVar("--graph-bg", "#ffffff"), full: true, scale: 2, maxWidth: 4000 });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${center.ticker ?? center.name}-관계도.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  // 배율이 1 안팎이 되게 그림 칸의 높이를 정한다 (너무 작게 줄어 이름을 못 읽지 않게)
  const height = Math.round(compact ? Math.min(520, Math.max(400, placed.height * 0.75)) : Math.min(800, Math.max(440, placed.height + 40)));
  const shown = tip ? byId.get(tip.id) : undefined;
  const sectors = useMemo(() => {
    const s = new Map<string, string>();
    for (const o of placed.others) {
      if (o.kind === "company" && o.sector && SECTOR_COLOR[o.sector]) s.set(o.sector, SECTOR_COLOR[o.sector]);
    }
    return [...s.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [placed]);
  const hasExternal = placed.others.some((o) => o.kind === "external" || (o.kind === "company" && !SECTOR_COLOR[o.sector ?? ""]));
  const hasAnonymous = placed.others.some((o) => o.kind === "anonymous");

  return (
    <>
      {full && <div className="ego-backdrop" onClick={() => setFull(false)} />}
      <div className={`ego-graph${full ? " full" : ""}${compact ? " compact" : ""}`}>
        <div className="ego-stage" style={full ? undefined : { height }}>
          <div ref={box} className="ego-canvas" role="img" aria-label={`${center.name}의 관계도`} />
          {!compact && (
            <div className="ego-tools" role="toolbar" aria-label="관계도 도구">
              <button onClick={() => zoomBy(1.3)} title="확대" aria-label="확대">
                <Icon d="M12 5v14M5 12h14" />
              </button>
              <button onClick={() => zoomBy(1 / 1.3)} title="축소" aria-label="축소">
                <Icon d="M5 12h14" />
              </button>
              <button onClick={fit} title="전체 보기" aria-label="전체 보기">
                <Icon d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
              </button>
              <span className="sep" />
              <button onClick={() => setFull((f) => !f)} title={full ? "작게 보기 (Esc)" : "크게 보기"} aria-label="크게 보기">
                <Icon d={full ? "M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7" : "M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"} />
              </button>
              <button onClick={savePng} title="그림 파일로 저장 (PNG)" aria-label="PNG로 저장">
                <Icon d="M12 4v11M7 10l5 5 5-5M5 20h14" />
              </button>
            </div>
          )}
          {shown && tip && (
            <div className={tip.below ? "ego-tip below" : "ego-tip"} style={{ left: tip.x, top: tip.y }}>
              <div className="ego-tip-head">
                <i className={shown.kind === "anonymous" ? "dot hollow" : "dot"} style={{ background: shown.kind === "anonymous" ? undefined : colorOf(shown) }} />
                <b>{shown.name}</b>
                {shown.ticker && <span className="ticker">{shown.ticker}</span>}
              </div>
              <div className="ego-tip-sub">
                {shown.kind === "company"
                  ? (SECTOR_KO[shown.sector ?? ""] ?? shown.sector ?? "섹터 모름")
                  : shown.kind === "external"
                    ? "분석 대상 밖 회사"
                    : "익명 공시 (이름 없이 적힌 고객 등)"}
              </div>
              <ul>
                {shown.rels.map((r) => (
                  <li key={r.edge_id}>
                    <i style={{ background: RELATION_COLOR[r.relation] }} />
                    {RELATION[r.relation]}
                    <span className="muted"> · 근거 {r.n_evidence}</span>
                    {r.state === "uncertain" && <span className="warn"> · {STATE[r.state]}</span>}
                    {r.status === "historical" && <span className="muted"> · {STATUS[r.status]}</span>}
                  </li>
                ))}
              </ul>
              <div className="ego-tip-foot">클릭: 근거 보기 · 더블클릭: 이 회사 관계도</div>
            </div>
          )}
        </div>
        {!compact && (
          <div className="ego-legend">
            <div className="row">
              {Object.entries(RELATION).map(([k, v]) => (
                <span key={k}>
                  <i className="line" style={{ background: RELATION_COLOR[k] }} /> {v}
                </span>
              ))}
              <span>
                <i className="line dash" /> 검수 대기
              </span>
              <span>
                <i className="line faint" /> 과거 관계
              </span>
              <span className="sizes" title="점 크기 = 근거 문장 수">
                <i style={{ width: 6, height: 6 }} />
                <i style={{ width: 9, height: 9 }} />
                <i style={{ width: 12, height: 12 }} /> 근거 문장 수
              </span>
            </div>
            <div className="row sectors">
              {sectors.map(([s, c]) => (
                <span key={s} title={s}>
                  <i className="dot" style={{ background: c }} /> {SECTOR_KO[s] ?? s}
                </span>
              ))}
              {hasExternal && (
                <span>
                  <i className="dot" style={{ background: EXTERNAL_COLOR }} /> 분석 대상 밖
                </span>
              )}
              {hasAnonymous && (
                <span>
                  <i className="dot hollow" /> 익명 공시
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

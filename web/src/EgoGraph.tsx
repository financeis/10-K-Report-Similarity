// 한 회사를 가운데 둔 관계도 (계획서 8.2). Cytoscape.js의 좌표 지정(preset) 배치.
// - 관계에 방향이 없으므로: 경쟁사는 위, 공급·협력은 아래, 지분만 있는 회사는 오른쪽.
// - 여러 관계인 회사는 점 하나로 그리고, 위치는 공급·협력 > 경쟁 > 지분 순으로 정한다.
// - 선 색 = 관계 유형, 점선 = 검수 대기, 흐린 선 = 과거 관계. 선 굵기는 일정하게 둔다.
// - 점 색 = GICS 섹터, 회색 = 분석 대상 밖, 빈 원 = 익명 공시.

import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { useEffect, useRef } from "react";
import type { NodeDetail, NodeRelation, RelationType } from "./api";
import { RELATION, RELATION_COLOR, SECTOR_COLOR } from "./labels";

const PRIORITY: RelationType[] = ["business", "competitor", "equity"];

interface Placed {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  hollow: boolean;
  roles: string;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** 역할별로 호(arc) 위에 고르게 놓는다. 많으면 두 줄(안쪽·바깥쪽)에 번갈아 놓아 반지름을 덜 키운다. */
function arc(n: number, center: number, spread: number, radius: number): { x: number; y: number }[] {
  if (n === 0) return [];
  if (n === 1) return [{ x: radius * Math.cos(center), y: radius * Math.sin(center) }];
  const twoRows = n > 14;
  return Array.from({ length: n }, (_, i) => {
    const t = center - spread / 2 + (spread * i) / (n - 1);
    const r = twoRows && i % 2 ? radius + 70 : radius;
    return { x: r * Math.cos(t), y: r * Math.sin(t) };
  });
}

/** 호의 가장 바깥 줄 반지름. */
function outer(n: number): number {
  return radiusFor(n) + (n > 14 ? 70 : 0);
}

/** 줄당 점 수에 맞춘 반지름. 두 줄이면 한 줄에 절반만 놓인다. */
function radiusFor(n: number): number {
  const perRow = n > 14 ? Math.ceil(n / 2) : n;
  return Math.max(170, 60 + perRow * 18);
}

function layout(relations: NodeRelation[]): Placed[] {
  const byOther = new Map<string, NodeRelation[]>();
  for (const r of relations) byOther.set(r.other_id, [...(byOther.get(r.other_id) ?? []), r]);
  const groups: Record<RelationType, NodeRelation[][]> = { business: [], competitor: [], equity: [] };
  for (const rs of byOther.values()) {
    const primary = PRIORITY.find((p) => rs.some((r) => r.relation === p))!;
    groups[primary].push(rs);
  }
  // 같은 섹터끼리 모이게 정렬
  for (const g of Object.values(groups))
    g.sort((a, b) => (a[0].other_sector ?? "~").localeCompare(b[0].other_sector ?? "~") || a[0].other_name.localeCompare(b[0].other_name));

  const spread = (n: number) => Math.min(Math.PI * 0.9, 0.35 * n);
  const spots: Record<RelationType, { x: number; y: number }[]> = {
    competitor: arc(groups.competitor.length, -Math.PI / 2, spread(groups.competitor.length), radiusFor(groups.competitor.length)),
    business: arc(groups.business.length, Math.PI / 2, spread(groups.business.length), radiusFor(groups.business.length)),
    // 지분 열은 두 호의 가장 바깥 줄(두 줄이면 +70)보다 오른쪽에 둔다
    equity: groups.equity.map((_, i) => ({
      x: Math.max(outer(groups.competitor.length), outer(groups.business.length)) + 60,
      y: (i - (groups.equity.length - 1) / 2) * 56,
    })),
  };
  const out: Placed[] = [];
  for (const p of PRIORITY) {
    groups[p].forEach((rs, i) => {
      const r = rs[0];
      const color =
        r.other_kind === "company" ? SECTOR_COLOR[r.other_sector ?? ""] ?? "#8a8f98" : r.other_kind === "external" ? "#9aa0a6" : "#ffffff";
      out.push({
        id: r.other_id,
        label: r.other_ticker ?? (r.other_name.length > 18 ? `${r.other_name.slice(0, 17)}…` : r.other_name),
        ...spots[p][i],
        color,
        hollow: r.other_kind === "anonymous",
        // '공급·협력' 안에 '·'가 있어 관계끼리는 ' + '로 잇는다
        roles: rs.length > 1 ? [...new Set(rs.map((x) => RELATION[x.relation]))].join(" + ") : "",
      });
    });
  }
  return out;
}

export function EgoGraph({
  center,
  relations,
  selectedEdge,
  onSelectEdge,
  onOpenNode,
}: {
  center: NodeDetail;
  relations: NodeRelation[];
  selectedEdge: string | null;
  onSelectEdge: (edgeId: string) => void;
  onOpenNode: (nodeId: string) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const handlers = useRef({ onSelectEdge, onOpenNode });
  handlers.current = { onSelectEdge, onOpenNode };

  useEffect(() => {
    if (!box.current) return;
    const text = cssVar("--text", "#1c1f24");
    const bg = cssVar("--panel", "#ffffff");
    const placed = layout(relations);
    const elements: ElementDefinition[] = [
      {
        data: { id: center.node_id, label: center.ticker ?? center.name, color: SECTOR_COLOR[center.gics_sector ?? ""] ?? "#8a8f98", roles: "" },
        position: { x: 0, y: 0 },
        classes: "center",
      },
      ...placed.map((p) => ({
        data: { id: p.id, label: p.roles ? `${p.label}\n${p.roles}` : p.label, color: p.color },
        position: { x: p.x, y: p.y },
        classes: p.hollow ? "hollow" : "",
      })),
      ...relations.map((r) => ({
        data: { id: r.edge_id, source: center.node_id, target: r.other_id, color: RELATION_COLOR[r.relation] },
        classes: [r.state === "uncertain" ? "uncertain" : "", r.status === "historical" ? "past" : ""].join(" "),
      })),
    ];
    const instance = cytoscape({
      container: box.current,
      elements,
      layout: { name: "preset", fit: true, padding: 36 },
      minZoom: 0.3,
      maxZoom: 2.5,
      wheelSensitivity: 0.2,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            color: text,
            "font-size": 11,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-wrap": "wrap",
            "text-outline-color": bg,
            "text-outline-width": 2,
            "min-zoomed-font-size": 6,
            width: 16,
            height: 16,
            "border-width": 1,
            "border-color": bg,
          },
        },
        { selector: "node.center", style: { width: 30, height: 30, "font-size": 14, "font-weight": "bold" } },
        { selector: "node.hollow", style: { "background-opacity": 0, "border-width": 2, "border-color": text } },
        {
          selector: "edge",
          style: { width: 2, "line-color": "data(color)", "curve-style": "bezier", opacity: 0.85 },
        },
        { selector: "edge.uncertain", style: { "line-style": "dashed" } },
        { selector: "edge.past", style: { opacity: 0.35 } },
        { selector: "edge.selected", style: { width: 5, opacity: 1 } },
        { selector: "node:active", style: { "overlay-opacity": 0 } },
      ],
    });
    instance.on("tap", "edge", (e) => handlers.current.onSelectEdge(e.target.id()));
    instance.on("tap", "node", (e) => {
      if (e.target.id() !== center.node_id) handlers.current.onOpenNode(e.target.id());
    });
    instance.on("mouseover", "node, edge", () => (box.current!.style.cursor = "pointer"));
    instance.on("mouseout", "node, edge", () => (box.current!.style.cursor = ""));
    cy.current = instance;
    return () => {
      instance.destroy();
      cy.current = null;
    };
  }, [center, relations]);

  useEffect(() => {
    const c = cy.current;
    if (!c) return;
    c.edges().removeClass("selected");
    if (selectedEdge) c.getElementById(selectedEdge).addClass("selected");
  }, [selectedEdge, relations]);

  // 이웃이 많으면 그림을 키운다 (너무 작게 줄어 이름을 못 읽지 않게)
  const nOthers = new Set(relations.map((r) => r.other_id)).size;
  const height = Math.min(760, Math.max(440, 300 + nOthers * 7));
  return (
    <div className="ego-graph">
      <div ref={box} className="ego-canvas" style={{ height }} role="img" aria-label={`${center.name}의 관계도`} />
      <div className="ego-legend small">
        {Object.entries(RELATION).map(([k, v]) => (
          <span key={k}>
            <i style={{ background: RELATION_COLOR[k] }} /> {v}
          </span>
        ))}
        <span>
          <i className="dash" /> 검수 대기
        </span>
        <span>
          <i className="faint" /> 과거 관계
        </span>
        <span className="muted">점 색은 GICS 섹터 · 점을 누르면 그 회사로 이동</span>
      </div>
    </div>
  );
}

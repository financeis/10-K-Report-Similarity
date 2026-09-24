import { useEffect, useState } from "react";

// 주소 (모두 인코딩):
// 회사:      #/node/<node_id>                    관계 화면
//            #/node/<node_id>?edge=<edge_id>     관계 하나를 고른 상태
//            #/node/<node_id>?tab=candidates&pair=<pair_key>   판정 전 후보 화면
// 관계 검수: #/queue, #/queue?edge=<edge_id>
// 표본 검수: #/review, #/review/<sample_id>/<ord>
export type NodeTab = "relations" | "candidates";

export interface Route {
  node: string | null;
  tab: NodeTab;
  pair: string | null;
  edge: string | null;
  queue: boolean;
  review: boolean;
  sample: string | null;
  ord: number | null;
}

function parseHash(): Route {
  const hash = window.location.hash.replace(/^#/, "");
  const [path, query = ""] = hash.split("?");
  const m = path.match(/^\/node\/(.+)$/);
  const r = path.match(/^\/review(?:\/([^/]+)(?:\/(\d+))?)?$/);
  const params = new URLSearchParams(query);
  const pair = params.get("pair");
  return {
    node: m ? decodeURIComponent(m[1]) : null,
    // 예전 주소(?pair=만 있음)도 후보 화면으로 연다
    tab: params.get("tab") === "candidates" || (pair && !params.get("tab")) ? "candidates" : "relations",
    pair,
    edge: params.get("edge"),
    queue: path === "/queue",
    review: !!r,
    sample: r?.[1] ? decodeURIComponent(r[1]) : null,
    ord: r?.[2] ? Number(r[2]) : null,
  };
}

export function reviewHref(sample?: string, ord?: number): string {
  if (!sample) return "#/review";
  return ord == null ? `#/review/${encodeURIComponent(sample)}` : `#/review/${encodeURIComponent(sample)}/${ord}`;
}

export function queueHref(edge?: string | null): string {
  return edge ? `#/queue?edge=${encodeURIComponent(edge)}` : "#/queue";
}

/** 회사 관계 화면. edge를 주면 그 관계를 고른 상태로 연다. */
export function href(node: string, edge?: string | null): string {
  const base = `#/node/${encodeURIComponent(node)}`;
  return edge ? `${base}?edge=${encodeURIComponent(edge)}` : base;
}

/** 회사의 판정 전 후보 화면. */
export function candidatesHref(node: string, pair?: string | null): string {
  const base = `#/node/${encodeURIComponent(node)}?tab=candidates`;
  return pair ? `${base}&pair=${encodeURIComponent(pair)}` : base;
}

export function useRoute(): Route {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export type Loadable<T> = { data?: T; error?: string; loading: boolean };

/** key가 바뀔 때마다 다시 불러온다. key가 null이면 부르지 않는다. */
export function useFetch<T>(key: string | null, load: () => Promise<T>): Loadable<T> {
  const [state, setState] = useState<Loadable<T>>({ loading: key !== null });
  useEffect(() => {
    if (key === null) {
      setState({ loading: false });
      return;
    }
    let alive = true;
    setState((s) => ({ data: s.data, loading: true }));
    load().then(
      (data) => alive && setState({ data, loading: false }),
      (e: Error) => alive && setState({ error: e.message, loading: false }),
    );
    return () => {
      alive = false;
    };
    // load는 key로 결정된다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}

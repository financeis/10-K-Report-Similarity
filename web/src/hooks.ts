import { useEffect, useState } from "react";

// 주소: #/node/<node_id>?pair=<pair_key>  (둘 다 인코딩)
// 검수: #/review, #/review/<sample_id>/<ord>
export interface Route {
  node: string | null;
  pair: string | null;
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
  return {
    node: m ? decodeURIComponent(m[1]) : null,
    pair: params.get("pair"),
    review: !!r,
    sample: r?.[1] ? decodeURIComponent(r[1]) : null,
    ord: r?.[2] ? Number(r[2]) : null,
  };
}

export function reviewHref(sample?: string, ord?: number): string {
  if (!sample) return "#/review";
  return ord == null ? `#/review/${encodeURIComponent(sample)}` : `#/review/${encodeURIComponent(sample)}/${ord}`;
}

export function href(node: string, pair?: string | null): string {
  const base = `#/node/${encodeURIComponent(node)}`;
  return pair ? `${base}?pair=${encodeURIComponent(pair)}` : base;
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

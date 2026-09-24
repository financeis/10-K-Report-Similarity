// 텍스트에 강조 구간을 칠한다. 구간이 겹치면 뒤에 오는 종류(이름)가 앞(구간)을 덮는다.

export interface Range {
  start: number;
  end: number;
  className: string;
  id?: string;
}

export function Highlighted({ text, ranges }: { text: string; ranges: Range[] }) {
  const cuts = new Set<number>([0, text.length]);
  for (const r of ranges) {
    cuts.add(Math.max(0, Math.min(text.length, r.start)));
    cuts.add(Math.max(0, Math.min(text.length, r.end)));
  }
  const points = [...cuts].sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < points.length - 1; i++) {
    const [s, e] = [points[i], points[i + 1]];
    if (s === e) continue;
    const covering = ranges.filter((r) => r.start <= s && e <= r.end);
    const piece = text.slice(s, e);
    if (!covering.length) {
      parts.push(piece);
      continue;
    }
    const classes = covering.map((r) => r.className).join(" ");
    const id = covering.find((r) => r.id && r.start === s)?.id;
    parts.push(
      <mark key={s} className={classes} id={id}>
        {piece}
      </mark>,
    );
  }
  return <>{parts}</>;
}

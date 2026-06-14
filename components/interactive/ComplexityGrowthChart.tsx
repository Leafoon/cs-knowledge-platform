'use client';

import { useState, useRef, useEffect } from 'react';

const COMPLEXITIES = [
  { label: 'O(1)', color: '#22c55e', fn: (_n: number) => 1, desc: '常数' },
  { label: 'O(log n)', color: '#3b82f6', fn: (n: number) => Math.log2(n), desc: '对数' },
  { label: 'O(n)', color: '#8b5cf6', fn: (n: number) => n, desc: '线性' },
  { label: 'O(n log n)', color: '#f59e0b', fn: (n: number) => n * Math.log2(n), desc: '线性对数' },
  { label: 'O(n²)', color: '#f97316', fn: (n: number) => n * n, desc: '平方' },
  { label: 'O(n³)', color: '#ef4444', fn: (n: number) => n * n * n, desc: '立方' },
  { label: 'O(2ⁿ)', color: '#6b7280', fn: (n: number) => Math.pow(2, n), desc: '指数' },
];

const WIDTH = 520;
const HEIGHT = 300;
const PAD = { top: 20, right: 20, bottom: 40, left: 56 };

export default function ComplexityGrowthChart() {
  const [maxN, setMaxN] = useState(20);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [visible, setVisible] = useState<boolean[]>(COMPLEXITIES.map(() => true));
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const toggle = (i: number) => {
    setVisible(v => v.map((b, idx) => (idx === i ? !b : b)));
  };

  // Build data points
  const ns = Array.from({ length: maxN }, (_, i) => i + 1);

  const vals = COMPLEXITIES.map(c => ns.map(n => c.fn(n)));

  // Compute y domain: max of *visible* series
  const maxY = Math.max(
    ...COMPLEXITIES.flatMap((_, ci) =>
      visible[ci] ? vals[ci] : [0]
    ),
    1
  );

  const toSVGX = (n: number) =>
    PAD.left + ((n - 1) / (maxN - 1)) * (WIDTH - PAD.left - PAD.right);
  const toSVGY = (v: number) =>
    PAD.top + (1 - Math.min(v, maxY) / maxY) * (HEIGHT - PAD.top - PAD.bottom);

  const buildPath = (fnVals: number[]) => {
    return fnVals
      .map((v, i) => {
        const x = toSVGX(i + 1);
        const y = toSVGY(v);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  };

  // Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(r => ({
    r,
    v: maxY * r,
    y: toSVGY(maxY * r),
  }));

  // X-axis ticks
  const xTickStep = Math.max(1, Math.floor(maxN / 5));
  const xTicks = ns.filter((_, i) => i % xTickStep === xTickStep - 1 || i === 0);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const rawX = e.clientX - rect.left;
    const n = Math.round(
      ((rawX - PAD.left) / (WIDTH - PAD.left - PAD.right)) * (maxN - 1) + 1
    );
    if (n < 1 || n > maxN) { setTooltip(null); return; }
    const lines = COMPLEXITIES
      .filter((_, ci) => visible[ci])
      .map(c => `${c.label}: ${c.fn(n) >= 1e9 ? '>1B' : c.fn(n) < 0.01 ? '<0.01' : c.fn(n).toFixed(1)}`);
    setTooltip({ x: rawX + 8, y: e.clientY - rect.top - 8, text: `n=${n}\n${lines.join('\n')}` });
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-4 space-y-3">
      <h3 className="text-base font-semibold text-text-primary">📈 复杂度增长曲线对比</h3>

      {/* Legend toggles */}
      <div className="flex flex-wrap gap-2">
        {COMPLEXITIES.map((c, i) => (
          <button
            key={c.label}
            onClick={() => toggle(i)}
            className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono border transition-opacity"
            style={{
              borderColor: c.color,
              color: visible[i] ? c.color : '#6b7280',
              opacity: visible[i] ? 1 : 0.4,
              backgroundColor: visible[i] ? `${c.color}18` : 'transparent',
            }}
          >
            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: visible[i] ? c.color : '#6b7280' }} />
            {c.label}
          </button>
        ))}
      </div>

      {/* N slider */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-text-tertiary w-24">最大 n = {maxN}</span>
        <input
          type="range"
          min={5}
          max={30}
          value={maxN}
          onChange={e => setMaxN(Number(e.target.value))}
          className="flex-1 accent-blue-500"
        />
      </div>

      {/* SVG Chart */}
      <div className="relative overflow-hidden rounded-lg bg-bg-primary border border-border-subtle">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        >
          {/* Grid lines */}
          {yTicks.map(t => (
            <g key={t.r}>
              <line x1={PAD.left} x2={WIDTH - PAD.right} y1={t.y} y2={t.y} stroke="currentColor" strokeOpacity={0.08} />
              <text x={PAD.left - 4} y={t.y + 4} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.5}>
                {t.v >= 1e9 ? `${(t.v / 1e9).toFixed(1)}B` : t.v >= 1e6 ? `${(t.v / 1e6).toFixed(1)}M` : t.v >= 1000 ? `${(t.v / 1000).toFixed(0)}K` : t.v.toFixed(0)}
              </text>
            </g>
          ))}
          {xTicks.map(n => (
            <g key={n}>
              <line x1={toSVGX(n)} x2={toSVGX(n)} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="currentColor" strokeOpacity={0.08} />
              <text x={toSVGX(n)} y={HEIGHT - PAD.bottom + 14} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.5}>{n}</text>
            </g>
          ))}

          {/* Axes */}
          <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="currentColor" strokeOpacity={0.3} />
          <line x1={PAD.left} x2={WIDTH - PAD.right} y1={HEIGHT - PAD.bottom} y2={HEIGHT - PAD.bottom} stroke="currentColor" strokeOpacity={0.3} />

          {/* Axis labels */}
          <text x={(WIDTH + PAD.left) / 2} y={HEIGHT - 2} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.5}>n（输入规模）</text>
          <text x={8} y={(HEIGHT + PAD.top) / 2} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.5} transform={`rotate(-90, 8, ${(HEIGHT + PAD.top) / 2})`}>操作次数</text>

          {/* Curves */}
          {COMPLEXITIES.map((c, ci) => (
            visible[ci] && (
              <path
                key={c.label}
                d={buildPath(vals[ci])}
                fill="none"
                stroke={c.color}
                strokeWidth={hoveredIdx === ci ? 3 : 1.8}
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity={hoveredIdx === null || hoveredIdx === ci ? 1 : 0.3}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredIdx(ci)}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            )
          ))}
        </svg>

        {/* Tooltip */}
        {tooltip && (
          <div
            className="absolute pointer-events-none rounded bg-bg-tertiary border border-border-subtle p-2 text-xs font-mono text-text-primary whitespace-pre shadow-lg z-10"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.text}
          </div>
        )}
      </div>

      {/* Info bar */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-tertiary">
        {COMPLEXITIES.map((c, i) => (
          visible[i] && (
            <span key={c.label} style={{ color: c.color }}>
              {c.label}（{c.desc}）: n={maxN} 时 ≈ {(() => {
                const v = c.fn(maxN);
                return v >= 1e12 ? '>1T' : v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v.toFixed(1);
              })()}
            </span>
          )
        ))}
      </div>
    </div>
  );
}

'use client';
import React, { useState, useMemo } from 'react';

// =====================================================
// HarmonicSeriesDemo — DSA Chapter 0
// 调和级数 H_n = 1 + 1/2 + ... + 1/n 增长速度可视化
// =====================================================

function harmonicSum(n: number): number {
  let sum = 0;
  for (let k = 1; k <= n; k++) sum += 1 / k;
  return sum;
}

// 生成折线图数据点（n=1..maxN，每隔 step 取一个点）
function buildPoints(maxN: number): { n: number; hn: number; lnn: number }[] {
  const pts: { n: number; hn: number; lnn: number }[] = [];
  const step = maxN <= 100 ? 1 : maxN <= 1000 ? 10 : 100;
  for (let n = 1; n <= maxN; n += step) {
    pts.push({ n, hn: harmonicSum(n), lnn: Math.log(n) });
  }
  if (pts[pts.length - 1]?.n !== maxN) {
    pts.push({ n: maxN, hn: harmonicSum(maxN), lnn: Math.log(maxN) });
  }
  return pts;
}

const PRESETS = [
  { label: 'n = 20', value: 20 },
  { label: 'n = 100', value: 100 },
  { label: 'n = 500', value: 500 },
  { label: 'n = 1000', value: 1000 },
];

// SVG chart dimensions
const W = 520, H = 220, PAD_L = 50, PAD_R = 20, PAD_T = 20, PAD_B = 36;
const CW = W - PAD_L - PAD_R;
const CH = H - PAD_T - PAD_B;

export default function HarmonicSeriesDemo() {
  const [maxN, setMaxN] = useState(100);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const points = useMemo(() => buildPoints(maxN), [maxN]);
  const maxHn = points[points.length - 1]?.hn ?? 1;

  // Map data → SVG coordinates
  const toX = (n: number) => PAD_L + ((n - 1) / (maxN - 1 || 1)) * CW;
  const toY = (v: number) => PAD_T + CH - (v / (maxHn * 1.1)) * CH;

  const polylineHn = points.map(p => `${toX(p.n)},${toY(p.hn)}`).join(' ');
  const polylineLn = points.map(p => `${toX(p.n)},${toY(p.lnn)}`).join(' ');

  // Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map(t => ({
    v: t * maxHn * 1.1,
    y: toY(t * maxHn * 1.1),
  }));

  // X-axis ticks
  const xTicks = [1, Math.round(maxN * 0.25), Math.round(maxN * 0.5), Math.round(maxN * 0.75), maxN];

  // Hovered point
  const hov = hoveredIdx !== null ? points[hoveredIdx] : null;

  // Cumulative terms table (first 10)
  const tableRows = useMemo(() => {
    const rows: { n: number; term: string; hn: string; approx: string }[] = [];
    let sum = 0;
    for (let k = 1; k <= Math.min(12, maxN); k++) {
      sum += 1 / k;
      rows.push({
        n: k,
        term: `1/${k}`,
        hn: sum.toFixed(4),
        approx: (Math.log(k) + 0.5772).toFixed(4),
      });
    }
    return rows;
  }, [maxN]);

  return (
    <div className="my-8 rounded-2xl border border-border-subtle bg-bg-secondary overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border-subtle bg-bg-tertiary flex items-center gap-3">
        <span className="text-2xl">📈</span>
        <div>
          <h3 className="font-bold text-text-primary text-lg">调和级数增长速度</h3>
          <p className="text-sm text-text-tertiary">Hₙ = 1 + 1/2 + 1/3 + … + 1/n ≈ ln n（增长极慢！）</p>
        </div>
      </div>

      {/* Preset buttons */}
      <div className="px-6 pt-4 flex flex-wrap gap-2">
        {PRESETS.map(p => (
          <button
            key={p.value}
            onClick={() => { setMaxN(p.value); setHoveredIdx(null); }}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
              maxN === p.value
                ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40'
                : 'bg-bg-tertiary text-text-tertiary border-border-subtle hover:border-indigo-400/40 hover:text-text-secondary'
            }`}
          >
            {p.label}
          </button>
        ))}
        <div className="flex items-center gap-2 ml-auto">
          <label className="text-xs text-text-tertiary">自定义 n =</label>
          <input
            type="number"
            min={2} max={5000}
            value={maxN}
            onChange={e => { const v = Math.max(2, Math.min(5000, parseInt(e.target.value) || 2)); setMaxN(v); setHoveredIdx(null); }}
            className="w-20 px-2 py-1 rounded-lg bg-bg-tertiary border border-border-subtle text-sm text-text-primary focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* SVG Chart */}
      <div className="px-4 pt-3 pb-1">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full rounded-xl bg-bg-tertiary border border-border-subtle"
          style={{ maxHeight: 240 }}
          onMouseLeave={() => setHoveredIdx(null)}
        >
          {/* Grid lines */}
          {yTicks.map((t, i) => (
            <React.Fragment key={i}>
              <line x1={PAD_L} y1={t.y} x2={W - PAD_R} y2={t.y} stroke="currentColor" strokeOpacity="0.07" strokeWidth="1"/>
              <text x={PAD_L - 5} y={t.y + 4} textAnchor="end" fontSize="9" fill="currentColor" opacity="0.5">
                {t.v.toFixed(1)}
              </text>
            </React.Fragment>
          ))}

          {/* X axis ticks */}
          {xTicks.map((n, i) => (
            <g key={i}>
              <line x1={toX(n)} y1={PAD_T + CH} x2={toX(n)} y2={PAD_T + CH + 4} stroke="currentColor" strokeOpacity="0.3" strokeWidth="1"/>
              <text x={toX(n)} y={H - 6} textAnchor="middle" fontSize="9" fill="currentColor" opacity="0.5">{n}</text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={PAD_L + CW / 2} y={H - 1} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.5">n</text>
          <text x={12} y={PAD_T + CH / 2} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.5"
            transform={`rotate(-90, 12, ${PAD_T + CH / 2})`}>值</text>

          {/* ln n line */}
          <polyline
            points={polylineLn}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="1.5"
            strokeDasharray="5 3"
            strokeOpacity="0.7"
          />

          {/* Hₙ line */}
          <polyline
            points={polylineHn}
            fill="none"
            stroke="#6366f1"
            strokeWidth="2"
          />

          {/* Hover overlay: invisible rect to capture mouse */}
          <rect
            x={PAD_L} y={PAD_T} width={CW} height={CH}
            fill="transparent"
            onMouseMove={e => {
              const rect = (e.target as SVGRectElement).getBoundingClientRect();
              const svgX = (e.clientX - rect.left) / rect.width * CW;
              const ratio = svgX / CW;
              const idx = Math.round(ratio * (points.length - 1));
              setHoveredIdx(Math.max(0, Math.min(points.length - 1, idx)));
            }}
          />

          {/* Hover crosshair */}
          {hov && (
            <>
              <line x1={toX(hov.n)} y1={PAD_T} x2={toX(hov.n)} y2={PAD_T + CH} stroke="#6366f1" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="4 2"/>
              <circle cx={toX(hov.n)} cy={toY(hov.hn)} r="4" fill="#6366f1" stroke="white" strokeWidth="1.5"/>
              <circle cx={toX(hov.n)} cy={toY(hov.lnn)} r="3" fill="#f59e0b" stroke="white" strokeWidth="1.5"/>
              {/* Tooltip box */}
              {(() => {
                const tx = toX(hov.n) + (toX(hov.n) > W * 0.7 ? -110 : 10);
                const ty = PAD_T + 10;
                return (
                  <g>
                    <rect x={tx} y={ty} width={100} height={50} rx="6" fill="#1e1e2e" fillOpacity="0.92" stroke="#6366f1" strokeOpacity="0.4" strokeWidth="1"/>
                    <text x={tx + 8} y={ty + 16} fontSize="10" fill="#a5b4fc">n = {hov.n}</text>
                    <text x={tx + 8} y={ty + 29} fontSize="10" fill="#818cf8">Hₙ = {hov.hn.toFixed(4)}</text>
                    <text x={tx + 8} y={ty + 42} fontSize="10" fill="#fbbf24">ln n = {hov.lnn.toFixed(4)}</text>
                  </g>
                );
              })()}
            </>
          )}

          {/* Legend */}
          <g>
            <line x1={W - PAD_R - 90} y1={PAD_T + 12} x2={W - PAD_R - 75} y2={PAD_T + 12} stroke="#6366f1" strokeWidth="2"/>
            <text x={W - PAD_R - 72} y={PAD_T + 16} fontSize="9" fill="#a5b4fc">Hₙ（精确）</text>
            <line x1={W - PAD_R - 90} y1={PAD_T + 26} x2={W - PAD_R - 75} y2={PAD_T + 26} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 2" strokeOpacity="0.8"/>
            <text x={W - PAD_R - 72} y={PAD_T + 30} fontSize="9" fill="#fcd34d">ln n（近似）</text>
          </g>
        </svg>
      </div>

      {/* Info Cards */}
      <div className="px-4 py-3 grid grid-cols-3 gap-3">
        {[
          { label: `H(${maxN})`, value: harmonicSum(maxN).toFixed(4), sub: '调和级数精确值', color: 'text-indigo-400' },
          { label: `ln(${maxN})`, value: Math.log(maxN).toFixed(4), sub: '自然对数近似值', color: 'text-amber-400' },
          { label: '误差', value: (harmonicSum(maxN) - Math.log(maxN)).toFixed(4), sub: '≈ 0.5772（欧拉常数γ）', color: 'text-emerald-400' },
        ].map(card => (
          <div key={card.label} className="rounded-xl bg-bg-tertiary border border-border-subtle p-3 text-center">
            <div className={`text-xl font-bold font-mono ${card.color}`}>{card.value}</div>
            <div className="text-xs font-semibold text-text-primary mt-0.5">{card.label}</div>
            <div className="text-xs text-text-tertiary mt-0.5">{card.sub}</div>
          </div>
        ))}
      </div>

      {/* Cumulative table */}
      <div className="px-4 pb-4">
        <div className="text-xs font-semibold text-text-tertiary uppercase tracking-wide mb-2 px-1">逐项累加过程（前 {tableRows.length} 项）</div>
        <div className="rounded-xl border border-border-subtle overflow-hidden text-xs">
          <div className="grid grid-cols-4 bg-bg-tertiary border-b border-border-subtle">
            {['k', '第 k 项（1/k）', 'Hₖ（精确）', 'ln k + γ（近似）'].map(h => (
              <div key={h} className="px-3 py-2 font-semibold text-text-tertiary">{h}</div>
            ))}
          </div>
          <div className="divide-y divide-border-subtle max-h-52 overflow-y-auto">
            {tableRows.map(row => (
              <div key={row.n} className="grid grid-cols-4 hover:bg-bg-tertiary/50 transition-colors">
                <div className="px-3 py-1.5 text-indigo-400 font-mono font-semibold">{row.n}</div>
                <div className="px-3 py-1.5 text-text-secondary font-mono">{row.term} = {(1/row.n).toFixed(4)}</div>
                <div className="px-3 py-1.5 text-indigo-300 font-mono">{row.hn}</div>
                <div className="px-3 py-1.5 text-amber-300 font-mono">{row.approx}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Key insight */}
      <div className="mx-4 mb-4 rounded-xl bg-indigo-500/5 border border-indigo-500/20 px-4 py-3">
        <p className="text-sm text-text-secondary leading-relaxed">
          <strong className="text-indigo-400">💡 核心洞察</strong>：调和级数增长速度极慢（比 √n 慢得多），但确实趋向无穷大。
          当 n = 10⁶ 时 Hₙ ≈ 14.4；当 n = 10⁹ 时 Hₙ ≈ 21.3。
          快速排序的平均比较次数 ≈ 2n·Hₙ ≈ 2n·ln n，这正是 <strong className="text-indigo-300">O(n log n)</strong> 的来源。
        </p>
      </div>
    </div>
  );
}

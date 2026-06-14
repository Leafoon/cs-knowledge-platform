'use client'

import { useState } from 'react'

// For a KD-tree NN search, the fraction of nodes visited grows rapidly with dimension
// Approximation: fraction visited ≈ 1 - (1 - r^d)^n where r is ratio of NN radius to space size
// More practically: empirical data shows it degrades to O(n) around d~20 for typical data
// We'll use a known empirical model: fraction ≈ min(1, (d/2)^2 / n) or just illustrate conceptually

function fractionVisited(d: number, n: number): number {
  // Empirical formula: in d dimensions, ratio of hypersphere to hypercube volume decreases rapidly
  // V(sphere, d) / V(cube) = π^(d/2) / (Γ(d/2+1) * 2^d)
  // As proxy: fraction of space within distance r grows as C(d)*r^d, making NN radius cover less
  // Simplified: fraction of nodes that must be visited ≈ min(1, 2^d / n)
  return Math.min(1, Math.pow(d, 1.8) / (n * 0.8))
}

function distConcentration(d: number): number {
  // (max_dist - min_dist) / min_dist ≈ 1/sqrt(n^(1/d) - 1) ≈ O(1/sqrt(d))
  // As d increases, this ratio → 0, meaning all points are equidistant
  return Math.max(0.01, 1 / Math.sqrt(d))
}

const N_VALUES = [100, 1000, 10000]
const DIM_RANGE = Array.from({ length: 20 }, (_, i) => i + 1)

const PALETTE = ['#6366f1', '#f97316', '#10b981']

export function KDTreeCurseOfDimensionality() {
  const [nIdx, setNIdx] = useState(1)    // n=1000 default
  const [metric, setMetric] = useState<'visited' | 'concentration'>('visited')
  const n = N_VALUES[nIdx]

  const W = 340, H = 200
  const PAD = { left: 40, right: 16, top: 16, bottom: 32 }
  const chartW = W - PAD.left - PAD.right
  const chartH = H - PAD.top - PAD.bottom

  function dataForN(ni: number) {
    return DIM_RANGE.map(d => metric === 'visited' ? fractionVisited(d, N_VALUES[ni]) : distConcentration(d))
  }

  const allSeries = N_VALUES.map((_, ni) => dataForN(ni))
  const maxVal = Math.max(...allSeries.flat(), 0.01)

  function toX(d: number) { return PAD.left + ((d - 1) / (DIM_RANGE.length - 1)) * chartW }
  function toY(v: number) { return H - PAD.bottom - Math.min(v / maxVal, 1) * chartH }

  function makePath(data: number[]) {
    return data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i + 1).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ')
  }

  const ANNOTATIONS = metric === 'visited'
    ? [
        { d: 2,  label: 'd=2: 高效', y: -14 },
        { d: 10, label: 'd=10: 退化明显', y: -14 },
        { d: 20, label: 'd=20: 接近暴力搜索', y: -14 },
      ]
    : [
        { d: 1,  label: '低维：距离差异大', y: -14 },
        { d: 10, label: '高维：距离趋于相等', y: -14 },
      ]

  return (
    <div className="rounded-2xl border border-rose-200 dark:border-rose-800 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-rose-600 to-pink-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📉 维度灾难：KD-Tree 在高维下的退化</h3>
        <p className="text-rose-50 text-xs mt-0.5">
          随维度 d 增大，最近邻搜索的剪枝效率急速下降，趋向 O(n) 暴力搜索
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          <div className="flex gap-1">
            {(['visited', 'concentration'] as const).map(m => (
              <button key={m} onClick={() => setMetric(m)}
                className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
                  metric === m ? 'bg-white text-rose-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
                }`}>
                {m === 'visited' ? '节点访问率' : '距离集中现象'}
              </button>
            ))}
          </div>
          {metric === 'visited' && (
            <div className="flex gap-1 ml-2">
              {N_VALUES.map((nv, i) => (
                <button key={i} onClick={() => setNIdx(i)}
                  className={`px-2.5 py-1 text-xs rounded-lg font-mono transition-colors ${
                    nIdx === i ? 'bg-white text-rose-700 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
                  }`}>
                  n={nv}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-4">
        <div className="overflow-x-auto">
          <svg width={W} height={H} className="mx-auto block">
            {/* Grid */}
            {[0, 0.25, 0.5, 0.75, 1].map(frac => {
              const y = H - PAD.bottom - frac * chartH
              const label = metric === 'visited' ? `${Math.round(frac * 100)}%` : frac.toFixed(2)
              return (
                <g key={frac}>
                  <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="#e2e8f0" strokeWidth={0.8} />
                  <text x={PAD.left - 4} y={y + 3} textAnchor="end" fontSize={8} fill="#94a3b8">{label}</text>
                </g>
              )
            })}
            {[1, 5, 10, 15, 20].map(d => {
              const x = toX(d)
              return (
                <g key={d}>
                  <line x1={x} y1={PAD.top} x2={x} y2={H - PAD.bottom} stroke="#e2e8f0" strokeWidth={0.8} />
                  <text x={x} y={H - PAD.bottom + 12} textAnchor="middle" fontSize={8} fill="#94a3b8">{d}</text>
                </g>
              )
            })}

            {/* Danger zone */}
            <rect x={toX(15)} y={PAD.top} width={toX(20) - toX(15)} height={chartH}
              fill="#fee2e2" opacity={0.3} />
            <text x={toX(17.5)} y={PAD.top + 10} textAnchor="middle" fontSize={8} fill="#dc2626" opacity={0.7}>危险区</text>

            {/* Lines */}
            {(metric === 'visited' ? N_VALUES.map((_, ni) => ni) : [nIdx]).map(ni => (
              <path key={ni} d={makePath(allSeries[ni])}
                fill="none" stroke={PALETTE[ni]} strokeWidth={2.5}
                strokeLinecap="round" strokeLinejoin="round" />
            ))}

            {/* Current n highlight point */}
            {metric === 'visited' && DIM_RANGE.map(d => {
              const v = fractionVisited(d, n)
              return (
                <circle key={d} cx={toX(d)} cy={toY(v)} r={d % 5 === 0 ? 3.5 : 0}
                  fill={PALETTE[nIdx]} />
              )
            })}

            {/* Axes labels */}
            <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="#64748b">维度 d</text>
            <text x={10} y={H / 2} textAnchor="middle" fontSize={9} fill="#64748b"
              transform={`rotate(-90, 10, ${H / 2})`}>
              {metric === 'visited' ? '节点访问率' : '距离相对差异'}
            </text>
          </svg>
        </div>

        {/* Legend */}
        {metric === 'visited' && (
          <div className="flex gap-4 justify-center text-xs">
            {N_VALUES.map((nv, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className="w-4 h-1.5 rounded" style={{ backgroundColor: PALETTE[i] }} />
                <span className="text-slate-600 dark:text-slate-400">n={nv}</span>
              </div>
            ))}
          </div>
        )}

        {/* Key insights */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {[
            { d: '≤ 5', desc: 'KD-Tree 有效', detail: 'O(log n)~O(n^{1-1/d})', color: 'emerald', icon: '✅' },
            { d: '5~15', desc: '效率开始退化', detail: '剪枝比例下降显著', color: 'yellow', icon: '⚠️' },
            { d: '> 20', desc: '趋向 O(n)', detail: '建议换 LSH/HNSW', color: 'rose', icon: '❌' },
          ].map(({ d, desc, detail, color, icon }) => (
            <div key={d} className={`rounded-xl p-3 border text-xs bg-${color}-50 dark:bg-${color}-900/10 border-${color}-200 dark:border-${color}-800`}>
              <p className={`font-bold text-${color}-700 dark:text-${color}-300`}>{icon} d {d}</p>
              <p className={`text-${color}-600 dark:text-${color}-400 font-medium mt-0.5`}>{desc}</p>
              <p className="text-slate-500 dark:text-slate-400 mt-0.5 text-[10px]">{detail}</p>
            </div>
          ))}
        </div>

        <p className="text-[11px] text-slate-400 dark:text-slate-600 text-center border-t border-slate-100 dark:border-slate-800 pt-3">
          高维近似最近邻（ANN）解法：LSH（局部敏感哈希）、HNSW（分层导航小世界图）、乘积量化（Product Quantization）
        </p>
      </div>
    </div>
  )
}

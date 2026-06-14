'use client'

import { useState } from 'react'

// Nodes: problem size at each recursion level
// T(u) = T(sqrt(u)) + O(1)
// Sizes: u, sqrt(u), u^(1/4), u^(1/8), ... until ≤ 2
function buildLevels(u: number): number[] {
  const levels: number[] = [u]
  let cur = u
  while (cur > 2) {
    cur = Math.round(Math.pow(cur, 0.5))
    levels.push(cur)
  }
  return levels
}

const PRESETS = [
  { label: 'u = 65536', u: 65536 },
  { label: 'u = 256',   u: 256 },
  { label: 'u = 16',    u: 16 },
]

export function VEBRecursionTree() {
  const [preset, setPreset] = useState(0)
  const [hovered, setHovered] = useState<number | null>(null)
  const { u } = PRESETS[preset]
  const levels = buildLevels(u)
  const depth = levels.length - 1

  const nodeW = 88
  const nodeH = 36

  // Compute widths for tree layout
  function nodesAtLevel(d: number) {
    // vEB has branching factor sqrt(u) at level 0
    // but our recursion here is a single chain (one T(sqrt(u)) call)
    // So it's a straight chain, not a proper tree
    return 1
  }

  const svgH = levels.length * 80 + 20
  const svgW = 400

  return (
    <div className="rounded-2xl border border-orange-200 dark:border-orange-800 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-orange-500 to-amber-400 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔁 vEB 递推树：T(u) = T(√u) + O(1)</h3>
        <p className="text-orange-50 text-xs mt-0.5">
          每层子问题规模开方，悬停任一节点查看累计递推深度
        </p>
        <div className="flex gap-2 mt-3">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => setPreset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono transition-all ${
                preset === i ? 'bg-white text-orange-600 font-bold' : 'bg-white/25 text-white hover:bg-white/35'
              }`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5">
        <div className="overflow-x-auto">
          <svg width={svgW} height={svgH} className="mx-auto block">
            {levels.map((size, d) => {
              const cx = svgW / 2
              const cy = d * 80 + 40
              const isHov = hovered === d
              const color = d === depth ? '#10b981' : d === 0 ? '#f97316' : '#6366f1'
              const label = d === depth ? 'BASE' : `T(${size})`
              const sub = d < levels.length - 1
                ? `= T(${levels[d+1]}) + O(1)`
                : `= O(1)  ← 基底`

              return (
                <g key={d}>
                  {d < levels.length - 1 && (
                    <line
                      x1={cx} y1={cy + nodeH / 2}
                      x2={cx} y2={cy + 80 - nodeH / 2}
                      stroke={isHov ? color : '#94a3b8'}
                      strokeWidth={isHov ? 2.5 : 1.5}
                      strokeDasharray="4 2"
                    />
                  )}
                  <g
                    onMouseEnter={() => setHovered(d)}
                    onMouseLeave={() => setHovered(null)}
                    style={{ cursor: 'pointer' }}>
                    <rect
                      x={cx - nodeW / 2} y={cy - nodeH / 2}
                      width={nodeW} height={nodeH} rx={8}
                      fill={isHov ? color : d === depth ? '#d1fae5' : d === 0 ? '#fed7aa' : '#e0e7ff'}
                      stroke={color} strokeWidth={isHov ? 2.5 : 1.5}
                      className="dark:opacity-90"
                    />
                    <text x={cx} y={cy - 2}
                      textAnchor="middle" fontSize={d === 0 ? 13 : 11}
                      fontWeight="bold"
                      fill={isHov ? 'white' : d === depth ? '#065f46' : d === 0 ? '#92400e' : '#3730a3'}>
                      {label}
                    </text>
                    <text x={cx} y={cy + 11}
                      textAnchor="middle" fontSize={9}
                      fill={isHov ? 'rgba(255,255,255,0.8)' : '#64748b'}>
                      size = {size}
                    </text>
                  </g>
                  {/* Right label */}
                  <text x={cx + nodeW / 2 + 8} y={cy + 5}
                    fontSize={9} fill="#94a3b8" fontFamily="monospace">
                    {sub}
                  </text>
                  {/* Depth marker */}
                  <text x={cx - nodeW / 2 - 8} y={cy + 5}
                    textAnchor="end" fontSize={9} fill="#94a3b8">
                    d={d}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-3 text-center text-xs">
          <div className="rounded-xl bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 py-2 px-3">
            <p className="text-orange-700 dark:text-orange-300 font-bold text-lg">{depth}</p>
            <p className="text-orange-600 dark:text-orange-400 text-[10px] mt-0.5">递归深度 log log u</p>
          </div>
          <div className="rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 py-2 px-3">
            <p className="text-indigo-700 dark:text-indigo-300 font-bold text-lg">O(log log {u})</p>
            <p className="text-indigo-600 dark:text-indigo-400 text-[10px] mt-0.5">≈ O({depth}) 每次操作</p>
          </div>
          <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 py-2 px-3">
            <p className="text-emerald-700 dark:text-emerald-300 font-bold text-lg">{levels[levels.length - 1]}</p>
            <p className="text-emerald-600 dark:text-emerald-400 text-[10px] mt-0.5">基底规模（≤ 2）</p>
          </div>
        </div>

        <p className="mt-3 text-[11px] text-slate-400 dark:text-slate-600 text-center">
          对比：BST 的 O(log n)，当 u=2^64，log log u = 6；而 log u = 64。vEB 极大加速了大整数集合操作。
        </p>
      </div>
    </div>
  )
}

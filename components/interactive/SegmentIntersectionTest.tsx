'use client'

import { useState } from 'react'

interface Pt { x: number; y: number }

const W = 340, H = 280
const PAD = 30

// Presets: [A, B, C, D] as {x,y} in [0,10] range
const PRESETS = [
  { label: '✅ 正常相交', A: {x:1,y:2}, B: {x:7,y:8}, C: {x:1,y:8}, D: {x:9,y:1}, desc: '两线段真正交叉，叉积乘积 < 0' },
  { label: '❌ 不相交', A: {x:1,y:7}, B: {x:4,y:9}, C: {x:5,y:2}, D: {x:9,y:6}, desc: 'C、D 在 AB 同侧，乘积 > 0，不相交' },
  { label: '🔀 T形（端点）', A: {x:2,y:5}, B: {x:8,y:5}, C: {x:5,y:1}, D: {x:5,y:5}, desc: 'D 恰好落在 AB 上，共线退化情形' },
  { label: '📏 共线重叠', A: {x:1,y:3}, B: {x:6,y:3}, C: {x:4,y:3}, D: {x:9,y:3}, desc: '四点共线，线段部分重叠' },
  { label: '⚡ 平行不交', A: {x:1,y:2}, B: {x:7,y:2}, C: {x:1,y:5}, D: {x:7,y:5}, desc: '两线段平行，叉积均为 0 但不重叠' },
]

function cross(O: Pt, A: Pt, B: Pt): number {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
}
function onSegment(pi: Pt, pj: Pt, pk: Pt): boolean {
  return (Math.min(pi.x, pj.x) <= pk.x && pk.x <= Math.max(pi.x, pj.x) &&
          Math.min(pi.y, pj.y) <= pk.y && pk.y <= Math.max(pi.y, pj.y))
}
function segmentsIntersect(A: Pt, B: Pt, C: Pt, D: Pt): boolean {
  const d1 = cross(C, D, A), d2 = cross(C, D, B)
  const d3 = cross(A, B, C), d4 = cross(A, B, D)
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true
  if (d1 === 0 && onSegment(C, D, A)) return true
  if (d2 === 0 && onSegment(C, D, B)) return true
  if (d3 === 0 && onSegment(A, B, C)) return true
  if (d4 === 0 && onSegment(A, B, D)) return true
  return false
}

function toSvg(p: Pt): Pt {
  return { x: PAD + p.x / 10 * (W - 2 * PAD), y: H - PAD - p.y / 10 * (H - 2 * PAD) }
}

function signLabel(v: number): string {
  if (v > 0) return `+${v.toFixed(0)} (左侧)`
  if (v < 0) return `${v.toFixed(0)} (右侧)`
  return `0 (共线)`
}
function signColor(v: number): string {
  if (v > 0) return 'text-emerald-600 dark:text-emerald-400'
  if (v < 0) return 'text-rose-600 dark:text-rose-400'
  return 'text-slate-500 dark:text-slate-400'
}

export function SegmentIntersectionTest() {
  const [presetIdx, setPresetIdx] = useState(0)
  const preset = PRESETS[presetIdx]
  const { A, B, C, D } = preset

  const d1 = cross(C, D, A)
  const d2 = cross(C, D, B)
  const d3 = cross(A, B, C)
  const d4 = cross(A, B, D)
  const intersects = segmentsIntersect(A, B, C, D)

  const sA = toSvg(A), sB = toSvg(B), sC = toSvg(C), sD = toSvg(D)

  // Find visual intersection point
  let ipt: Pt | null = null
  if (intersects && d1 * d2 < 0 && d3 * d4 < 0) {
    const t = d1 / (d1 - d2)
    ipt = { x: A.x + t * (B.x - A.x), y: A.y + t * (B.y - A.y) }
  }

  return (
    <div className="rounded-2xl border border-teal-200 dark:border-teal-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-teal-600 to-emerald-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔀 线段交叉判断：叉积方向测试完整演示</h3>
        <p className="text-teal-50 text-xs mt-0.5">包含正常情形与退化情形（端点重合、共线）的完整判断逻辑</p>
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => setPresetIdx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
                presetIdx === i ? 'bg-white text-teal-700 font-bold' : 'bg-white/20 text-white hover:bg-white/30'
              }`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">{preset.desc}</p>

        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Grid */}
            {Array.from({ length: 11 }, (_, i) => (
              <g key={i}>
                <line x1={PAD + i*(W-2*PAD)/10} y1={PAD} x2={PAD + i*(W-2*PAD)/10} y2={H-PAD} stroke="#e2e8f0" strokeWidth={0.6} className="dark:stroke-slate-700" />
                <line x1={PAD} y1={H-PAD-i*(H-2*PAD)/10} x2={W-PAD} y2={H-PAD-i*(H-2*PAD)/10} stroke="#e2e8f0" strokeWidth={0.6} className="dark:stroke-slate-700" />
              </g>
            ))}

            {/* Segment AB */}
            <line x1={sA.x} y1={sA.y} x2={sB.x} y2={sB.y}
              stroke={intersects ? '#6366f1' : '#6366f1'} strokeWidth={3} strokeLinecap="round"
              opacity={0.9} />
            {/* Segment CD */}
            <line x1={sC.x} y1={sC.y} x2={sD.x} y2={sD.y}
              stroke="#f97316" strokeWidth={3} strokeLinecap="round" opacity={0.9} />

            {/* Intersection point */}
            {ipt && (
              <g>
                <circle cx={toSvg(ipt).x} cy={toSvg(ipt).y} r={12} fill="#10b981" opacity={0.2} />
                <circle cx={toSvg(ipt).x} cy={toSvg(ipt).y} r={6} fill="#10b981" stroke="white" strokeWidth={2} />
              </g>
            )}

            {/* Points */}
            {[{p: sA, label: 'A', c: '#6366f1'}, {p: sB, label: 'B', c: '#6366f1'},
              {p: sC, label: 'C', c: '#f97316'}, {p: sD, label: 'D', c: '#f97316'}]
              .map(({ p, label, c }) => (
              <g key={label}>
                <circle cx={p.x} cy={p.y} r={6} fill={c} stroke="white" strokeWidth={2} />
                <text x={p.x + 8} y={p.y - 6} fontSize={12} fontWeight="bold" fill={c}>{label}</text>
              </g>
            ))}

            {/* Result badge */}
            <rect x={W-100} y={8} width={90} height={26} rx={8}
              fill={intersects ? '#10b981' : '#ef4444'} />
            <text x={W-55} y={26} textAnchor="middle" fontSize={11} fontWeight="bold" fill="white">
              {intersects ? '✓ 相交' : '✗ 不相交'}
            </text>
          </svg>

          {/* Direction table */}
          <div className="flex-1 min-w-[190px] space-y-2">
            <p className="text-xs font-bold text-slate-700 dark:text-slate-200">方向测试结果</p>
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
              {[
                { label: 'd1 = cross(C,D,A)', val: d1, hint: 'A 相对直线 CD' },
                { label: 'd2 = cross(C,D,B)', val: d2, hint: 'B 相对直线 CD' },
                { label: 'd3 = cross(A,B,C)', val: d3, hint: 'C 相对直线 AB' },
                { label: 'd4 = cross(A,B,D)', val: d4, hint: 'D 相对直线 AB' },
              ].map((row, i) => (
                <div key={i} className={`px-3 py-2 ${i % 2 ? 'bg-slate-50 dark:bg-slate-800/50' : ''}`}>
                  <div className="font-mono text-slate-500 dark:text-slate-400 text-[10px]">{row.label}</div>
                  <div className={`font-bold mt-0.5 ${signColor(row.val)}`}>{signLabel(row.val)}</div>
                  <div className="text-slate-400 text-[10px]">{row.hint}</div>
                </div>
              ))}
            </div>

            {/* Logic */}
            <div className={`rounded-xl p-3 border text-xs ${
              intersects
                ? 'bg-emerald-50 dark:bg-emerald-900/10 border-emerald-200 dark:border-emerald-800'
                : 'bg-rose-50 dark:bg-rose-900/10 border-rose-200 dark:border-rose-800'
            }`}>
              <p className={`font-bold mb-1 ${intersects ? 'text-emerald-700 dark:text-emerald-300' : 'text-rose-700 dark:text-rose-300'}`}>
                判断过程
              </p>
              <div className={`space-y-1 text-[11px] font-mono ${intersects ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                <p>d1 × d2 = {(d1 * d2).toFixed(0)} {d1 * d2 < 0 ? '< 0 ✓' : '≥ 0'}</p>
                <p>d3 × d4 = {(d3 * d4).toFixed(0)} {d3 * d4 < 0 ? '< 0 ✓' : '≥ 0'}</p>
                {(d1 === 0 || d2 === 0 || d3 === 0 || d4 === 0) && (
                  <p className="text-amber-600 dark:text-amber-400">⚠ 存在共线，检查 ON_SEGMENT</p>
                )}
                <p className="font-bold text-sm mt-1">{intersects ? '→ 相交 ✓' : '→ 不相交 ✗'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

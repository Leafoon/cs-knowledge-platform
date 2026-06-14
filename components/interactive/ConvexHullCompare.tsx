'use client'

import { useState } from 'react'

interface Pt { x: number; y: number }

const POINT_SETS = [
  {
    label: '标准集 (n=12, h=7)',
    pts: [{x:60,y:180},{x:130,y:50},{x:230,y:30},{x:310,y:90},{x:330,y:210},{x:260,y:250},{x:160,y:260},{x:60,y:230},{x:160,y:130},{x:200,y:160},{x:100,y:110},{x:250,y:170}],
  },
  {
    label: '圆形分布 (n=12, h=12)',
    pts: Array.from({length:12}, (_,i) => ({ x: 195 + 120*Math.cos(i*Math.PI/6), y: 150 + 90*Math.sin(i*Math.PI/6) })),
  },
  {
    label: '团簇 (n=14, h=4)',
    pts: [
      {x:60,y:40},{x:65,y:55},{x:55,y:50},{x:70,y:45},
      {x:300,y:40},{x:295,y:50},{x:310,y:55},
      {x:60,y:250},{x:70,y:240},{x:55,y:245},
      {x:300,y:250},{x:305,y:240},{x:290,y:255},{x:310,y:245},
    ],
  },
]

function cross(O: Pt, A: Pt, B: Pt): number {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
}

function grahamHull(pts: Pt[]): { hull: Pt[]; ops: number } {
  if (pts.length < 3) return { hull: pts, ops: pts.length }
  let ops = 0
  const pivot = [...pts].sort((a, b) => a.y - b.y || a.x - b.x)[0]
  const sorted = [...pts].sort((a, b) => {
    ops++
    const aA = Math.atan2(a.y - pivot.y, a.x - pivot.x)
    const bA = Math.atan2(b.y - pivot.y, b.x - pivot.x)
    return aA - bA
  })
  const stack: Pt[] = []
  for (const p of sorted) {
    while (stack.length >= 2 && cross(stack[stack.length-2], stack[stack.length-1], p) <= 0) {
      ops++; stack.pop()
    }
    ops++; stack.push(p)
  }
  return { hull: stack, ops }
}

function jarvisHull(pts: Pt[]): { hull: Pt[]; ops: number } {
  if (pts.length < 3) return { hull: pts, ops: pts.length }
  let ops = 0
  const hull: Pt[] = []
  let l = pts.reduce((m, p) => p.x < m.x ? p : m, pts[0])
  let cur = l
  do {
    hull.push(cur)
    let next = pts[0]
    for (const p of pts) {
      ops++
      if (next === cur || cross(cur, next, p) < 0) next = p
    }
    cur = next
  } while (cur !== l && hull.length <= pts.length)
  return { hull, ops }
}

function hullPath(hull: Pt[]) {
  if (!hull.length) return ''
  return hull.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + 'Z'
}

export function ConvexHullCompare() {
  const [setIdx, setSetIdx] = useState(0)
  const [show, setShow] = useState<'both' | 'graham' | 'jarvis'>('both')
  const data = POINT_SETS[setIdx]

  const { hull: gh, ops: gOps } = grahamHull(data.pts)
  const { hull: jh, ops: jOps } = jarvisHull(data.pts)
  const n = data.pts.length, hG = gh.length, hJ = jh.length

  return (
    <div className="rounded-2xl border border-violet-200 dark:border-violet-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-violet-600 to-purple-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">⚔️ Graham Scan vs Jarvis March 对比</h3>
        <p className="text-violet-100 text-xs mt-0.5">同一点集，对比两算法结果、时间复杂度与操作次数</p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {POINT_SETS.map((s, i) => (
            <button key={i} onClick={() => setSetIdx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
                setIdx === i ? 'bg-white text-violet-700 font-bold' : 'bg-white/20 text-white hover:bg-white/30'
              }`}>{s.label}</button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        {/* Toggle */}
        <div className="flex gap-2 mb-4 text-xs">
          {(['both', 'graham', 'jarvis'] as const).map(v => (
            <button key={v} onClick={() => setShow(v)}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                show === v ? 'bg-violet-600 text-white' : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
              }`}>
              {v === 'both' ? '并排显示' : v === 'graham' ? 'Graham' : 'Jarvis'}
            </button>
          ))}
        </div>

        {/* SVG panels */}
        <div className={`flex gap-3 flex-wrap ${show === 'both' ? '' : 'justify-center'}`}>
          {(show === 'both' || show === 'graham') && (
            <div className="flex-1 min-w-[180px]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-indigo-700 dark:text-indigo-300">Graham Scan</span>
                <span className="text-[10px] bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 px-2 py-0.5 rounded-full font-mono">O(n log n)</span>
              </div>
              <svg width="100%" viewBox={`0 0 380 280`}
                className="rounded-xl border border-indigo-200 dark:border-indigo-800 bg-slate-50 dark:bg-slate-800">
                <path d={hullPath(gh)} fill="#6366f110" stroke="#6366f1" strokeWidth={2.5} strokeLinejoin="round" />
                {data.pts.map((p, i) => {
                  const isH = gh.some(h => h.x === p.x && h.y === p.y)
                  return (
                    <circle key={i} cx={p.x} cy={p.y} r={isH ? 7 : 4}
                      fill={isH ? '#6366f1' : '#94a3b8'} stroke="white" strokeWidth={1.5} />
                  )
                })}
                <text x={10} y={22} fontSize={11} fill="#6366f1" fontWeight="bold">凸包顶点: {hG} / {n}</text>
                <text x={10} y={38} fontSize={10} fill="#94a3b8">比较次数: ~{gOps}</text>
              </svg>
            </div>
          )}

          {(show === 'both' || show === 'jarvis') && (
            <div className="flex-1 min-w-[180px]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-orange-700 dark:text-orange-300">Jarvis March</span>
                <span className="text-[10px] bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300 px-2 py-0.5 rounded-full font-mono">O(n·h)</span>
              </div>
              <svg width="100%" viewBox="0 0 380 280"
                className="rounded-xl border border-orange-200 dark:border-orange-800 bg-slate-50 dark:bg-slate-800">
                <path d={hullPath(jh)} fill="#f9731610" stroke="#f97316" strokeWidth={2.5} strokeLinejoin="round" />
                {data.pts.map((p, i) => {
                  const isH = jh.some(h => h.x === p.x && h.y === p.y)
                  return (
                    <circle key={i} cx={p.x} cy={p.y} r={isH ? 7 : 4}
                      fill={isH ? '#f97316' : '#94a3b8'} stroke="white" strokeWidth={1.5} />
                  )
                })}
                <text x={10} y={22} fontSize={11} fill="#f97316" fontWeight="bold">凸包顶点: {hJ} / {n}</text>
                <text x={10} y={38} fontSize={10} fill="#94a3b8">比较次数: ~{jOps}</text>
              </svg>
            </div>
          )}
        </div>

        {/* Comparison table */}
        <div className="mt-4 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
          <div className="grid grid-cols-3">
            <div className="bg-slate-100 dark:bg-slate-800 px-3 py-2 font-bold text-slate-500 dark:text-slate-400 text-[10px] uppercase">指标</div>
            <div className="bg-indigo-50 dark:bg-indigo-900/20 px-3 py-2 font-bold text-indigo-700 dark:text-indigo-300 text-center">Graham</div>
            <div className="bg-orange-50 dark:bg-orange-900/20 px-3 py-2 font-bold text-orange-700 dark:text-orange-300 text-center">Jarvis</div>
          </div>
          {[
            { label: '时间复杂度', g: 'O(n log n)', j: 'O(n·h)' },
            { label: '排序预处理', g: '需要（主瓶颈）', j: '不需要' },
            { label: `n=${n}, h=${hG}`, g: `≈${gOps} ops`, j: `≈${jOps} ops` },
            { label: '最优场景', g: 'h ≈ n（所有点在壳上）', j: 'h 极小（如≤log n）' },
            { label: '实现复杂度', g: '中等', j: '简单' },
            { label: '输出敏感性', g: '否（固定 n log n）', j: '是（随 h 变化）' },
          ].map((row, i) => (
            <div key={i} className={`grid grid-cols-3 border-t border-slate-100 dark:border-slate-700 ${i%2===1?'bg-slate-50/50 dark:bg-slate-800/30':''}`}>
              <div className="px-3 py-2 text-slate-600 dark:text-slate-400 font-medium">{row.label}</div>
              <div className={`px-3 py-2 text-center font-mono ${
                row.label.includes('ops') && gOps <= jOps ? 'text-indigo-600 dark:text-indigo-400 font-bold' : 'text-slate-600 dark:text-slate-300'
              }`}>{row.g}</div>
              <div className={`px-3 py-2 text-center font-mono ${
                row.label.includes('ops') && jOps <= gOps ? 'text-orange-600 dark:text-orange-400 font-bold' : 'text-slate-600 dark:text-slate-300'
              }`}>{row.j}</div>
            </div>
          ))}
        </div>

        <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-2">
          💡 当 h ≪ n 时（如固定4顶点），Jarvis 更优；当点均在凸包上时，Graham 更稳定。
        </p>
      </div>
    </div>
  )
}

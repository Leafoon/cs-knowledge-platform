'use client'

import { useState } from 'react'

const W = 380, H = 280

type Pt = { x: number; y: number }
type Seg = { a: Pt; b: Pt; id: number }

const PRESETS: { label: string; segs: Seg[] }[] = [
  {
    label: '无交叉',
    segs: [
      { id: 0, a: { x: 40, y: 60 }, b: { x: 140, y: 200 } },
      { id: 1, a: { x: 160, y: 50 }, b: { x: 240, y: 180 } },
      { id: 2, a: { x: 260, y: 70 }, b: { x: 350, y: 200 } },
    ],
  },
  {
    label: '有交叉',
    segs: [
      { id: 0, a: { x: 60, y: 80 }, b: { x: 300, y: 200 } },
      { id: 1, a: { x: 80, y: 200 }, b: { x: 280, y: 80 } },
      { id: 2, a: { x: 160, y: 50 }, b: { x: 200, y: 240 } },
    ],
  },
  {
    label: '复杂场景',
    segs: [
      { id: 0, a: { x: 40, y: 120 }, b: { x: 200, y: 80 } },
      { id: 1, a: { x: 80, y: 200 }, b: { x: 320, y: 100 } },
      { id: 2, a: { x: 100, y: 60 }, b: { x: 160, y: 240 } },
      { id: 3, a: { x: 220, y: 200 }, b: { x: 360, y: 80 } },
    ],
  },
]

function cross(o: Pt, a: Pt, b: Pt) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

function onSegment(p: Pt, a: Pt, b: Pt) {
  return (
    Math.min(a.x, b.x) <= p.x && p.x <= Math.max(a.x, b.x) &&
    Math.min(a.y, b.y) <= p.y && p.y <= Math.max(a.y, b.y)
  )
}

function segmentsIntersect(s1: Seg, s2: Seg): Pt | null {
  const { a: p1, b: p2 } = s1, { a: p3, b: p4 } = s2
  const d1 = cross(p3, p4, p1), d2 = cross(p3, p4, p2)
  const d3 = cross(p1, p2, p3), d4 = cross(p1, p2, p4)
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    // Parametric intersection
    const denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    const t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom
    return { x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y) }
  }
  if (d1 === 0 && onSegment(p1, p3, p4)) return p1
  if (d2 === 0 && onSegment(p2, p3, p4)) return p2
  return null
}

type Event = { x: number; type: 'start' | 'end'; segId: number; label: string }

function buildEvents(segs: Seg[]): Event[] {
  const evs: Event[] = []
  for (const s of segs) {
    const left = s.a.x < s.b.x ? s.a : s.b
    const right = s.a.x < s.b.x ? s.b : s.a
    evs.push({ x: left.x, type: 'start', segId: s.id, label: `S${s.id}` })
    evs.push({ x: right.x, type: 'end', segId: s.id, label: `E${s.id}` })
  }
  return evs.sort((a, b) => a.x - b.x || (a.type === 'start' ? -1 : 1))
}

const SEG_COLORS = ['#6366f1', '#0ea5e9', '#f59e0b', '#10b981']

export function ShamosHoeyDemo() {
  const [presetIdx, setPresetIdx] = useState(1)
  const [activeEvent, setActiveEvent] = useState<number | null>(null)

  const { segs } = PRESETS[presetIdx]
  const events = buildEvents(segs)
  const curEventX = activeEvent !== null ? events[activeEvent]?.x : null

  // Check all pairs for intersections
  const intersections: { i: number; j: number; pt: Pt }[] = []
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const pt = segmentsIntersect(segs[i], segs[j])
      if (pt) intersections.push({ i, j, pt })
    }
  }
  const hasIntersection = intersections.length > 0

  // Active segments at sweep line position
  const activeSegs = curEventX !== null
    ? segs.filter(s => Math.min(s.a.x, s.b.x) <= curEventX && Math.max(s.a.x, s.b.x) >= curEventX)
    : []

  return (
    <div className="rounded-2xl border border-violet-200 dark:border-violet-700 overflow-hidden shadow-lg font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-700 via-purple-600 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-base">📐 Shamos-Hoey 算法：扫描线判交</h3>
        <p className="text-violet-100 text-xs mt-0.5">点击事件队列按步骤推进扫描线，检查当前活跃线段是否存在交叉</p>
        <div className="flex gap-1.5 mt-3 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setPresetIdx(i); setActiveEvent(null) }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
                presetIdx === i ? 'bg-white text-violet-700 font-bold' : 'bg-white/20 text-white hover:bg-white/30'
              }`}>{p.label}</button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG Canvas */}
          <div className="flex-shrink-0">
            <svg width={W} height={H}
              className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
              {/* Segments */}
              {segs.map(s => {
                const isActive = activeSegs.some(as => as.id === s.id)
                return (
                  <g key={s.id}>
                    <line x1={s.a.x} y1={s.a.y} x2={s.b.x} y2={s.b.y}
                      stroke={SEG_COLORS[s.id % SEG_COLORS.length]}
                      strokeWidth={isActive ? 4 : 2}
                      strokeLinecap="round"
                      opacity={isActive ? 1 : 0.4}
                    />
                    {/* Start/End dots */}
                    <circle cx={s.a.x} cy={s.a.y} r={5}
                      fill={SEG_COLORS[s.id % SEG_COLORS.length]} stroke="white" strokeWidth={1.5}/>
                    <circle cx={s.b.x} cy={s.b.y} r={5}
                      fill={SEG_COLORS[s.id % SEG_COLORS.length]} stroke="white" strokeWidth={1.5}/>
                    {/* Label */}
                    <text
                      x={(s.a.x + s.b.x) / 2}
                      y={(s.a.y + s.b.y) / 2 - 10}
                      textAnchor="middle"
                      fontSize={11} fontWeight="bold"
                      fill={SEG_COLORS[s.id % SEG_COLORS.length]}>
                      s{s.id}
                    </text>
                  </g>
                )
              })}

              {/* Intersection points */}
              {intersections.map((p, i) => (
                <g key={i}>
                  <circle cx={p.pt.x} cy={p.pt.y} r={9}
                    fill="#ef4444" stroke="white" strokeWidth={2} opacity={0.9}/>
                  <text x={p.pt.x} y={p.pt.y} textAnchor="middle" dominantBaseline="middle"
                    fontSize={8} fill="white" fontWeight="bold">✕</text>
                </g>
              ))}

              {/* Sweep line */}
              {curEventX !== null && (
                <line x1={curEventX} y1={0} x2={curEventX} y2={H}
                  stroke="#f59e0b" strokeWidth={2} strokeDasharray="6,3"/>
              )}
            </svg>
          </div>

          {/* Event queue + result */}
          <div className="flex flex-col gap-3 flex-1 min-w-[165px]">
            <div>
              <p className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">
                事件队列（点击推进扫描线）
              </p>
              <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden text-xs">
                <div className="grid grid-cols-3 bg-slate-100 dark:bg-slate-800 text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">
                  <div className="px-2 py-1.5">事件</div>
                  <div className="px-2 py-1.5 text-center">x 坐标</div>
                  <div className="px-2 py-1.5 text-center">类型</div>
                </div>
                {events.map((ev, i) => {
                  const isActive = activeEvent === i
                  const isPast = activeEvent !== null && i < activeEvent
                  return (
                    <button key={i} onClick={() => setActiveEvent(isActive ? null : i)}
                      className={`w-full grid grid-cols-3 border-t border-slate-100 dark:border-slate-700 text-left transition-colors ${
                        isActive ? 'bg-amber-50 dark:bg-amber-900/20' :
                        isPast ? 'bg-slate-50 dark:bg-slate-800/50 opacity-50' :
                        'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                      }`}>
                      <div className="px-2 py-1.5">
                        <span className="font-bold font-mono"
                          style={{ color: SEG_COLORS[ev.segId % SEG_COLORS.length] }}>
                          s{ev.segId}
                        </span>
                      </div>
                      <div className="px-2 py-1.5 text-center font-mono text-slate-500 dark:text-slate-400">
                        {Math.round(ev.x)}
                      </div>
                      <div className="px-2 py-1.5 text-center">
                        {ev.type === 'start'
                          ? <span className="text-emerald-600 dark:text-emerald-400 font-bold">插入</span>
                          : <span className="text-rose-500 dark:text-rose-400">删除</span>
                        }
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Active segments info */}
            {curEventX !== null && activeSegs.length > 0 && (
              <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs">
                <p className="font-bold text-slate-600 dark:text-slate-300 mb-1">活跃线段（BST 中）</p>
                <div className="flex gap-1.5 flex-wrap">
                  {activeSegs.map(s => (
                    <span key={s.id} className="px-2 py-0.5 rounded-full font-bold text-white"
                      style={{ background: SEG_COLORS[s.id % SEG_COLORS.length] }}>
                      s{s.id}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Result */}
            <div className={`rounded-xl border-2 text-center py-3 transition-all ${
              hasIntersection
                ? 'border-rose-400 bg-rose-50 dark:bg-rose-900/10'
                : 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/10'
            }`}>
              <p className="text-xl">{hasIntersection ? '❌' : '✅'}</p>
              <p className={`font-black text-sm mt-0.5 ${
                hasIntersection ? 'text-rose-700 dark:text-rose-300' : 'text-emerald-700 dark:text-emerald-300'
              }`}>
                {hasIntersection ? `存在 ${intersections.length} 个交点` : '无交叉！'}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">O((n+k) log n)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

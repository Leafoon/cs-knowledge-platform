'use client'

import { useState, useCallback, useMemo } from 'react'

const W = 420, H = 260, PAD = 20

interface Segment { id: number; y: number; x1: number; x2: number; color: string }
interface Event { x: number; type: 'start' | 'end'; seg: Segment }

const SEGS_PRESETS = [
  {
    label: '5 条线段',
    segments: [
      { id: 0, y: 55,  x1: 40,  x2: 200, color: '#6366f1' },
      { id: 1, y: 100, x1: 100, x2: 320, color: '#f97316' },
      { id: 2, y: 145, x1: 60,  x2: 180, color: '#10b981' },
      { id: 3, y: 195, x1: 150, x2: 360, color: '#e11d48' },
      { id: 4, y: 225, x1: 220, x2: 390, color: '#8b5cf6' },
    ] as Segment[],
  },
  {
    label: '3 条（含重叠）',
    segments: [
      { id: 0, y: 70,  x1: 40,  x2: 300, color: '#6366f1' },
      { id: 1, y: 130, x1: 80,  x2: 280, color: '#f97316' },
      { id: 2, y: 200, x1: 160, x2: 380, color: '#10b981' },
    ] as Segment[],
  },
]

function buildEvents(segs: Segment[]): Event[] {
  const evs: Event[] = []
  for (const s of segs) {
    evs.push({ x: s.x1, type: 'start', seg: s })
    evs.push({ x: s.x2, type: 'end',   seg: s })
  }
  return evs.sort((a, b) => a.x - b.x || (a.type === 'start' ? -1 : 1))
}

interface StepState {
  sweepX: number; active: number[]; event: Event; desc: string
}

function buildSteps(segs: Segment[]): StepState[] {
  const events = buildEvents(segs)
  const steps: StepState[] = []
  let active: number[] = []

  for (const ev of events) {
    if (ev.type === 'start') {
      active = [...active, ev.seg.id].sort((a, b) => {
        const sa = segs.find(s=>s.id===a)!, sb = segs.find(s=>s.id===b)!
        return sa.y - sb.y
      })
      steps.push({ sweepX: ev.x, active: [...active], event: ev,
        desc: `x=${ev.x}：线段 S${ev.seg.id} 开始 → 插入活跃集（按 y 排序）` })
    } else {
      active = active.filter(id => id !== ev.seg.id)
      steps.push({ sweepX: ev.x, active: [...active], event: ev,
        desc: `x=${ev.x}：线段 S${ev.seg.id} 结束 → 从活跃集移除` })
    }
  }
  steps.push({ sweepX: W - PAD, active: [], event: events[events.length-1],
    desc: '扫描完毕，所有端点事件处理完成' })
  return steps
}

export function SweepLineDemo() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [step, setStep] = useState(0)

  const preset = SEGS_PRESETS[presetIdx]
  const steps = useMemo(() => buildSteps(preset.segments), [presetIdx, preset.segments])
  const cur = steps[Math.min(step, steps.length - 1)]
  const activeSet = new Set(cur.active)

  const allEvents = useMemo(() => buildEvents(preset.segments), [presetIdx, preset.segments])

  const prev = useCallback(() => setStep(s => Math.max(0, s - 1)), [])
  const next = useCallback(() => setStep(s => Math.min(steps.length - 1, s + 1)), [steps.length])
  const reset = useCallback(() => { setStep(0) }, [])
  const changePreset = useCallback((i: number) => { setPresetIdx(i); setStep(0) }, [])

  return (
    <div className="rounded-2xl border border-violet-200 dark:border-violet-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-violet-600 to-fuchsia-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">📡 扫描线算法（Sweep Line）动画</h3>
        <p className="text-violet-100 text-xs mt-0.5">
          竖线从左至右扫描，处理"线段开始/结束"事件，维护 BST 活跃集
        </p>
        <div className="flex gap-2 mt-3 flex-wrap items-center">
          {SEGS_PRESETS.map((p, i) => (
            <button key={i} onClick={() => changePreset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${presetIdx===i?'bg-white text-violet-700 font-bold':'bg-white/20 text-white hover:bg-white/30'}`}>
              {p.label}
            </button>
          ))}
          <span className="text-xs text-white/70 ml-auto font-mono">{step+1}/{steps.length}</span>
        </div>
      </div>
      <div className="h-1.5 bg-slate-100 dark:bg-slate-800">
        <div className="h-full bg-violet-500 transition-all duration-300" style={{width:`${step/(steps.length-1)*100}%`}}/>
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG canvas */}
          <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Event markers on x-axis */}
            {allEvents.map((ev, i) => (
              <g key={i}>
                <line x1={ev.x} y1={H-PAD} x2={ev.x} y2={H-PAD+5}
                  stroke={ev.x === cur.sweepX ? '#a855f7' : '#cbd5e1'} strokeWidth={ev.x === cur.sweepX ? 2 : 1}/>
                <circle cx={ev.x} cy={H-PAD+10} r={4}
                  fill={ev.type==='start' ? '#10b981' : '#ef4444'}
                  stroke="white" strokeWidth={1}/>
              </g>
            ))}
            {/* X-axis */}
            <line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD} stroke="#94a3b8" strokeWidth={1}/>
            <text x={W-PAD} y={H-PAD+18} textAnchor="end" fontSize={9} fill="#94a3b8">x 轴（时间）</text>

            {/* Segments */}
            {preset.segments.map(seg => {
              const isActive = activeSet.has(seg.id)
              const isCurrent = cur.event.seg.id === seg.id
              return (
                <g key={seg.id}>
                  {/* Shadow for active */}
                  {isActive && <line x1={seg.x1} y1={seg.y} x2={seg.x2} y2={seg.y}
                    stroke={seg.color} strokeWidth={12} strokeLinecap="round" opacity={0.12}/>}
                  {/* Segment line */}
                  <line x1={seg.x1} y1={seg.y} x2={Math.min(seg.x2, cur.sweepX > seg.x1 ? seg.x2 : seg.x1)} y2={seg.y}
                    stroke={seg.color}
                    strokeWidth={isActive ? 4 : 2.5}
                    strokeLinecap="round"
                    opacity={isActive ? 1 : 0.35}/>
                  {/* Endpoints */}
                  <circle cx={seg.x1} cy={seg.y} r={5} fill={seg.color} stroke="white" strokeWidth={1.5}/>
                  <circle cx={seg.x2} cy={seg.y} r={5} fill={seg.color} stroke="white" strokeWidth={1.5} opacity={isActive||cur.sweepX>=seg.x2?1:0.4}/>
                  {/* Label */}
                  <text x={seg.x1 - 14} y={seg.y + 4} fontSize={10} fill={seg.color} fontWeight="bold">S{seg.id}</text>
                  {/* Active highlight */}
                  {isCurrent && <circle cx={cur.event.type==='start'?seg.x1:seg.x2} cy={seg.y} r={10} fill={seg.color} opacity={0.2}/>}
                </g>
              )
            })}

            {/* Sweep line */}
            <line x1={cur.sweepX} y1={PAD-5} x2={cur.sweepX} y2={H-PAD}
              stroke="#a855f7" strokeWidth={2.5} strokeDasharray="6,3"/>
            <text x={cur.sweepX+4} y={PAD+4} fontSize={10} fill="#a855f7" fontWeight="bold">x={cur.sweepX}</text>
            {/* Arrow head */}
            <polygon points={`${cur.sweepX},${PAD-5} ${cur.sweepX-5},${PAD+8} ${cur.sweepX+5},${PAD+8}`} fill="#a855f7"/>
          </svg>

          {/* Right panel */}
          <div className="flex-1 min-w-[160px] space-y-3 text-xs">
            {/* Event desc */}
            <div className={`rounded-xl p-3 border text-slate-700 dark:text-slate-300 leading-relaxed ${
              cur.event.type === 'start'
                ? 'bg-emerald-50 dark:bg-emerald-900/10 border-emerald-200 dark:border-emerald-800'
                : 'bg-rose-50 dark:bg-rose-900/10 border-rose-200 dark:border-rose-800'
            }`}>
              <span className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full mb-1.5 ${
                cur.event.type==='start' ? 'bg-emerald-200 text-emerald-700 dark:bg-emerald-800 dark:text-emerald-300'
                  : 'bg-rose-200 text-rose-700 dark:bg-rose-800 dark:text-rose-300'
              }`}>
                {cur.event.type === 'start' ? '▶ 开始事件' : '■ 结束事件'}
              </span>
              <p>{cur.desc}</p>
            </div>

            {/* Active BST */}
            <div>
              <p className="font-bold text-slate-600 dark:text-slate-300 mb-1.5">活跃集（BST，按 y 排序）</p>
              {cur.active.length === 0 ? (
                <p className="text-slate-400 italic text-[11px]">（空）</p>
              ) : (
                <div className="space-y-1">
                  {cur.active.map(id => {
                    const seg = preset.segments.find(s => s.id === id)!
                    return (
                      <div key={id} className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700">
                        <span className="w-3 h-3 rounded-full flex-shrink-0" style={{background: seg.color}}/>
                        <span className="font-mono font-bold text-slate-700 dark:text-slate-200">S{seg.id}</span>
                        <span className="text-slate-400 text-[10px]">y={seg.y}, [{seg.x1}–{seg.x2}]</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Event queue preview */}
            <div>
              <p className="font-bold text-slate-500 dark:text-slate-400 mb-1">剩余事件队列</p>
              <div className="space-y-0.5">
                {allEvents.filter(e => e.x > cur.sweepX).slice(0, 4).map((e, i) => (
                  <div key={i} className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                    <span className="w-2 h-2 rounded-full" style={{background: e.type==='start'?'#10b981':'#ef4444'}}/>
                    x={e.x} — S{e.seg.id} {e.type==='start'?'开始':'结束'}
                  </div>
                ))}
                {allEvents.filter(e => e.x > cur.sweepX).length > 4 && (
                  <p className="text-[10px] text-slate-400">+{allEvents.filter(e=>e.x>cur.sweepX).length-4} 个事件...</p>
                )}
              </div>
            </div>

            {/* Controls */}
            <div className="flex gap-2">
              <button onClick={reset} className="text-xs px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors">↺</button>
              <button onClick={prev} disabled={step===0} className="flex-1 text-xs py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors">← 上一步</button>
              <button onClick={next} disabled={step===steps.length-1} className="flex-1 text-xs py-1.5 rounded-lg bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-40 transition-colors">下一步 →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

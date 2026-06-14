'use client'

import { useState, useCallback } from 'react'

interface Pt { x: number; y: number; id: number }

const W = 360, H = 280, PAD = 28

const INITIAL_POINTS: Omit<Pt, 'id'>[] = [
  {x:40,y:200},{x:100,y:60},{x:200,y:30},{x:300,y:80},{x:310,y:200},
  {x:240,y:240},{x:150,y:250},{x:60,y:240},{x:160,y:130},{x:230,y:160},
  {x:90,y:130},{x:270,y:130},{x:180,y:190},
]

function cross(O: Pt, A: Pt, B: Pt): number {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
}
function polarAngle(pivot: Pt, p: Pt): number {
  return Math.atan2(p.y - pivot.y, p.x - pivot.x)
}
function dist2(a: Pt, b: Pt): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
}

function buildGrahamSteps(pts: Pt[]) {
  const steps: { stack: number[]; current: number | null; popped: number[]; phase: string; desc: string }[] = []
  // Step 0: find pivot (lowest y, then leftmost x)
  const pivot = pts.reduce((m, p) => (p.y > m.y || (p.y === m.y && p.x < m.x)) ? p : m, pts[0])
  steps.push({ stack: [pivot.id], current: null, popped: [], phase: 'pivot', desc: `找到最低点 P${pivot.id} (${pivot.x}, ${pivot.y}) 作为基准点` })

  // Sort by polar angle
  const sorted = pts
    .filter(p => p.id !== pivot.id)
    .sort((a, b) => {
      const da = polarAngle(pivot, a), db = polarAngle(pivot, b)
      if (Math.abs(da - db) > 1e-9) return da - db
      return dist2(pivot, a) - dist2(pivot, b)
    })
  sorted.unshift(pivot)

  steps.push({ stack: [pivot.id], current: null, popped: [], phase: 'sort', desc: `按极角排序 ${sorted.map(p => `P${p.id}`).join(' → ')}` })

  const stack: number[] = [sorted[0].id, sorted[1].id]
  steps.push({ stack: [...stack], current: sorted[1].id, popped: [], phase: 'scan', desc: `初始化栈: P${sorted[0].id}, P${sorted[1].id}` })

  for (let i = 2; i < sorted.length; i++) {
    const cur = sorted[i]
    const poppedThisStep: number[] = []
    while (stack.length >= 2) {
      const top2 = pts.find(p => p.id === stack[stack.length - 2])!
      const top1 = pts.find(p => p.id === stack[stack.length - 1])!
      const c = cross(top2, top1, cur)
      if (c <= 0) {
        poppedThisStep.push(stack.pop()!)
        steps.push({ stack: [...stack], current: cur.id, popped: [...poppedThisStep], phase: 'pop',
          desc: `叉积 ≤ 0 → 弹出 P${top1.id} (右转/共线)` })
      } else break
    }
    stack.push(cur.id)
    steps.push({ stack: [...stack], current: cur.id, popped: [...poppedThisStep], phase: 'push',
      desc: `左转 ✓ → 压栈 P${cur.id}，当前栈: [${stack.map(id => `P${id}`).join(',')}]` })
  }
  steps.push({ stack: [...stack], current: null, popped: [], phase: 'done', desc: `Graham Scan 完成！凸包有 ${stack.length} 个顶点` })
  return { steps, sorted }
}

const PTS: Pt[] = INITIAL_POINTS.map((p, i) => ({ ...p, id: i }))

export function GrahamScanAnimation() {
  const { steps } = buildGrahamSteps(PTS)
  const [step, setStep] = useState(0)
  const cur = steps[Math.min(step, steps.length - 1)]

  const stackSet = new Set(cur.stack)
  const poppedSet = new Set(cur.popped)

  const hullPoly = cur.stack.map(id => PTS.find(p => p.id === id)!)
  const pivot = PTS.reduce((m, p) => (p.y > m.y || (p.y === m.y && p.x < m.x)) ? p : m, PTS[0])

  const phaseColor = {
    pivot:  'from-amber-500 to-orange-400',
    sort:   'from-blue-600 to-cyan-500',
    scan:   'from-indigo-600 to-violet-500',
    pop:    'from-rose-600 to-pink-500',
    push:   'from-emerald-600 to-teal-500',
    done:   'from-green-600 to-emerald-500',
  }[cur.phase] ?? 'from-slate-600 to-slate-500'

  const prev = useCallback(() => setStep(s => Math.max(0, s - 1)), [])
  const next = useCallback(() => setStep(s => Math.min(steps.length - 1, s + 1)), [steps.length])
  const reset = useCallback(() => setStep(0), [])

  return (
    <div className="rounded-2xl border border-indigo-200 dark:border-indigo-700 overflow-hidden shadow-lg font-sans">
      <div className={`bg-gradient-to-r ${phaseColor} px-5 py-4 transition-all duration-500`}>
        <h3 className="text-white font-bold text-base">📐 Graham Scan 动态演示</h3>
        <p className="text-white/80 text-xs mt-0.5">逐步观察极角排序与栈操作，理解何时弹栈（右转）、何时压栈（左转）</p>
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          <span className="bg-white/20 text-white text-xs px-2.5 py-1 rounded-full font-mono">
            步骤 {step + 1} / {steps.length}
          </span>
          <span className="bg-white/20 text-white text-xs px-2.5 py-1 rounded-full font-bold">
            {cur.phase === 'pivot' ? '寻找基准点' : cur.phase === 'sort' ? '极角排序' :
             cur.phase === 'pop' ? '弹出（右转）' : cur.phase === 'push' ? '压栈（左转）' :
             cur.phase === 'done' ? '完成 🎉' : '初始化'}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-slate-100 dark:bg-slate-800">
        <div className="h-full bg-indigo-500 transition-all duration-300"
          style={{ width: `${(step / (steps.length - 1)) * 100}%` }} />
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          {/* SVG */}
          <svg width={W} height={H} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* Hull polygon */}
            {hullPoly.length >= 2 && (
              <polygon
                points={hullPoly.map(p => `${p.x},${p.y}`).join(' ')}
                fill={cur.phase === 'done' ? '#6366f120' : '#a5b4fc20'}
                stroke={cur.phase === 'done' ? '#10b981' : '#6366f1'}
                strokeWidth={cur.phase === 'done' ? 2.5 : 1.5}
                strokeDasharray={cur.phase === 'done' ? 'none' : '5,3'}
                strokeLinejoin="round"
              />
            )}

            {/* Stack edges */}
            {cur.stack.length >= 2 && cur.stack.slice(0, -1).map((id, i) => {
              const a = PTS.find(p => p.id === id)!, b = PTS.find(p => p.id === cur.stack[i + 1])!
              return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#6366f1" strokeWidth={2.5} strokeLinecap="round" />
            })}

            {/* All points */}
            {PTS.map(p => {
              let fill = '#94a3b8', r = 5
              if (p.id === pivot.id) { fill = '#f59e0b'; r = 8 }
              if (stackSet.has(p.id) && p.id !== pivot.id) { fill = '#6366f1'; r = 7 }
              if (poppedSet.has(p.id)) { fill = '#ef4444'; r = 7 }
              if (p.id === cur.current) { fill = '#10b981'; r = 9 }
              return (
                <g key={p.id}>
                  {p.id === cur.current && <circle cx={p.x} cy={p.y} r={14} fill="#10b98120" />}
                  <circle cx={p.x} cy={p.y} r={r} fill={fill} stroke="white" strokeWidth={1.5} />
                  <text x={p.x + 8} y={p.y - 7} fontSize={9} fill={fill} fontWeight="bold">P{p.id}</text>
                </g>
              )
            })}
          </svg>

          {/* Right panel */}
          <div className="flex-1 min-w-[160px] space-y-3">
            {/* Step description */}
            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {cur.desc}
            </div>

            {/* Stack visualization */}
            <div>
              <p className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-1.5">当前栈 (栈顶 →)</p>
              <div className="flex gap-1 flex-wrap">
                {cur.stack.length === 0
                  ? <span className="text-xs text-slate-400">（空）</span>
                  : cur.stack.map((id, i) => (
                    <span key={i} className={`text-xs px-2 py-1 rounded-lg font-mono font-bold ${
                      i === cur.stack.length - 1
                        ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 ring-2 ring-indigo-400'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                    }`}>P{id}</span>
                  ))}
              </div>
            </div>

            {/* Legend */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-1.5 text-[10px] font-bold text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800">图例</div>
              {[
                { c: '#f59e0b', label: '基准点 Pivot' },
                { c: '#10b981', label: '当前处理点' },
                { c: '#6366f1', label: '栈内（凸包候选）' },
                { c: '#ef4444', label: '已弹出（右转）' },
                { c: '#94a3b8', label: '未处理点' },
              ].map(({ c, label }) => (
                <div key={label} className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-600 dark:text-slate-400 border-t border-slate-100 dark:border-slate-700">
                  <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: c }} />
                  {label}
                </div>
              ))}
            </div>

            {/* Controls */}
            <div className="flex gap-2">
              <button onClick={reset} className="text-xs px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors">↺ 重置</button>
              <button onClick={prev} disabled={step === 0}
                className="flex-1 text-xs py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors">← 上一步</button>
              <button onClick={next} disabled={step === steps.length - 1}
                className="flex-1 text-xs py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 transition-colors">下一步 →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

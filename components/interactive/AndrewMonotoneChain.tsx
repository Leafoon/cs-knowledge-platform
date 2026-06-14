'use client'

import { useState, useCallback } from 'react'

interface Pt { x: number; y: number; id: number }

const RAW: Omit<Pt,'id'>[] = [
  {x:40,y:200},{x:80,y:80},{x:130,y:250},{x:180,y:40},{x:220,y:210},
  {x:260,y:80},{x:300,y:240},{x:330,y:130},{x:290,y:40},{x:100,y:180},
]
const PTS: Pt[] = RAW.map((p,i) => ({...p, id:i}))

function cross(O: Pt, A: Pt, B: Pt): number {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
}

interface StepInfo {
  phase: 'sort' | 'lower' | 'upper' | 'done'
  lower: number[]; upper: number[]; current: number | null; desc: string; popped: number[]
}

function buildSteps(): StepInfo[] {
  const steps: StepInfo[] = []
  const sorted = [...PTS].sort((a,b) => a.x - b.x || a.y - b.y)

  steps.push({ phase:'sort', lower:[], upper:[], current:null, popped:[],
    desc:`按 x 坐标排序: ${sorted.map(p=>`P${p.id}`).join(' → ')}` })

  // Lower hull
  const lower: number[] = []
  const poppedLower: number[] = []
  for (const p of sorted) {
    const popped: number[] = []
    while (lower.length >= 2) {
      const a = PTS.find(q=>q.id===lower[lower.length-2])!
      const b = PTS.find(q=>q.id===lower[lower.length-1])!
      if (cross(a,b,p) <= 0) { popped.push(lower.pop()!); poppedLower.push(popped[popped.length-1]) }
      else break
    }
    lower.push(p.id)
    steps.push({ phase:'lower', lower:[...lower], upper:[], current:p.id, popped:[...popped],
      desc: popped.length
        ? `下凸包：弹出 P${popped.join(',P')}（右转）→ 压入 P${p.id}`
        : `下凸包：左转 ✓ → 压入 P${p.id}，[${lower.map(id=>`P${id}`).join(',')}]`
    })
  }
  steps.push({ phase:'lower', lower:[...lower], upper:[], current:null, popped:[],
    desc:`下凸包完成: [${lower.map(id=>`P${id}`).join(' → ')}]` })

  // Upper hull
  const upper: number[] = []
  for (const p of [...sorted].reverse()) {
    const popped: number[] = []
    while (upper.length >= 2) {
      const a = PTS.find(q=>q.id===upper[upper.length-2])!
      const b = PTS.find(q=>q.id===upper[upper.length-1])!
      if (cross(a,b,p) <= 0) popped.push(upper.pop()!)
      else break
    }
    upper.push(p.id)
    steps.push({ phase:'upper', lower:[...lower], upper:[...upper], current:p.id, popped:[...popped],
      desc: popped.length
        ? `上凸包：弹出 P${popped.join(',P')} → 压入 P${p.id}`
        : `上凸包（从右→左）：压入 P${p.id}，[${upper.map(id=>`P${id}`).join(',')}]`
    })
  }

  const hull = [...new Set([...lower, ...upper])]
  steps.push({ phase:'done', lower:[...lower], upper:[...upper], current:null, popped:[],
    desc:`Andrew 单调链完成！凸包 = 下凸包 ∪ 上凸包，共 ${hull.length} 顶点` })
  return steps
}

const STEPS = buildSteps()

export function AndrewMonotoneChain() {
  const [step, setStep] = useState(0)
  const cur = STEPS[Math.min(step, STEPS.length-1)]
  const sorted = [...PTS].sort((a,b) => a.x-b.x || a.y-b.y)

  const lowerSet = new Set(cur.lower)
  const upperSet = new Set(cur.upper)

  const lPts = cur.lower.map(id => PTS.find(p=>p.id===id)!)
  const uPts = cur.upper.map(id => PTS.find(p=>p.id===id)!)

  const phaseGrad = {
    sort: 'from-slate-600 to-slate-500',
    lower: 'from-blue-600 to-cyan-500',
    upper: 'from-orange-500 to-amber-400',
    done: 'from-emerald-600 to-teal-500',
  }[cur.phase]

  const prev = useCallback(() => setStep(s => Math.max(0,s-1)), [])
  const next = useCallback(() => setStep(s => Math.min(STEPS.length-1,s+1)), [])

  return (
    <div className="rounded-2xl border border-sky-200 dark:border-sky-700 overflow-hidden shadow-lg font-sans">
      <div className={`bg-gradient-to-r ${phaseGrad} px-5 py-4 transition-all duration-500`}>
        <h3 className="text-white font-bold text-base">⛓️ Andrew 单调链算法可视化</h3>
        <p className="text-white/80 text-xs mt-0.5">分别构建下凸包（蓝）和上凸包（橙），合并得到完整凸包</p>
        <div className="flex gap-2 mt-2">
          <span className="bg-white/20 text-white text-xs px-2 py-1 rounded-full font-mono">步骤 {step+1}/{STEPS.length}</span>
          <span className={`text-xs px-2.5 py-1 rounded-full font-bold ${
            cur.phase==='lower' ? 'bg-blue-200 text-blue-800' :
            cur.phase==='upper' ? 'bg-orange-200 text-orange-800' :
            cur.phase==='done'  ? 'bg-emerald-200 text-emerald-800' : 'bg-white/20 text-white'
          }`}>
            {cur.phase==='sort'?'极角排序':cur.phase==='lower'?'构建下凸包':cur.phase==='upper'?'构建上凸包':'完成 🎉'}
          </span>
        </div>
      </div>
      <div className="h-1.5 bg-slate-100 dark:bg-slate-800">
        <div className="h-full bg-sky-500 transition-all duration-300" style={{width:`${step/(STEPS.length-1)*100}%`}} />
      </div>

      <div className="bg-white dark:bg-slate-900 p-4">
        <div className="flex gap-4 flex-wrap items-start">
          <svg width={370} height={285} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 flex-shrink-0">
            {/* x-sorted order line */}
            {sorted.map((p,i) => i < sorted.length-1 && (
              <line key={i} x1={p.x} y1={p.y} x2={sorted[i+1].x} y2={sorted[i+1].y}
                stroke="#cbd5e1" strokeWidth={1} strokeDasharray="4,3" />
            ))}
            {/* Lower hull */}
            {lPts.length >= 2 && lPts.map((p,i) => i < lPts.length-1 && (
              <line key={i} x1={p.x} y1={p.y} x2={lPts[i+1].x} y2={lPts[i+1].y}
                stroke="#3b82f6" strokeWidth={3} strokeLinecap="round" />
            ))}
            {/* Upper hull */}
            {uPts.length >= 2 && uPts.map((p,i) => i < uPts.length-1 && (
              <line key={i} x1={p.x} y1={p.y} x2={uPts[i+1].x} y2={uPts[i+1].y}
                stroke="#f97316" strokeWidth={3} strokeLinecap="round" />
            ))}
            {/* Fill when done */}
            {cur.phase === 'done' && (
              <polygon
                points={[...lPts, ...uPts.slice(1,-1)].map(p=>`${p.x},${p.y}`).join(' ')}
                fill="#10b98115" stroke="#10b981" strokeWidth={2.5} strokeLinejoin="round"
              />
            )}
            {/* Points */}
            {PTS.map(p => {
              const inL = lowerSet.has(p.id), inU = upperSet.has(p.id)
              const isCur = p.id === cur.current
              let fill = '#94a3b8', r = 5
              if (inL && inU) { fill = '#8b5cf6'; r = 7 }
              else if (inL) { fill = '#3b82f6'; r = 7 }
              else if (inU) { fill = '#f97316'; r = 7 }
              if (isCur) { fill = '#10b981'; r = 9 }
              return (
                <g key={p.id}>
                  {isCur && <circle cx={p.x} cy={p.y} r={15} fill="#10b98120"/>}
                  <circle cx={p.x} cy={p.y} r={r} fill={fill} stroke="white" strokeWidth={2}/>
                  <text x={p.x+8} y={p.y-7} fontSize={9} fill={fill} fontWeight="bold">P{p.id}</text>
                </g>
              )
            })}
          </svg>

          <div className="flex-1 min-w-[160px] space-y-3">
            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-xs leading-relaxed text-slate-700 dark:text-slate-300">
              {cur.desc}
            </div>

            {/* Stack states */}
            {(cur.phase === 'lower' || cur.phase === 'done') && cur.lower.length > 0 && (
              <div>
                <p className="text-xs font-bold text-blue-600 dark:text-blue-400 mb-1">下凸包栈</p>
                <div className="flex gap-1 flex-wrap">
                  {cur.lower.map((id,i) => (
                    <span key={i} className={`text-xs px-2 py-0.5 rounded-lg font-mono ${
                      i===cur.lower.length-1 ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-2 ring-blue-400'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300'
                    }`}>P{id}</span>
                  ))}
                </div>
              </div>
            )}
            {(cur.phase === 'upper' || cur.phase === 'done') && cur.upper.length > 0 && (
              <div>
                <p className="text-xs font-bold text-orange-600 dark:text-orange-400 mb-1">上凸包栈</p>
                <div className="flex gap-1 flex-wrap">
                  {cur.upper.map((id,i) => (
                    <span key={i} className={`text-xs px-2 py-0.5 rounded-lg font-mono ${
                      i===cur.upper.length-1 ? 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300 ring-2 ring-orange-400'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300'
                    }`}>P{id}</span>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-3 text-xs space-y-1 text-slate-600 dark:text-slate-400">
              {[{c:'#3b82f6',l:'下凸包顶点（蓝）'},{c:'#f97316',l:'上凸包顶点（橙）'},
                {c:'#8b5cf6',l:'两壳共有（紫）'},{c:'#10b981',l:'当前处理点（绿）'},{c:'#94a3b8',l:'未参与（灰）'}]
                .map(({c,l}) => (
                <div key={l} className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full flex-shrink-0" style={{background:c}}/>
                  {l}
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <button onClick={() => setStep(0)} className="text-xs px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors">↺</button>
              <button onClick={prev} disabled={step===0} className="flex-1 text-xs py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors">← 上一步</button>
              <button onClick={next} disabled={step===STEPS.length-1} className="flex-1 text-xs py-1.5 rounded-lg bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-40 transition-colors">下一步 →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

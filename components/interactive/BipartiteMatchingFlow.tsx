'use client'
import React, { useState } from 'react'

// ── 二分图匹配转最大流 ─────────────────────────────────────────
// 左部: 工人 W1-W4  右部: 任务 J1-J4
// 超源 s 连左部（容量1），右部连超汇 t（容量1）
// 中间边代表工人可完成的任务（容量1）

interface Preset {
  name: string
  desc: string
  workerEdges: [number, number][]   // [wi, ji] pairs (0-indexed)
  matching: [number, number][]       // matched pairs
}

const PRESETS: Preset[] = [
  {
    name: '场景一',
    desc: '4 工人 4 任务，最大匹配 = 3（不完美匹配）',
    workerEdges: [[0,0],[0,1],[1,1],[1,2],[2,2],[3,2],[3,3]],
    matching: [[0,0],[1,1],[2,2]],
  },
  {
    name: '场景二',
    desc: '完美匹配：4 工人 4 任务，最大匹配 = 4',
    workerEdges: [[0,0],[0,2],[1,1],[1,3],[2,2],[3,3]],
    matching: [[0,0],[1,1],[2,2],[3,3]],
  },
  {
    name: '场景三',
    desc: '稠密二分图：最大匹配 = 4（多种可能）',
    workerEdges: [[0,0],[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,3],[0,3],[2,2]],
    matching: [[0,1],[1,0],[2,3],[3,2]],
  },
]

const WORKERS = ['W₁','W₂','W₃','W₄']
const JOBS    = ['J₁','J₂','J₃','J₄']

// Layout (SVG 480x260):
// s: x=35, y=130
// workers: x=130, y = 50, 95, 155, 210
// jobs: x=320, y = 50, 95, 155, 210
// t: x=425, y=130

const SX = 35, TX = 430
const WX = 130, JX = 320
const WY = [50, 95, 155, 210]
const JY = [50, 95, 155, 210]
const SY = 130

type Mode = 'original' | 'flow'

export default function BipartiteMatchingFlow() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [mode, setMode] = useState<Mode>('original')
  const [hoverW, setHoverW] = useState<number | null>(null)

  const preset = PRESETS[presetIdx]
  const matchSet = new Set(preset.matching.map(([w,j]) => `${w}-${j}`))
  const matchedWorkers = new Set(preset.matching.map(([w]) => w))
  const matchedJobs    = new Set(preset.matching.map(([,j]) => j))

  function edgeColor(w: number, j: number) {
    const isMat = matchSet.has(`${w}-${j}`)
    if (mode === 'flow') return isMat ? '#10b981' : '#e2e8f0'
    if (isMat) return '#10b981'
    if (hoverW === w) return '#a78bfa'
    return '#94a3b8'
  }

  function edgeWidth(w: number, j: number) {
    const isMat = matchSet.has(`${w}-${j}`)
    return isMat ? 2.5 : 1.5
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 via-violet-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">二分图匹配 × 最大流</h3>
        <p className="text-purple-200 text-sm mt-0.5">最大匹配 = 超源到超汇的最大流 · 切换三种匹配场景</p>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 flex flex-wrap gap-2 items-center">
        {/* Presets */}
        <div className="flex gap-1.5 text-xs font-semibold">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setPresetIdx(i); setHoverW(null) }}
              className={`px-3 py-1.5 rounded-lg border transition-all ${presetIdx === i
                ? 'bg-violet-600 text-white border-violet-600'
                : 'border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:border-violet-400'}`}>
              {p.name}
            </button>
          ))}
        </div>

        {/* Mode toggle */}
        <div className="ml-auto flex rounded-lg overflow-hidden border border-slate-200 dark:border-slate-600 text-xs font-semibold">
          {(['original','flow'] as Mode[]).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1.5 transition-colors ${mode === m
                ? 'bg-violet-600 text-white'
                : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'}`}>
              {m === 'original' ? '二分图' : '流网络'}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-3">
        {/* Scenario desc */}
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700/40 px-3 py-2 text-[11px] text-violet-700 dark:text-violet-300">
          {preset.desc}
        </div>

        {/* SVG */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 overflow-hidden">
          <svg viewBox="0 0 465 260" className="w-full" style={{ maxHeight: 260 }}>
            <defs>
              <marker id="arr-bp-g" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#10b981" />
              </marker>
              <marker id="arr-bp-s" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#94a3b8" />
              </marker>
              <marker id="arr-bp-v" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#a78bfa" />
              </marker>
              <marker id="arr-bp-b" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#3b82f6" />
              </marker>
              <marker id="arr-bp-p" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill="#8b5cf6" />
              </marker>
            </defs>

            {/* Partition backgrounds */}
            <rect x={105} y={25} width={50} height={215} rx={8} fill="#3b82f6" opacity={0.05} />
            <rect x={305} y={25} width={50} height={215} rx={8} fill="#8b5cf6" opacity={0.05} />
            <text x={130} y={18} textAnchor="middle" fontSize={9} fill="#3b82f6" fontWeight="bold">工人</text>
            <text x={330} y={18} textAnchor="middle" fontSize={9} fill="#8b5cf6" fontWeight="bold">任务</text>

            {/* s→worker edges */}
            {WORKERS.map((_, w) => {
              const wy = WY[w]
              const dx = WX-18-SX, dy = wy-SY, len = Math.hypot(dx,dy)
              const ex = WX-18, ey = wy
              const sx2 = SX + dx/len*18
              const sy2 = SY + dy/len*18
              const isMat = matchedWorkers.has(w)
              const fColor = mode === 'flow' ? (isMat ? '#10b981' : '#e2e8f0') : '#3b82f6'
              return (
                <g key={w}>
                  <line x1={sx2} y1={sy2} x2={ex} y2={ey}
                    stroke={fColor} strokeWidth={isMat && mode === 'flow' ? 2.5 : 1.5}
                    markerEnd={isMat && mode === 'flow' ? 'url(#arr-bp-g)' : 'url(#arr-bp-b)'} />
                  {mode === 'flow' && (
                    <text x={(sx2+ex)/2-6} y={(sy2+ey)/2+4} fontSize={8} fill={fColor} textAnchor="middle">
                      {isMat ? '1/1' : '0/1'}
                    </text>
                  )}
                </g>
              )
            })}

            {/* job→t edges */}
            {JOBS.map((_, j) => {
              const jy = JY[j]
              const dx = TX-JX-18, dy = SY-jy, len = Math.hypot(dx+18, dy)
              const isMat = matchedJobs.has(j)
              const fColor = mode === 'flow' ? (isMat ? '#10b981' : '#e2e8f0') : '#8b5cf6'
              return (
                <g key={j}>
                  <line x1={JX+18} y1={jy} x2={TX-18} y2={SY}
                    stroke={fColor} strokeWidth={isMat && mode === 'flow' ? 2.5 : 1.5}
                    markerEnd={isMat && mode === 'flow' ? 'url(#arr-bp-g)' : 'url(#arr-bp-p)'} />
                  {mode === 'flow' && (
                    <text x={(JX+18+TX-18)/2+6} y={(jy+SY)/2+4} fontSize={8} fill={fColor} textAnchor="middle">
                      {isMat ? '1/1' : '0/1'}
                    </text>
                  )}
                </g>
              )
            })}

            {/* Worker-job edges */}
            {preset.workerEdges.map(([w,j], idx) => {
              const isMat = matchSet.has(`${w}-${j}`)
              const col = edgeColor(w, j)
              const markerColor = isMat ? 'g' : (hoverW === w ? 'v' : 's')
              return (
                <g key={idx}>
                  <line x1={WX+18} y1={WY[w]} x2={JX-18} y2={JY[j]}
                    stroke={col} strokeWidth={edgeWidth(w,j)}
                    markerEnd={`url(#arr-bp-${markerColor})`} />
                  {isMat && mode === 'flow' && (
                    <text x={(WX+18+JX-18)/2} y={(WY[w]+JY[j])/2-4} fontSize={8}
                      fill={col} textAnchor="middle">1/1</text>
                  )}
                </g>
              )
            })}

            {/* s node */}
            <circle cx={SX} cy={SY} r={16} fill="#3b82f6" />
            <text x={SX} y={SY+5} textAnchor="middle" fontSize={13} fontWeight="bold" fill="white">s</text>
            <text x={SX} y={SY+28} textAnchor="middle" fontSize={9} fill="#3b82f6">超源</text>

            {/* Worker nodes */}
            {WORKERS.map((w, i) => (
              <g key={i} className="cursor-pointer"
                onMouseEnter={() => setHoverW(i)} onMouseLeave={() => setHoverW(null)}>
                <circle cx={WX} cy={WY[i]} r={16}
                  fill={matchedWorkers.has(i) ? '#10b981' : '#3b82f6'}
                  stroke={hoverW === i ? '#a78bfa' : 'transparent'} strokeWidth={2} />
                <text x={WX} y={WY[i]+5} textAnchor="middle" fontSize={11} fontWeight="bold" fill="white">{w}</text>
              </g>
            ))}

            {/* Job nodes */}
            {JOBS.map((j, i) => (
              <g key={i}>
                <circle cx={JX} cy={JY[i]} r={16}
                  fill={matchedJobs.has(i) ? '#10b981' : '#8b5cf6'} />
                <text x={JX} y={JY[i]+5} textAnchor="middle" fontSize={11} fontWeight="bold" fill="white">{j}</text>
              </g>
            ))}

            {/* t node */}
            <circle cx={TX} cy={SY} r={16} fill="#8b5cf6" />
            <text x={TX} y={SY+5} textAnchor="middle" fontSize={13} fontWeight="bold" fill="white">t</text>
            <text x={TX} y={SY+28} textAnchor="middle" fontSize={9} fill="#8b5cf6">超汇</text>
          </svg>
        </div>

        {/* Matching details */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-emerald-200 dark:border-emerald-700/50 bg-emerald-50 dark:bg-emerald-900/20 p-3">
            <div className="text-[10px] font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wide mb-2">最大匹配 / 最大流</div>
            <div className="text-3xl font-black text-emerald-600 dark:text-emerald-400">{preset.matching.length}</div>
            <div className="text-[10px] text-emerald-600 dark:text-emerald-500 mt-1">= |f*| = max flow</div>
          </div>
          <div className="rounded-xl border border-violet-200 dark:border-violet-700/50 bg-violet-50 dark:bg-violet-900/20 p-3">
            <div className="text-[10px] font-bold text-violet-700 dark:text-violet-400 uppercase tracking-wide mb-2">匹配对</div>
            <div className="space-y-0.5">
              {preset.matching.map(([w,j], i) => (
                <div key={i} className="text-xs font-semibold text-emerald-700 dark:text-emerald-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block"/>
                  {WORKERS[w]} → {JOBS[j]}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Equivalence note */}
        <div className="rounded-lg bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700 px-3 py-2 text-[10px] text-slate-500 dark:text-slate-400">
          构造等价：s→每个工人(容量1)；工人→可选任务(容量1)；每个任务→t(容量1)。
          最大流 = 最大匹配大小（König's theorem）。匹配边对应饱和中间边。
        </div>
      </div>
    </div>
  )
}

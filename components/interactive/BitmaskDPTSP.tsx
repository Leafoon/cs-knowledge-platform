'use client'
import { useState, useEffect } from 'react'

/* ─── TSP 状压 DP ─────────────────────────────────────────── */
interface TspStep {
  S: number; i: number; j: number; newS: number
  newCost: number; oldCost: number; better: boolean; baseCost: number
}

const PRESETS = [
  {
    label: '4城市',
    pos: [[80, 55], [220, 55], [220, 155], [80, 155]] as [number, number][],
    dist: [[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]],
  },
  {
    label: '3城市',
    pos: [[150, 45], [245, 165], [55, 165]] as [number, number][],
    dist: [[0,12,18],[12,0,15],[18,15,0]],
  },
]

function computeTSP(n: number, dist: number[][]) {
  const INF = 1e9
  const dp: number[][] = Array.from({ length: 1 << n }, () => new Array(n).fill(INF))
  const par: number[][] = Array.from({ length: 1 << n }, () => new Array(n).fill(-1))
  dp[1][0] = 0
  const steps: TspStep[] = []
  for (let S = 1; S < (1 << n); S++) {
    for (let i = 0; i < n; i++) {
      if (!(S >> i & 1) || dp[S][i] === INF) continue
      for (let j = 0; j < n; j++) {
        if (S >> j & 1) continue
        const nS = S | (1 << j)
        const nc = dp[S][i] + dist[i][j]
        const better = nc < dp[nS][j]
        steps.push({ S, i, j, newS: nS, newCost: nc, oldCost: dp[nS][j], better, baseCost: dp[S][i] })
        if (better) { dp[nS][j] = nc; par[nS][j] = i }
      }
    }
  }
  return { dp, par, steps }
}

function buildTour(par: number[][], n: number, dist: number[][]): { path: number[]; cost: number } {
  const FULL = (1 << n) - 1
  let best = 1e9; let endCity = 1
  for (let i = 1; i < n; i++) {
    if (par[FULL][i] === -1) continue
    const t = 0 // placeholder to track dp externally
    endCity = i
  }
  // reconstruct from par
  const path: number[] = []
  let S = FULL; let cur = endCity
  path.push(cur)
  while (S !== 1) {
    const prev = par[S][cur]; if (prev === -1) break
    S = S & ~(1 << cur); cur = prev; path.push(cur)
  }
  path.reverse()
  // compute total via dist
  let totalCost = 0
  for (let i = 0; i < path.length - 1; i++) totalCost += dist[path[i]][path[i+1]]
  totalCost += dist[path[path.length-1]][0]
  path.push(0)
  return { path, cost: totalCost }
}

const CITY_COLORS = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626']

function toBin(S: number, n: number) { return S.toString(2).padStart(n, '0') }

export default function BitmaskDPTSP() {
  const [preIdx, setPreIdx] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(350)

  const { pos, dist } = PRESETS[preIdx]
  const n = dist.length
  const { dp, par, steps } = computeTSP(n, dist)
  const maxStep = steps.length - 1

  useEffect(() => { setStep(0); setPlaying(false) }, [preIdx])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, maxStep, speed])

  const cur = steps[Math.min(step, maxStep)]
  const isDone = step >= maxStep
  const { path: tourPath, cost: tourCost } = isDone ? buildTour(par, n, dist) : { path: [], cost: 0 }

  // build snapshot dp table at current step
  const snapDp: number[][] = Array.from({ length: 1 << n }, () => new Array(n).fill(1e9))
  snapDp[1][0] = 0
  for (let si = 0; si <= Math.min(step, maxStep); si++) {
    const st = steps[si]
    if (st.better && st.newCost < snapDp[st.newS][st.j]) snapDp[st.newS][st.j] = st.newCost
  }

  const SVG_W = 300, SVG_H = 220

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden shadow-sm">
      <div className="px-6 py-4 bg-gradient-to-r from-slate-700 to-zinc-700 dark:from-slate-800 dark:to-zinc-800">
        <h3 className="text-white font-bold text-base">旅行商问题（TSP）— 状压 DP</h3>
        <p className="text-slate-300 text-sm mt-0.5">
          dp[S][i] = 从城市 0 出发，已访问集合 S，当前在城市 i 的最短路程
        </p>
      </div>

      <div className="p-5 space-y-4">
        {/* 控制栏 */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-1">
            {PRESETS.map((pr, idx) => (
              <button key={idx} onClick={() => setPreIdx(idx)}
                className={`px-3 py-1 text-xs rounded-lg font-semibold transition-all ${preIdx === idx ? 'bg-slate-700 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300 hover:bg-slate-200 dark:hover:bg-zinc-700'}`}>
                {pr.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">←</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium ${playing ? 'bg-orange-500' : 'bg-slate-700 hover:bg-slate-600'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">→</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 rounded-lg text-slate-700 dark:text-zinc-200">↺</button>
            <span className="text-xs text-slate-400 dark:text-zinc-500">
              {(([['慢',600],['中',350],['快',150]] as [string,number][])).map(([l,ms]) => (
                <button key={ms} onClick={() => setSpeed(ms)}
                  className={`ml-1 px-2 py-1 rounded ${speed === ms ? 'bg-slate-700 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400'}`}>{l}</button>
              ))}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* 城市地图 SVG */}
          <div className="bg-slate-900 dark:bg-zinc-950 rounded-xl border border-slate-700 dark:border-zinc-700 p-3">
            <p className="text-xs font-bold text-slate-400 mb-2">城市地图</p>
            <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="block">
              {/* 所有边 */}
              {Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
                if (j <= i) return null
                const x1 = pos[i][0], y1 = pos[i][1], x2 = pos[j][0], y2 = pos[j][1]
                const isActive = cur && ((cur.i === i && cur.j === j) || (cur.i === j && cur.j === i))
                const inTour = isDone && tourPath.some((_, k) =>
                  k < tourPath.length - 1 &&
                  ((tourPath[k] === i && tourPath[k+1] === j) || (tourPath[k] === j && tourPath[k+1] === i))
                )
                return (
                  <g key={`e-${i}-${j}`}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke={inTour ? '#10b981' : isActive ? '#f59e0b' : '#374151'}
                      strokeWidth={inTour ? 3 : isActive ? 2 : 1.2}
                      strokeDasharray={inTour ? '' : '5,4'} />
                    <text x={(x1+x2)/2+2} y={(y1+y2)/2-4} textAnchor="middle" fontSize={9}
                      fill={isActive ? '#f59e0b' : '#6b7280'}>{dist[i][j]}</text>
                  </g>
                )
              }))}
              {/* 城市节点 */}
              {Array.from({ length: n }, (_, i) => {
                const [x, y] = pos[i]
                const isFrom = cur && cur.i === i
                const isTo = cur && cur.j === i
                return (
                  <g key={`c-${i}`}>
                    <circle cx={x} cy={y} r={isFrom || isTo ? 19 : 16}
                      fill={isFrom ? '#92400e' : isTo ? '#1d4ed8' : '#1e293b'}
                      stroke={CITY_COLORS[i]} strokeWidth={2.5} />
                    <text x={x} y={y+1} textAnchor="middle" dominantBaseline="middle"
                      fontSize={12} fontWeight="bold" fill="white">
                      {String.fromCharCode(65+i)}
                    </text>
                    {i === 0 && (
                      <text x={x} y={y+28} textAnchor="middle" fontSize={9} fill="#7c3aed">起点</text>
                    )}
                  </g>
                )
              })}
            </svg>
          </div>

          {/* 右侧：状态说明 + dp 表 */}
          <div className="space-y-3">
            {/* 当前转移说明 */}
            {cur && (
              <div className={`rounded-xl border p-3 text-xs space-y-1.5 ${cur.better ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800' : 'bg-slate-50 dark:bg-zinc-900 border-slate-200 dark:border-zinc-700'}`}>
                <p className="font-bold text-slate-600 dark:text-zinc-200">步骤 {step+1}/{maxStep+1}</p>
                {/* 集合 S 显示 */}
                <div className="flex items-center gap-2">
                  <span className="text-slate-500 dark:text-zinc-400">S =</span>
                  <div className="flex gap-0.5">
                    {Array.from({ length: n }, (_, b) => {
                      const isSet = (cur.S >> b) & 1
                      return (
                        <div key={b} className={`w-5 h-5 flex items-center justify-center rounded text-[10px] font-bold
                          ${isSet ? 'text-white' : 'bg-slate-200 dark:bg-zinc-700 text-slate-400 dark:text-zinc-500'}`}
                          style={isSet ? { background: CITY_COLORS[b] } : {}}>
                          {isSet ? String.fromCharCode(65+b) : '·'}
                        </div>
                      )
                    })}
                  </div>
                  <span className="font-mono text-slate-400 dark:text-zinc-500 text-[10px]">{toBin(cur.S,n)}</span>
                </div>
                <div className="font-mono text-slate-600 dark:text-zinc-300 space-y-0.5 text-[11px]">
                  <p>当前在 <span className="font-bold text-amber-600 dark:text-amber-400">{String.fromCharCode(65+cur.i)}</span> → 尝试去 <span className="font-bold text-blue-600 dark:text-blue-400">{String.fromCharCode(65+cur.j)}</span></p>
                  <p>dp[S][{cur.i}] + dist[{cur.i}][{cur.j}]</p>
                  <p className="pl-3">= {cur.baseCost} + {dist[cur.i][cur.j]} = {cur.newCost}</p>
                  <p className={`border-t border-slate-200 dark:border-zinc-700 pt-1 font-bold ${cur.better ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400 dark:text-zinc-500'}`}>
                    dp[{toBin(cur.newS,n)}][{cur.j}] = {cur.better ? cur.newCost : `${cur.oldCost >= 1e9 ? '∞' : cur.oldCost}（不更新）`} {cur.better ? '✓' : ''}
                  </p>
                </div>
              </div>
            )}

            {/* DP 表 */}
            <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 overflow-auto p-3">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-zinc-500 mb-1.5">dp[S][城市] 快照</p>
              <table className="text-[11px] w-full text-center">
                <thead>
                  <tr>
                    <th className="px-1 py-0.5 text-slate-400 dark:text-zinc-500">S (bin)</th>
                    {Array.from({ length: n }, (_, i) => (
                      <th key={i} className="px-2 py-0.5 font-bold" style={{ color: CITY_COLORS[i] }}>
                        {String.fromCharCode(65+i)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: 1 << n }, (_, S) => {
                    const hasAny = Array.from({ length: n }, (_, i) => snapDp[S][i]).some(v => v < 1e9)
                    if (!hasAny) return null
                    return (
                      <tr key={S} className={cur && (S === cur.S || S === cur.newS) ? 'bg-amber-50 dark:bg-amber-950/30' : ''}>
                        <td className="px-1 py-0.5 font-mono text-slate-500 dark:text-zinc-400 text-[10px]">{toBin(S,n)}</td>
                        {Array.from({ length: n }, (_, i) => {
                          const v = snapDp[S][i]
                          const hl = cur && S === cur.newS && i === cur.j && cur.better
                          return (
                            <td key={i} className={`px-2 py-0.5 font-mono ${hl ? 'text-emerald-600 dark:text-emerald-400 font-bold' : v >= 1e9 ? 'text-slate-300 dark:text-zinc-600' : 'text-slate-700 dark:text-zinc-200'}`}>
                              {v >= 1e9 ? '∞' : v}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* 最终答案 */}
            {isDone && (
              <div className="rounded-xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 p-3">
                <p className="text-xs font-bold text-emerald-700 dark:text-emerald-300 mb-1">最优哈密顿回路</p>
                <p className="font-mono text-sm text-emerald-800 dark:text-emerald-200">
                  {tourPath.map(c => String.fromCharCode(65+c)).join(' → ')}
                </p>
                <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">总路程 = {tourCost}</p>
              </div>
            )}
          </div>
        </div>

        {/* 进度条 */}
        <div className="w-full h-1.5 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-slate-600 dark:bg-slate-500 rounded-full transition-all duration-150"
            style={{ width: `${((step+1)/(maxStep+1))*100}%` }} />
        </div>
        <p className="text-xs text-slate-400 dark:text-zinc-500 text-right">
          {step+1}/{maxStep+1} · 状态空间: 2^{n} × {n} = {(1<<n)*n}
        </p>
      </div>
    </div>
  )
}

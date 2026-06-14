'use client'
import { useState, useEffect } from 'react'

/* ─── 数位 DP 可视化 ─────────────────────────────────────────
   题目：统计 [0, N] 中数字和 ≤ S 的整数个数
   以 2 位数为例，逐步填充 dp[pos][tight][sum] 表格，
   并展示答案的汇总过程。
   ─────────────────────────────────────────────────────────── */

interface Preset { label: string; N: number; S: number; answer: number }

const PRESETS: Preset[] = [
  { label: 'N=32, S=4', N: 32, S: 4, answer: 14 },
  { label: 'N=75, S=5', N: 75, S: 5, answer: 21 },
  { label: 'N=99, S=3', N: 99, S: 3, answer: 10 },
]

// 计算 2 位数时的 dp 表
function computeDP(N: number, S: number) {
  const D = [Math.floor(N / 10), N % 10] // 两个数字 [tens, ones]

  // dp[1][tight][sum] （个位）
  const dp1: number[][] = [new Array(S + 2).fill(0), new Array(S + 2).fill(0)] // [free][sum], [tight][sum]
  for (let t = 0; t <= 1; t++) {
    const ub = t ? D[1] : 9
    for (let s = 0; s <= S; s++) {
      let cnt = 0
      for (let d = 0; d <= ub; d++) if (s + d <= S) cnt++
      dp1[t][s] = cnt
    }
  }

  // dp[0][tight=1][0] （十位，从 0 到 D[0]）
  let answer = 0
  const contributions: { d: number; tight: boolean; sum: number; cnt: number }[] = []
  for (let d = 0; d <= D[0]; d++) {
    if (d > S) { contributions.push({ d, tight: d === D[0], sum: d, cnt: 0 }); continue }
    const nextTight = d === D[0] ? 1 : 0
    const cnt = dp1[nextTight][d] ?? 0
    contributions.push({ d, tight: d === D[0], sum: d, cnt })
    answer += cnt
  }

  return { D, dp1, contributions, answer }
}

// 每个 preset 的 step 数：2*(S+1) for pos=1, (D[0]+1) for pos=0 contributions
function totalSteps(N: number, S: number) {
  const D0 = Math.floor(N / 10)
  return 2 * (S + 1) + (D0 + 1)
}

export default function DigitDPViz() {
  const [preIdx, setPreIdx] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(450)

  const preset = PRESETS[preIdx]
  const { N, S } = preset
  const { D, dp1, contributions, answer } = computeDP(N, S)
  const maxStep = totalSteps(N, S)
  const phase1Len = S + 1
  const phase2Len = S + 1
  const phase1End = phase1Len
  const phase2End = phase1End + phase2Len
  // phase3: contributions of each tens digit d=0..D[0]
  const phase3Len = D[0] + 1

  useEffect(() => { setStep(0); setPlaying(false) }, [preIdx])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), speed)
    return () => clearTimeout(id)
  }, [playing, step, maxStep, speed])

  // ─── 当前各阶段进度 ────────────────────────────────────
  const p1Revealed = Math.min(step, phase1End) // dp1[free][0..p1Revealed-1]
  const p2Revealed = step > phase1End ? Math.min(step - phase1End, phase2Len) : 0 // dp1[tight]
  const p3Revealed = step > phase2End ? Math.min(step - phase2End, phase3Len) : 0 // contributions

  const isDone = step >= maxStep

  // 累计答案（只计 revealed contributions）
  const partialAnswer = contributions.slice(0, p3Revealed).reduce((acc, c) => acc + c.cnt, 0)

  const sumColors = ['#7c3aed','#2563eb','#059669','#d97706','#dc2626','#0891b2']

  return (
    <div className="rounded-2xl border border-fuchsia-200 dark:border-fuchsia-900 bg-white dark:bg-zinc-950 overflow-hidden shadow-sm">
      <div className="px-6 py-4 bg-gradient-to-r from-fuchsia-600 to-purple-600 dark:from-fuchsia-700 dark:to-purple-700">
        <h3 className="text-white font-bold text-base">数位 DP 可视化 — 逐步填充 dp[pos][tight][sum]</h3>
        <p className="text-fuchsia-100 text-sm mt-0.5">
          统计 [0, N] 中<strong className="text-white">数字和 ≤ S</strong> 的整数个数；以 2 位数拆解展示 DP 转移全过程
        </p>
      </div>

      <div className="p-5 space-y-4">
        {/* 控制栏 */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-1">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => setPreIdx(i)}
                className={`px-3 py-1 text-xs rounded-lg font-semibold transition-all ${preIdx === i ? 'bg-fuchsia-600 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300'}`}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">←</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium ${playing ? 'bg-orange-500' : 'bg-fuchsia-600 hover:bg-fuchsia-500'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">→</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-2.5 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 rounded-lg text-slate-700 dark:text-zinc-200">↺</button>
            {([['慢', 750], ['中', 450], ['快', 180]] as [string, number][]).map(([l, ms]) => (
              <button key={ms} onClick={() => setSpeed(ms)}
                className={`px-2 py-1 text-xs rounded ${speed === ms ? 'bg-fuchsia-600 text-white' : 'bg-slate-100 dark:bg-zinc-800 text-slate-500 dark:text-zinc-400'}`}>{l}</button>
            ))}
          </div>
        </div>

        {/* 题目信息栏 */}
        <div className="flex flex-wrap items-center gap-4 bg-fuchsia-50 dark:bg-fuchsia-950/30 rounded-xl border border-fuchsia-100 dark:border-fuchsia-900 px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-fuchsia-600 dark:text-fuchsia-400">上界 N =</span>
            {D.map((d, i) => (
              <span key={i} className="inline-flex items-center justify-center w-9 h-9 rounded-lg border-2 border-fuchsia-400 dark:border-fuchsia-600 bg-white dark:bg-zinc-900 text-lg font-bold text-fuchsia-700 dark:text-fuchsia-300 font-mono">
                {d}
              </span>
            ))}
            <span className="text-xs text-slate-400 dark:text-zinc-500">= {N}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-purple-600 dark:text-purple-400">数字和限制 S =</span>
            <span className="inline-flex items-center justify-center w-9 h-9 rounded-lg border-2 border-purple-400 dark:border-purple-600 bg-white dark:bg-zinc-900 text-lg font-bold text-purple-700 dark:text-purple-300 font-mono">
              {S}
            </span>
          </div>
          <div className="ml-auto text-xs text-slate-500 dark:text-zinc-400 font-mono">
            dp[pos][tight][sum] → 合法完成方案数
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* 左：pos=1（个位）DP 表 */}
          <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300">pos = 1（个位）</span>
              <span className="text-[10px] text-slate-400 dark:text-zinc-500">从右向左先填</span>
            </div>
            <table className="text-xs w-full text-center">
              <thead>
                <tr className="text-slate-400 dark:text-zinc-500">
                  <th className="py-1.5 px-2">当前数字和</th>
                  <th className="py-1.5 px-2 text-sky-500">free（无约束）</th>
                  <th className="py-1.5 px-2 text-amber-500">tight（≤ {D[1]}）</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: S + 1 }, (_, s) => {
                  const showFree = s < p1Revealed
                  const showTight = s < p2Revealed
                  const isCurFree = s === p1Revealed - 1 && step <= phase1End
                  const isCurTight = s === p2Revealed - 1 && step > phase1End && step <= phase2End
                  return (
                    <tr key={s} className={isCurFree || isCurTight ? 'bg-fuchsia-50 dark:bg-fuchsia-950/30' : ''}>
                      <td className="py-1 px-2 font-mono font-bold" style={{ color: sumColors[s % sumColors.length] }}>
                        sum = {s}
                      </td>
                      <td className={`py-1 px-3 font-mono rounded transition-all ${showFree ? 'text-sky-700 dark:text-sky-300 font-bold' : 'text-slate-200 dark:text-zinc-700'} ${isCurFree ? 'bg-sky-100 dark:bg-sky-900/40 ring-1 ring-sky-400' : ''}`}>
                        {showFree ? dp1[0][s] : '?'}
                      </td>
                      <td className={`py-1 px-3 font-mono rounded transition-all ${showTight ? 'text-amber-700 dark:text-amber-300 font-bold' : 'text-slate-200 dark:text-zinc-700'} ${isCurTight ? 'bg-amber-100 dark:bg-amber-900/40 ring-1 ring-amber-400' : ''}`}>
                        {showTight ? dp1[1][s] : '?'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="mt-2 text-[10px] text-slate-400 dark:text-zinc-500 space-y-0.5">
              <p className="text-sky-600 dark:text-sky-400">free: 个位可取 0..9，计满足 sum+d ≤ {S} 的 d 数</p>
              <p className="text-amber-600 dark:text-amber-400">tight: 个位最大为 {D[1]}（tight 约束），同样满足 sum 限制</p>
            </div>
          </div>

          {/* 右：pos=0（十位）答案汇总 */}
          <div className="rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-900 p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-fuchsia-100 dark:bg-fuchsia-950 text-fuchsia-700 dark:text-fuchsia-300">pos = 0（十位）</span>
              <span className="text-[10px] text-slate-400 dark:text-zinc-500">从 d=0 逐一汇总</span>
            </div>
            <div className="space-y-1.5">
              {contributions.map((c, idx) => {
                const revealed = idx < p3Revealed
                const isCur = idx === p3Revealed - 1
                return (
                  <div key={idx} className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs transition-all
                    ${!revealed ? 'opacity-30' : isCur ? 'bg-fuchsia-100 dark:bg-fuchsia-950/40 ring-1 ring-fuchsia-400 dark:ring-fuchsia-600' : 'bg-white dark:bg-zinc-800'}`}>
                    <span className="font-mono font-bold text-slate-500 dark:text-zinc-400 w-4">d={c.d}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${c.tight ? 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300' : 'bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300'}`}>
                      {c.tight ? 'tight' : 'free'}
                    </span>
                    <span className="text-slate-400 dark:text-zinc-500 flex-1 text-[11px]">
                      dp[1][{c.tight ? 1 : 0}][{c.sum}] = <span className={`font-bold ${revealed ? 'text-fuchsia-700 dark:text-fuchsia-300' : ''}`}>{revealed ? c.cnt : '?'}</span>
                    </span>
                    <span className={`font-mono text-[10px] ${c.d > S ? 'text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                      {revealed ? (c.d > S ? '(sum超限)' : `+${c.cnt}`) : ''}
                    </span>
                  </div>
                )
              })}
            </div>

            {/* 动态求和 */}
            <div className={`mt-3 px-3 py-2 rounded-xl border transition-all ${p3Revealed > 0 ? 'border-fuchsia-200 dark:border-fuchsia-800 bg-fuchsia-50 dark:bg-fuchsia-950/30' : 'border-slate-200 dark:border-zinc-700 bg-slate-100 dark:bg-zinc-800 opacity-40'}`}>
              <p className="text-xs font-bold text-slate-600 dark:text-zinc-200">
                {isDone ? '✓ 最终答案' : '累计'} = <span className="text-fuchsia-700 dark:text-fuchsia-300 text-base font-bold font-mono">{isDone ? answer : partialAnswer}</span>
              </p>
              {isDone && (
                <p className="text-[11px] text-slate-500 dark:text-zinc-400 mt-0.5">
                  [0, {N}] 中共 <strong className="text-fuchsia-700 dark:text-fuchsia-300">{answer}</strong> 个整数满足数字和 ≤ {S}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* 当前阶段说明 */}
        <div className="rounded-xl bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 px-4 py-2 text-xs text-slate-500 dark:text-zinc-400 flex items-center gap-4">
          {step === 0 && <span>点击播放，观察 dp 表如何从个位向十位反向填充，最终汇总出答案。</span>}
          {step > 0 && step <= phase1End && <span className="text-sky-600 dark:text-sky-400">第一阶段：填充 dp[1][<strong>free</strong>][{p1Revealed-1}] = {dp1[0][p1Revealed-1]}（个位无约束，sum = {p1Revealed-1}）</span>}
          {step > phase1End && step <= phase2End && <span className="text-amber-600 dark:text-amber-400">第二阶段：填充 dp[1][<strong>tight</strong>][{p2Revealed-1}] = {dp1[1][p2Revealed-1]}（个位 ≤ {D[1]}，sum = {p2Revealed-1}）</span>}
          {step > phase2End && step <= maxStep && !isDone && (() => {
            const cur = contributions[p3Revealed - 1]
            return <span className="text-fuchsia-600 dark:text-fuchsia-400">第三阶段：十位 d={cur?.d}（{cur?.tight ? 'tight' : 'free'}），贡献 {cur?.cnt} 个方案</span>
          })()}
          {isDone && <span className="text-emerald-600 dark:text-emerald-400 font-bold">✓ 完成！答案 = {answer}</span>}
        </div>

        {/* 进度条 */}
        <div className="w-full h-1.5 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
          <div className="h-full bg-fuchsia-500 rounded-full transition-all duration-150"
            style={{ width: `${(step / maxStep) * 100}%` }} />
        </div>
        <p className="text-xs text-slate-400 dark:text-zinc-500 text-right">
          步骤 {step}/{maxStep}
          {step <= phase1End ? ' · 阶段1：dp[1][free]' : step <= phase2End ? ' · 阶段2：dp[1][tight]' : ' · 阶段3：汇总答案'}
        </p>
      </div>
    </div>
  )
}

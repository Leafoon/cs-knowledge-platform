'use client'
import { useState, useEffect } from 'react'

/* ─── LIS 耐心排序步骤计算 ───────────────────────────────────────────────── */
interface LISStep {
  idx: number          // 当前处理的数组下标
  x: number            // 当前数字
  tailsBefore: number[] // 操作前的 tails
  tailsAfter: number[]  // 操作后的 tails
  pos: number          // 插入/替换位置（-1 表示追加）
  action: 'extend' | 'replace'
  dp: number[]         // O(n²) 的 dp 数组（同步展示）
}

function computeLISSteps(nums: number[]): LISStep[] {
  const steps: LISStep[] = []
  const tails: number[] = []
  const dp: number[] = new Array(nums.length).fill(1)

  for (let i = 0; i < nums.length; i++) {
    // O(n²) dp
    for (let j = 0; j < i; j++) {
      if (nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1)
    }

    // 耐心排序
    const before = [...tails]
    let lo = 0, hi = tails.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (tails[mid] < nums[i]) lo = mid + 1
      else hi = mid
    }
    const action: 'extend' | 'replace' = lo === tails.length ? 'extend' : 'replace'
    const pos = lo === tails.length ? -1 : lo
    if (lo === tails.length) tails.push(nums[i])
    else tails[lo] = nums[i]

    steps.push({ idx: i, x: nums[i], tailsBefore: before, tailsAfter: [...tails], pos: lo, action, dp: [...dp] })
  }
  return steps
}

const PRESETS = [
  { nums: [10, 9, 2, 5, 3, 7, 101, 18], label: 'LeetCode #300' },
  { nums: [0, 1, 0, 3, 2, 3], label: '长度为 4' },
  { nums: [3, 5, 6, 2, 5, 4, 19, 5, 6, 7, 12], label: '复杂序列' },
]

export default function LIS_PatienceSort() {
  const [numsStr, setNumsStr] = useState('10,9,2,5,3,7,101,18')
  const [stepIdx, setStepIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const nums = numsStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)).slice(0, 12)
  const steps = computeLISSteps(nums)
  const maxStep = steps.length - 1
  const done = stepIdx >= maxStep
  const curStep = steps[Math.min(stepIdx, maxStep)]

  useEffect(() => { setStepIdx(0); setPlaying(false) }, [numsStr])
  useEffect(() => {
    if (!playing) return
    if (stepIdx >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStepIdx(s => s + 1), 600)
    return () => clearTimeout(id)
  }, [playing, stepIdx, maxStep])

  const lisLen = curStep?.tailsAfter.length ?? 0
  const dpLen = done ? Math.max(...(curStep?.dp ?? [1])) : 0

  // 颜色：数组元素 — 已处理/当前/未处理
  const numStyle = (i: number) => {
    if (!curStep) return 'bg-slate-100 dark:bg-zinc-800 text-slate-400 dark:text-zinc-500'
    if (i === curStep.idx) return curStep.action === 'extend'
      ? 'bg-emerald-500 border-emerald-400 text-white ring-2 ring-emerald-300 scale-110'
      : 'bg-orange-500 border-orange-400 text-white ring-2 ring-orange-300 scale-110'
    if (i < curStep.idx) return 'bg-slate-200 dark:bg-zinc-700 text-slate-500 dark:text-zinc-400'
    return 'bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-200'
  }

  // tails 格子颜色
  const tailStyle = (i: number) => {
    if (!curStep) return 'bg-slate-100 dark:bg-zinc-800'
    if (i < curStep.tailsBefore.length && i === curStep.pos) {
      return curStep.action === 'replace'
        ? 'bg-orange-500 border-orange-400 text-white scale-110 shadow-lg ring-2 ring-orange-300'
        : 'bg-emerald-500 border-emerald-400 text-white scale-110 shadow-lg ring-2 ring-emerald-300'
    }
    if (i === curStep.pos && curStep.action === 'extend') {
      return 'bg-emerald-500 border-emerald-400 text-white scale-110 shadow-lg ring-2 ring-emerald-300'
    }
    if (i < curStep.tailsAfter.length) return 'bg-indigo-600 border-indigo-400 text-white'
    return 'bg-slate-100 dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-300 dark:text-zinc-600'
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      {/* 头部 */}
      <div className="px-6 py-4 bg-gradient-to-r from-emerald-600 to-teal-600">
        <h3 className="text-white font-bold text-base">LIS 耐心排序可视化</h3>
        <p className="text-emerald-200 text-sm mt-0.5">O(n log n) 算法 — 追踪 tails[] 数组的动态更新过程</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 预设 & 输入 */}
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {PRESETS.map(p => (
              <button key={p.label} onClick={() => setNumsStr(p.nums.join(','))}
                className="px-3 py-1 text-xs rounded-lg bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 dark:hover:bg-emerald-900 transition-colors">
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 dark:text-zinc-400 whitespace-nowrap">输入数组:</span>
            <input value={numsStr} onChange={e => setNumsStr(e.target.value)}
              className="flex-1 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-1.5 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-400"
              placeholder="逗号分隔，最多12个数" />
          </div>
        </div>

        {/* 控制 */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">← 上一步</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium transition-colors ${playing ? 'bg-orange-500 hover:bg-orange-400' : 'bg-emerald-600 hover:bg-emerald-500'}`}>
            {playing ? '⏸ 暂停' : '▶ 播放'}
          </button>
          <button onClick={() => setStepIdx(s => Math.min(maxStep, s + 1))} disabled={stepIdx >= maxStep}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">下一步 →</button>
          <button onClick={() => setStepIdx(maxStep)} className="px-3 py-1.5 text-xs bg-teal-600 hover:bg-teal-500 rounded-lg text-white">⚡ 完成</button>
          <button onClick={() => { setStepIdx(0); setPlaying(false) }} className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200">↺ 重置</button>
          <span className="text-xs text-slate-400 dark:text-zinc-500 ml-auto">步骤 {stepIdx + 1}/{maxStep + 1}</span>
        </div>

        {/* 当前处理元素说明 */}
        {curStep && (
          <div className={`px-4 py-2.5 rounded-xl text-sm font-mono border flex items-center gap-3 ${
            curStep.action === 'extend'
              ? 'bg-emerald-50 dark:bg-emerald-950 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
              : 'bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800 text-orange-800 dark:text-orange-200'
          }`}>
            <span className={`text-lg font-bold w-8 h-8 rounded-lg flex items-center justify-center text-white ${curStep.action === 'extend' ? 'bg-emerald-500' : 'bg-orange-500'}`}>
              {curStep.x}
            </span>
            {curStep.action === 'extend'
              ? <span>nums[{curStep.idx}] = <b>{curStep.x}</b> &gt; tails 所有元素 → <b>追加</b>到末尾，LIS 长度 +1 → {curStep.tailsAfter.length}</span>
              : <span>nums[{curStep.idx}] = <b>{curStep.x}</b> → 替换 tails[{curStep.pos}]（二分找到的第一个 ≥ {curStep.x} 的位置）</span>
            }
          </div>
        )}

        {/* 原始数组 */}
        <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4">
          <div className="text-xs font-semibold text-slate-500 dark:text-zinc-400 mb-3">原始数组 nums[]</div>
          <div className="flex flex-wrap gap-2">
            {nums.map((v, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className={`w-11 h-11 rounded-xl flex flex-col items-center justify-center border-2 font-mono font-bold text-sm transition-all duration-300 ${numStyle(i)}`}>
                  <span className="text-[9px] opacity-60 leading-none">i={i}</span>
                  <span>{v}</span>
                </div>
                {curStep && (
                  <div className={`text-[9px] font-mono font-bold ${curStep.dp[i] > 0 && i <= curStep.idx ? 'text-teal-600 dark:text-teal-400' : 'text-transparent'}`}>
                    dp={curStep.dp[i]}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* tails 数组 */}
        <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-semibold text-slate-500 dark:text-zinc-400">tails[] 数组（LIS 各长度的最小结尾）</div>
            <span className="text-xs px-2 py-1 bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 rounded font-mono">len(tails) = {lisLen} = 当前 LIS 长度</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {(curStep?.tailsAfter ?? []).map((v, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className={`w-11 h-11 rounded-xl flex flex-col items-center justify-center border-2 font-mono font-bold text-sm transition-all duration-300 ${tailStyle(i)}`}>
                  <span className="text-[9px] opacity-60 leading-none">k={i}</span>
                  <span>{v}</span>
                </div>
                <span className="text-[9px] text-slate-400 dark:text-zinc-500">长{i+1}→min</span>
              </div>
            ))}
            {(curStep?.tailsAfter.length ?? 0) === 0 && (
              <div className="text-xs text-slate-400 dark:text-zinc-500 italic py-3">（空，等待第一个元素）</div>
            )}
          </div>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-4 text-xs">
          {[
            { cls: 'bg-emerald-500 border-emerald-400 text-white', label: '追加（LIS 长度 +1）' },
            { cls: 'bg-orange-500 border-orange-400 text-white', label: '替换（贪心，保持结尾更小）' },
            { cls: 'bg-indigo-600 border-indigo-400 text-white', label: 'tails 中的值（各长度最小结尾）' },
          ].map(({ cls, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-5 h-5 rounded border-2 ${cls}`} />
              <span className="text-slate-600 dark:text-zinc-400">{label}</span>
            </div>
          ))}
        </div>

        {/* 结果 & 注意事项 */}
        {done && (
          <div className="space-y-3">
            <div className="bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-800 rounded-xl px-4 py-3 flex items-center gap-3">
              <span className="text-2xl">🏆</span>
              <div>
                <div className="text-sm font-bold text-emerald-800 dark:text-emerald-200">LIS 长度 = {lisLen}（O(n log n) 耐心排序结果）</div>
                <div className="text-sm font-bold text-emerald-800 dark:text-emerald-200">LIS 长度 = {dpLen}（O(n²) DP 结果验证✓）</div>
              </div>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-800 rounded-xl px-4 py-3 text-xs text-yellow-800 dark:text-yellow-200">
              ⚠️ <b>注意</b>：tails[] 本身并不是 LIS！它只保证 <b>长度正确</b>。例如处理完后 tails = {JSON.stringify(curStep?.tailsAfter)}，这未必是实际的 LIS 序列。若需重建 LIS，需额外维护 parent[] 数组记录来路。
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

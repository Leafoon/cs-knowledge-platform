'use client'

import { useState } from 'react'

function computePi(s: string): number[] {
  const m = s.length
  const pi = new Array<number>(m).fill(0)
  let k = 0
  for (let i = 1; i < m; i++) {
    while (k > 0 && s[k] !== s[i]) k = pi[k - 1]
    if (s[k] === s[i]) k++
    pi[i] = k
  }
  return pi
}

const PRESETS = [
  { s: 'ABABABAB', label: '重复 "AB"×4' },
  { s: 'ABCABCABC', label: '重复 "ABC"×3' },
  { s: 'AAAAAA', label: '重复 "A"×6' },
  { s: 'ABCABD', label: '无整数周期' },
  { s: 'AABAAAB', label: '非均匀周期' },
]

// 生成带颜色的周期块
function chunkByPeriod(s: string, period: number) {
  const chunks: string[] = []
  for (let i = 0; i < s.length; i += period) chunks.push(s.slice(i, i + period))
  return chunks
}

const CHUNK_COLORS = [
  'bg-cyan-100 dark:bg-cyan-900/50 border-cyan-300 dark:border-cyan-600 text-cyan-800 dark:text-cyan-200',
  'bg-violet-100 dark:bg-violet-900/50 border-violet-300 dark:border-violet-600 text-violet-800 dark:text-violet-200',
  'bg-amber-100 dark:bg-amber-900/50 border-amber-300 dark:border-amber-600 text-amber-800 dark:text-amber-200',
  'bg-rose-100 dark:bg-rose-900/50 border-rose-300 dark:border-rose-600 text-rose-800 dark:text-rose-200',
  'bg-emerald-100 dark:bg-emerald-900/50 border-emerald-300 dark:border-emerald-600 text-emerald-800 dark:text-emerald-200',
]

export default function KMPPeriodDetector() {
  const [input, setInput] = useState('ABABABAB')
  const [editVal, setEditVal] = useState('ABABABAB')

  const apply = (s: string) => {
    const clean = s.toUpperCase().replace(/[^A-Z]/g, '').slice(0, 18)
    if (clean.length < 2) return
    setInput(clean); setEditVal(clean)
  }

  const s = input
  const pi = computePi(s)
  const m = s.length
  const piLast = pi[m - 1]
  const period = m - piLast
  const isPeriodic = period < m && m % period === 0
  const repeatCount = isPeriodic ? m / period : 1
  const chunks = isPeriodic ? chunkByPeriod(s, period) : [s]

  return (
    <div className="my-8 rounded-2xl border border-cyan-200 dark:border-cyan-800 bg-white dark:bg-gray-900 shadow-lg overflow-hidden">
      {/* 标题 */}
      <div className="px-6 py-4 bg-gradient-to-r from-cyan-500 to-sky-600 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white text-lg">⟲</div>
        <div>
          <h3 className="text-white font-bold text-base">KMP 周期检测器</h3>
          <p className="text-cyan-100 text-xs mt-0.5">利用 π[m−1] 发现字符串的最小正周期</p>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* 预设 */}
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => (
            <button key={p.s}
              onClick={() => apply(p.s)}
              className={`px-3 py-1.5 rounded-lg text-xs border font-mono transition-all ${
                input === p.s
                  ? 'border-cyan-500 bg-cyan-100 dark:bg-cyan-900 text-cyan-700 dark:text-cyan-300'
                  : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-cyan-400'
              }`}>
              <span className="font-bold">{p.s}</span>
              <span className="text-gray-400 dark:text-gray-500 ml-1 font-sans">({p.label})</span>
            </button>
          ))}
        </div>

        {/* 输入 */}
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-mono uppercase focus:outline-none focus:ring-2 focus:ring-cyan-500"
            value={editVal}
            onChange={e => setEditVal(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && apply(editVal)}
            placeholder="输入字符串（A-Z，最多18字符）"
            maxLength={18}
          />
          <button onClick={() => apply(editVal)}
            className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium transition-colors">检测</button>
        </div>

        {/* 字符串 + π 数组展示 */}
        <div className="bg-gray-50 dark:bg-gray-800/60 rounded-xl p-5 space-y-4">
          {/* 字符格 */}
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">字符串 s（按最小周期着色）</div>
            <div className="flex flex-wrap gap-1.5">
              {[...s].map((ch, idx) => {
                const chunkIdx = isPeriodic ? Math.floor(idx / period) % CHUNK_COLORS.length : 0
                return (
                  <div key={idx} className="flex flex-col items-center gap-0.5">
                    <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm border-2 ${
                      isPeriodic ? CHUNK_COLORS[chunkIdx] : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 border-gray-300 dark:border-gray-600'
                    }`}>{ch}</div>
                    <div className="text-[9px] text-gray-400 font-mono">{idx}</div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* π 数组 */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
            <div className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">π 数组</div>
            <div className="flex flex-wrap gap-1.5">
              {pi.map((v, idx) => (
                <div key={idx} className="flex flex-col items-center gap-0.5">
                  <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm border ${
                    idx === m - 1
                      ? 'bg-sky-500 text-white border-sky-400 shadow-md shadow-sky-400/40 scale-110'
                      : 'bg-sky-50 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-700'
                  }`}>{v}</div>
                  <div className="text-[9px] text-gray-400 font-mono">{idx}</div>
                </div>
              ))}
            </div>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span className="font-mono text-sky-600 dark:text-sky-400">π[{m-1}] = {piLast}</span>
              {' '}（最后一个值，最关键！）
            </div>
          </div>
        </div>

        {/* 计算步骤 */}
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 text-xs font-medium text-gray-500 dark:text-gray-400">周期性推导</div>
          <div className="p-4 space-y-2 text-sm font-mono">
            <div className="text-gray-700 dark:text-gray-300">
              <span className="text-gray-400 dark:text-gray-500">m</span> = {m}，
              <span className="text-gray-400 dark:text-gray-500"> π[m−1]</span> = {piLast}
            </div>
            <div className="text-gray-700 dark:text-gray-300">
              候选最小周期 = m − π[m−1] = {m} − {piLast} = <span className="text-sky-600 dark:text-sky-400 font-bold">{period}</span>
            </div>
            <div className={`font-bold ${isPeriodic ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500'}`}>
              {isPeriodic
                ? `${m} % ${period} === 0  ✓  具有整数周期！`
                : `${m} % ${period} = ${m % period} ≠ 0  ✗  无整数周期`}
            </div>
          </div>
        </div>

        {/* 结论 */}
        <div className={`rounded-xl border p-5 ${
          isPeriodic
            ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700'
            : 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700'
        }`}>
          {isPeriodic ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl">🔁</span>
                <div>
                  <div className="font-bold text-emerald-700 dark:text-emerald-300 text-base">
                    最小正周期 = {period}，共重复 {repeatCount} 次
                  </div>
                  <div className="text-xs text-emerald-600/80 dark:text-emerald-400/80 mt-0.5">
                    周期子串：<span className="font-mono font-bold">"{s.slice(0, period)}"</span>
                  </div>
                </div>
              </div>
              {/* 可视化分块 */}
              <div className="flex flex-wrap gap-2 mt-1">
                {chunks.map((chunk, i) => (
                  <div key={i}
                    className={`px-3 py-2 rounded-lg border-2 font-mono font-bold text-sm ${CHUNK_COLORS[i % CHUNK_COLORS.length]}`}>
                    {chunk}
                    {chunk.length < period && <span className="text-xs opacity-60 ml-1">(不完整)</span>}
                  </div>
                ))}
              </div>
              <div className="text-xs text-emerald-600/70 dark:text-emerald-400/70">
                ↳ 表达式："{s.slice(0, period)}" × {repeatCount} = "{s}"
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-3">
              <span className="text-2xl mt-0.5">🔍</span>
              <div>
                <div className="font-bold text-rose-700 dark:text-rose-300 text-base">无法由任何子串整数倍重复构成</div>
                <div className="text-xs text-rose-600/80 dark:text-rose-400/80 mt-1">
                  候选周期 {period}，但 {m} 不能被 {period} 整除（余 {m % period}）。
                  该字符串不具有整数周期性，对应 LeetCode #459 应返回 False。
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

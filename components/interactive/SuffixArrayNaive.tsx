'use client'

import { useState, useMemo } from 'react'

const PRESETS = [
  { label: 'banana', s: 'banana' },
  { label: 'abab', s: 'abab' },
  { label: 'mississippi', s: 'mississippi' },
  { label: 'aabaa', s: 'aabaa' },
]

function buildSA(s: string): number[] {
  const n = s.length
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const sa = s.slice(a), sb = s.slice(b)
    return sa < sb ? -1 : sa > sb ? 1 : 0
  })
}

const RANK_COLORS = [
  'bg-blue-500', 'bg-indigo-500', 'bg-violet-500', 'bg-purple-500',
  'bg-pink-500', 'bg-rose-500', 'bg-orange-500', 'bg-amber-500',
  'bg-yellow-500', 'bg-lime-500', 'bg-emerald-500',
]

export default function SuffixArrayNaive() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [revealed, setRevealed] = useState(0)   // 0 = show unsorted, = sorted count revealed
  const [showSA, setShowSA] = useState(false)

  const s = PRESETS[presetIdx].s
  const n = s.length
  const sa = useMemo(() => buildSA(s), [s])

  // Unsorted: suffix i sorted by start index
  const unsorted = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])

  // Sorted list progressively revealed
  const sortedUp = sa.slice(0, revealed)

  function reset() { setRevealed(0); setShowSA(false) }

  return (
    <div className="rounded-2xl overflow-hidden border border-indigo-200 dark:border-indigo-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🗂️ 后缀数组——朴素构造可视化</h3>
          <p className="text-indigo-100 text-xs mt-0.5">所有后缀按字典序排序 → 起点下标序列即 SA</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i}
              onClick={() => { setPresetIdx(i); reset() }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${presetIdx === i ? 'bg-white text-indigo-700 font-semibold' : 'bg-indigo-500/40 text-white hover:bg-indigo-500/60'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-4">
        {/* String display */}
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">原串 s（下标从 0 开始）</p>
          <div className="flex flex-wrap gap-0">
            {s.split('').map((c, i) => (
              <div key={i} className="text-center">
                <div className="w-9 h-9 flex items-center justify-center bg-indigo-600 text-white font-mono font-bold text-sm rounded-sm border-r border-indigo-500 last:border-r-0">
                  {c}
                </div>
                <div className="text-[10px] text-gray-400 dark:text-gray-500 font-mono">{i}</div>
              </div>
            ))}
          </div>
        </div>

        {/* All suffixes table */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400">全部 {n} 个后缀（按起始下标）</p>
            <div className="flex gap-2">
              <button
                onClick={() => { setRevealed(v => Math.min(n, v + 1)); setShowSA(false) }}
                disabled={revealed >= n}
                className="px-3 py-1 text-xs bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-40 transition-all">
                排序下一个 →
              </button>
              <button
                onClick={() => { setRevealed(n); setShowSA(true) }}
                className="px-3 py-1 text-xs bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-lg hover:bg-indigo-200 dark:hover:bg-indigo-900/50 transition-all">
                全部完成 ⤵
              </button>
              <button onClick={reset}
                className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-lg">
                ↺ 重置
              </button>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-800">
                  <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium w-16">SA 排名</th>
                  <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium w-20">起点下标</th>
                  <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">后缀字符串</th>
                </tr>
              </thead>
              <tbody>
                {/* Sorted rows already revealed */}
                {sortedUp.map((pos, rank) => (
                  <tr key={`sorted-${pos}`} className="border-t border-gray-100 dark:border-gray-800 bg-indigo-50 dark:bg-indigo-900/20">
                    <td className="py-1.5 px-3 font-mono text-indigo-600 dark:text-indigo-400 font-bold">{rank}</td>
                    <td className="py-1.5 px-3 font-mono">
                      <span className={`inline-block w-6 h-6 text-[11px] text-white rounded-md leading-6 text-center font-bold ${RANK_COLORS[pos % RANK_COLORS.length]}`}>{pos}</span>
                    </td>
                    <td className="py-1.5 px-3 font-mono tracking-wide">
                      {s.slice(pos).split('').map((c, j) => (
                        <span key={j} className={j < (sa[rank - 1] !== undefined ? 0 : 0) ? 'text-indigo-500 font-bold' : 'text-gray-700 dark:text-gray-200'}>{c}</span>
                      ))}
                    </td>
                  </tr>
                ))}
                {/* Remaining unsorted rows */}
                {unsorted
                  .filter(i => !sa.slice(0, revealed).includes(i))
                  .sort((a, b) => a - b)
                  .map(pos => (
                    <tr key={`unsorted-${pos}`} className="border-t border-gray-100 dark:border-gray-800 opacity-50">
                      <td className="py-1.5 px-3 text-gray-400 font-mono">—</td>
                      <td className="py-1.5 px-3 font-mono">
                        <span className={`inline-block w-6 h-6 text-[11px] text-white rounded-md leading-6 text-center font-bold ${RANK_COLORS[pos % RANK_COLORS.length]}`}>{pos}</span>
                      </td>
                      <td className="py-1.5 px-3 font-mono text-gray-500 dark:text-gray-500 tracking-wide">{s.slice(pos)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* SA result */}
        {showSA && (
          <div className="p-3 rounded-xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-700">
            <p className="text-xs font-semibold text-indigo-700 dark:text-indigo-300 mb-2">
              SA = 后缀数组（排序后起点下标序列）
            </p>
            <div className="flex gap-1 flex-wrap">
              {sa.map((pos, rank) => (
                <div key={rank} className="text-center">
                  <div className={`w-9 h-9 flex items-center justify-center rounded-lg text-white font-mono font-bold text-sm ${RANK_COLORS[pos % RANK_COLORS.length]}`}>
                    {pos}
                  </div>
                  <div className="text-[10px] text-gray-400 mt-0.5">{rank}</div>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-indigo-600 dark:text-indigo-400 mt-2">
              SA[i] = {sa.map((v, i) => `${i}→${v}`).join('，')}
            </p>
          </div>
        )}

        <p className="text-xs text-gray-400 dark:text-gray-500">
          ⚠️ 朴素法复杂度 $O(n^2 \log n)$：排序 $O(n\log n)$ 次比较，每次最坏 $O(n)$——数据量大时需用倍增法优化
        </p>
      </div>
    </div>
  )
}

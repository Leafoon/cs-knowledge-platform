'use client'

import { useState, useMemo } from 'react'

const EXAMPLES = [
  { label: 'banana', s: 'banana', patterns: ['na', 'ban', 'an', 'xyz', 'a'] },
  { label: 'mississippi', s: 'mississippi', patterns: ['issi', 'miss', 'ippi', 'pp', 'abc'] },
  { label: 'abcabc', s: 'abcabc', patterns: ['abc', 'bc', 'cab', 'z'] },
]

function buildSA(s: string): number[] {
  return Array.from({ length: s.length }, (_, i) => i).sort((a, b) => {
    const sa = s.slice(a), sb = s.slice(b)
    return sa < sb ? -1 : sa > sb ? 1 : 0
  })
}

interface BinStep {
  lo: number; hi: number; mid: number
  comparison: string; direction: 'left' | 'right' | 'found' | 'miss'
  desc: string
}

function binarySearch(s: string, sa: number[], p: string): { steps: BinStep[]; found: boolean; positions: number[] } {
  const n = sa.length, m = p.length
  const steps: BinStep[] = []
  let lo = 0, hi = n - 1

  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const start = sa[mid]
    const suffix = s.slice(start, start + m)
    const cmp = suffix < p ? -1 : suffix > p ? 1 : 0
    let dir: BinStep['direction'] = 'miss'
    let desc = ''
    if (cmp === 0) {
      dir = 'found'
      desc = `SA[${mid}]=${sa[mid]}，后缀前 ${m} 字符="${suffix}" == "${p}" → 找到匹配!`
    } else if (cmp < 0) {
      dir = 'right'
      desc = `SA[${mid}]=${sa[mid]}，"${suffix}" < "${p}" → lo 移到 ${mid + 1}`
    } else {
      dir = 'left'
      desc = `SA[${mid}]=${sa[mid]}，"${suffix}" > "${p}" → hi 移到 ${mid - 1}`
    }
    steps.push({ lo, hi, mid, comparison: suffix, direction: dir, desc })
    if (cmp === 0) break
    if (cmp < 0) lo = mid + 1
    else hi = mid - 1
  }

  // Collect all matching positions
  const positions: number[] = []
  for (let i = 0; i < n; i++) {
    if (s.slice(sa[i], sa[i] + m) === p) positions.push(sa[i])
  }
  const found = positions.length > 0
  return { steps, found, positions }
}

const INDEX_COLORS = [
  'bg-sky-500', 'bg-blue-500', 'bg-indigo-500', 'bg-violet-500',
  'bg-purple-500', 'bg-fuchsia-500', 'bg-pink-500', 'bg-rose-500',
  'bg-red-500', 'bg-orange-500', 'bg-amber-500',
]

export default function SuffixSearchDemo() {
  const [exIdx, setExIdx] = useState(0)
  const [patIdx, setPatIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)
  const [customPat, setCustomPat] = useState('')

  const { s, patterns } = EXAMPLES[exIdx]
  const sa = useMemo(() => buildSA(s), [s])
  const pat = customPat || patterns[patIdx]

  const { steps, found, positions } = useMemo(() => binarySearch(s, sa, pat), [s, sa, pat])
  const step = steps[Math.min(stepIdx, steps.length - 1)]

  function resetSearch() { setStepIdx(0) }

  return (
    <div className="rounded-2xl overflow-hidden border border-sky-200 dark:border-sky-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-500 to-cyan-500 dark:from-sky-600 dark:to-cyan-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🔎 SA + 二分搜索子串</h3>
          <p className="text-sky-100 text-xs mt-0.5">在排好序的后缀数组上二分定位 → $O(m\log n)$</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => { setExIdx(i); setPatIdx(0); setCustomPat(''); resetSearch() }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-sky-700 font-semibold' : 'bg-sky-400/40 text-white hover:bg-sky-400/60'}`}>
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-3">
        {/* Pattern selection */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">模式串：</span>
          {EXAMPLES[exIdx].patterns.map((p, i) => (
            <button key={i}
              onClick={() => { setPatIdx(i); setCustomPat(''); resetSearch() }}
              className={`px-2.5 py-1 text-xs font-mono rounded-lg border transition-all ${!customPat && patIdx === i ? 'bg-sky-500 text-white border-sky-500' : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-sky-300'}`}>
              "{p}"
            </button>
          ))}
          <input value={customPat} onChange={e => { setCustomPat(e.target.value); resetSearch() }}
            placeholder="自定义..."
            className="px-2 py-1 text-xs font-mono rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 w-24 focus:outline-none focus:border-sky-400" />
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={resetSearch} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">↺ 重置</button>
          <button onClick={() => setStepIdx(v => Math.max(0, v - 1))} disabled={stepIdx === 0}
            className="px-2.5 py-1 text-xs bg-sky-100 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 rounded-lg disabled:opacity-40">← 上一步</button>
          <button onClick={() => setStepIdx(v => Math.min(steps.length - 1, v + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-3 py-1 text-xs bg-sky-500 text-white rounded-lg hover:bg-sky-600 disabled:opacity-40">下一步 →</button>
          <span className="text-xs text-gray-400">第 {Math.min(stepIdx + 1, steps.length)} / {steps.length} 步</span>
        </div>

        {/* Step desc */}
        {step && (
          <div className={`p-2.5 rounded-xl border text-xs leading-relaxed transition-all ${
            step.direction === 'found' ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-800 dark:text-emerald-200' :
            step.direction === 'miss' ? 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700 text-rose-800 dark:text-rose-200' :
            'bg-sky-50 dark:bg-sky-900/20 border-sky-200 dark:border-sky-700 text-sky-800 dark:text-sky-200'
          }`}>
            {step.desc}
          </div>
        )}

        {/* SA table with binary search highlighting */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs border-collapse w-full min-w-max">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <th className="py-2 px-3 text-center text-gray-500 font-medium w-12">rank</th>
                <th className="py-2 px-3 text-center text-gray-500 font-medium w-20">SA</th>
                <th className="py-2 px-3 text-left text-gray-500 font-medium">后缀</th>
                <th className="py-2 px-3 text-center text-gray-500 font-medium w-12">状态</th>
              </tr>
            </thead>
            <tbody>
              {sa.map((pos, r) => {
                const isMid = step && step.mid === r
                const inRange = step && r >= step.lo && r <= step.hi
                const isMatch = found && s.slice(pos, pos + pat.length) === pat
                return (
                  <tr key={r} className={`border-t border-gray-100 dark:border-gray-800 transition-colors ${
                    isMid ? (step.direction === 'found' ? 'bg-emerald-50 dark:bg-emerald-900/30' : 'bg-sky-100 dark:bg-sky-900/30') :
                    isMatch && stepIdx === steps.length - 1 ? 'bg-emerald-50/50 dark:bg-emerald-900/10' :
                    inRange ? 'bg-sky-50/50 dark:bg-sky-900/10' : 'opacity-40'
                  }`}>
                    <td className="py-1.5 px-3 text-center font-mono text-gray-400">{r}</td>
                    <td className="py-1.5 px-3 text-center">
                      <span className={`inline-block w-6 h-6 text-[11px] text-white rounded font-bold leading-6 text-center ${INDEX_COLORS[pos % INDEX_COLORS.length]}`}>{pos}</span>
                    </td>
                    <td className="py-1.5 px-3 font-mono tracking-wide">
                      {/* Highlight matching prefix */}
                      <span className="text-emerald-600 dark:text-emerald-400 font-bold">{s.slice(pos, pos + pat.length)}</span>
                      <span className="text-gray-400 dark:text-gray-500">{s.slice(pos + pat.length)}</span>
                    </td>
                    <td className="py-1.5 px-3 text-center text-base">
                      {isMid ? (step.direction === 'found' ? '✅' : step.direction === 'left' ? '⬆️' : '⬇️') : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Result */}
        {stepIdx === steps.length - 1 && (
          <div className={`p-3 rounded-xl border ${found ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700' : 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700'}`}>
            {found ? (
              <>
                <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-1">
                  ✅ 模式串 "{pat}" 在文本中出现 {positions.length} 次
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {positions.sort((a, b) => a - b).map(pos => (
                    <span key={pos} className="px-2 py-0.5 bg-emerald-500 text-white text-xs font-mono rounded">
                      起点 {pos}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-xs font-semibold text-rose-700 dark:text-rose-300">
                ❌ 模式串 "{pat}" 不在文本中
              </p>
            )}
            <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-1">
              共查询 {steps.length} 步，复杂度 O({pat.length} × log {s.length}) = O({pat.length}×{Math.ceil(Math.log2(s.length + 1))})
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

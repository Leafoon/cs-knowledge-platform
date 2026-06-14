'use client'

import { useState, useMemo } from 'react'

const PRESETS = [
  { label: 'banana', s: 'banana' },
  { label: 'aabaa', s: 'aabaa' },
  { label: 'abcabc', s: 'abcabc' },
  { label: 'mississippi', s: 'mississippi' },
]

function buildSA(s: string): number[] {
  const n = s.length
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const sa = s.slice(a), sb = s.slice(b)
    return sa < sb ? -1 : sa > sb ? 1 : 0
  })
}

interface KasaiStep {
  i: number
  r: number
  j: number
  k: number
  lcpVal: number
  desc: string
  lcp: number[]
}

function computeKasaiSteps(s: string, sa: number[]): KasaiStep[] {
  const n = s.length
  const rank = new Array(n).fill(0)
  for (let i = 0; i < n; i++) rank[sa[i]] = i
  const lcp = new Array(n).fill(0)
  const steps: KasaiStep[] = []
  let k = 0
  for (let i = 0; i < n; i++) {
    const r = rank[i]
    if (r === 0) {
      k = 0
      steps.push({
        i, r, j: -1, k: 0, lcpVal: 0,
        desc: `i=${i}：SA 名次为 0，无前驱，LCP[0]=0，k 重置为 0`,
        lcp: [...lcp],
      })
      continue
    }
    const j = sa[r - 1]
    const prevK = k
    while (i + k < n && j + k < n && s[i + k] === s[j + k]) k++
    lcp[r] = k
    steps.push({
      i, r, j, k, lcpVal: k,
      desc: `i=${i}（名次 ${r}）：与前驱后缀 j=${j}（名次 ${r - 1}）比较，k 从 ${prevK} 扩展到 ${k}，LCP[${r}]=${k}${k > 0 ? '，下轮 k 先减 1' : ''}`,
      lcp: [...lcp],
    })
    if (k > 0) k--
  }
  return steps
}

const INDEX_COLORS = [
  'bg-blue-500', 'bg-indigo-500', 'bg-violet-500', 'bg-purple-500',
  'bg-pink-500', 'bg-rose-500', 'bg-orange-500', 'bg-amber-500',
  'bg-yellow-500', 'bg-lime-500', 'bg-emerald-500',
]

export default function LCPArrayKasai() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const s = PRESETS[presetIdx].s
  const n = s.length
  const sa = useMemo(() => buildSA(s), [s])
  const rank = useMemo(() => { const r = new Array(n).fill(0); sa.forEach((pos, i) => r[pos] = i); return r }, [sa, n])
  const steps = useMemo(() => computeKasaiSteps(s, sa), [s, sa])

  const step = steps[stepIdx]

  function reset(pi: number) { setPresetIdx(pi); setStepIdx(0) }

  return (
    <div className="rounded-2xl overflow-hidden border border-amber-200 dark:border-amber-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 dark:from-amber-600 dark:to-orange-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🔍 Kasai 算法——LCP 数组线性构造</h3>
          <p className="text-amber-100 text-xs mt-0.5">按原串下标遍历，利用 k 单调性实现 O(n)</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => reset(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${presetIdx === i ? 'bg-white text-amber-700 font-semibold' : 'bg-amber-400/40 text-white hover:bg-amber-400/60'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-3">
        {/* Navigation */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => setStepIdx(0)} className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-500 rounded-lg">↺ 重置</button>
          <button onClick={() => setStepIdx(v => Math.max(0, v - 1))} disabled={stepIdx === 0}
            className="px-2.5 py-1 text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg disabled:opacity-40">← 上一步</button>
          <button onClick={() => setStepIdx(v => Math.min(steps.length - 1, v + 1))} disabled={stepIdx === steps.length - 1}
            className="px-3 py-1 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-40">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)} className="px-2.5 py-1 text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg">全部完成 ⤵</button>
          <span className="text-xs text-gray-400 dark:text-gray-500">{stepIdx + 1} / {steps.length} 步</span>
        </div>

        {/* Step description */}
        <div className="p-2.5 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700">
          <p className="text-xs leading-relaxed text-amber-800 dark:text-amber-200">{step.desc}</p>
          <div className="flex gap-4 mt-2 text-[11px] font-mono">
            <span className="text-gray-500">当前 k = <span className="text-amber-600 dark:text-amber-400 font-bold">{step.k}</span></span>
            {step.j >= 0 && (
              <>
                <span className="text-blue-500">i = {step.i}</span>
                <span className="text-violet-500">j = {step.j}</span>
              </>
            )}
          </div>
        </div>

        {/* SA and LCP table */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs border-collapse w-full min-w-max">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">rank</th>
                <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">SA[rank]</th>
                <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">后缀</th>
                <th className="py-2 px-3 text-center text-gray-500 dark:text-gray-400 font-medium">LCP[rank]</th>
                <th className="py-2 px-3 text-left text-gray-500 dark:text-gray-400 font-medium">公共前缀</th>
              </tr>
            </thead>
            <tbody>
              {sa.map((pos, r) => {
                const isI = step.i === pos
                const isJ = step.j === pos
                const lcpVal = step.lcp[r]
                const filled = lcpVal > 0 || (r === 0 && stepIdx > rank[step.i] - 1)
                const activeRow = isI || isJ
                const highlight = r <= stepIdx && step.lcp[r] !== undefined

                return (
                  <tr key={r} className={`border-t border-gray-100 dark:border-gray-800 transition-colors ${
                    isI ? 'bg-blue-50 dark:bg-blue-900/20' :
                    isJ ? 'bg-violet-50 dark:bg-violet-900/20' :
                    highlight && lcpVal > 0 ? 'bg-amber-50/50 dark:bg-amber-900/10' : ''
                  }`}>
                    <td className="py-1.5 px-3 text-center font-mono text-gray-400">{r}</td>
                    <td className="py-1.5 px-3 text-center">
                      <span className={`inline-block w-6 h-6 text-[11px] text-white rounded font-bold leading-6 text-center ${INDEX_COLORS[pos % INDEX_COLORS.length]} ${activeRow ? 'ring-2 ring-offset-1 ring-current' : ''}`}>
                        {pos}
                      </span>
                    </td>
                    <td className="py-1.5 px-3 font-mono tracking-wide">
                      {s.slice(pos).split('').map((c, ci) => (
                        <span key={ci} className={ci < lcpVal && filled ? 'text-amber-600 dark:text-amber-400 font-bold' : 'text-gray-600 dark:text-gray-300'}>{c}</span>
                      ))}
                    </td>
                    <td className="py-1.5 px-3 text-center">
                      {r === 0 ? (
                        <span className="font-mono text-gray-400">0</span>
                      ) : step.lcp[r] > 0 ? (
                        <span className={`inline-block px-2 py-0.5 rounded font-mono font-bold text-white ${isI ? 'bg-amber-500' : 'bg-orange-400'}`}>
                          {step.lcp[r]}
                        </span>
                      ) : (
                        <span className="text-gray-300 dark:text-gray-600 font-mono">—</span>
                      )}
                    </td>
                    <td className="py-1.5 px-3 font-mono text-amber-600 dark:text-amber-400 font-bold tracking-wide">
                      {filled && lcpVal > 0 ? s.slice(pos, pos + lcpVal) : ''}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Progress bar */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">进度</span>
          <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-amber-500 rounded-full transition-all duration-300"
              style={{ width: `${((stepIdx + 1) / steps.length) * 100}%` }} />
          </div>
          <span className="text-xs text-gray-400">{Math.round(((stepIdx + 1) / steps.length) * 100)}%</span>
        </div>

        {/* Summary */}
        {stepIdx === steps.length - 1 && (
          <div className="p-3 rounded-xl bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-700">
            <p className="text-xs font-semibold text-orange-700 dark:text-orange-300 mb-1.5">✅ LCP 构建完成</p>
            <div className="flex gap-1 flex-wrap">
              {step.lcp.map((v, i) => (
                <div key={i} className="text-center">
                  <div className={`w-9 h-9 flex items-center justify-center rounded-lg font-mono font-bold text-sm ${v > 0 ? 'bg-amber-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-500'}`}>{v}</div>
                  <div className="text-[10px] text-gray-400 mt-0.5">LCP[{i}]</div>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-orange-600 dark:text-orange-400 mt-2">
              最长公共前缀 = max(LCP) = <span className="font-bold">{Math.max(...step.lcp)}</span>，即最长重复子串长度
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

'use client'

import { useState, useMemo } from 'react'

// ---------------------------------------------------------------------------
// Polynomial rolling hash — use BigInt to avoid float precision issues,
// but we keep values small enough to display nicely
// ---------------------------------------------------------------------------
const BASE1 = 131n, MOD1 = 1_000_000_007n
const BASE2 = 137n, MOD2 = 998_244_353n

function buildHash(s: string): {
  h1: bigint[]; h2: bigint[]; pw1: bigint[]; pw2: bigint[]
} {
  const n = s.length
  const h1 = new Array<bigint>(n + 1).fill(0n)
  const h2 = new Array<bigint>(n + 1).fill(0n)
  const pw1 = new Array<bigint>(n + 1).fill(1n)
  const pw2 = new Array<bigint>(n + 1).fill(1n)
  for (let i = 1; i <= n; i++) {
    const c = BigInt(s.charCodeAt(i - 1))
    h1[i] = (h1[i - 1] * BASE1 + c) % MOD1
    h2[i] = (h2[i - 1] * BASE2 + c) % MOD2
    pw1[i] = pw1[i - 1] * BASE1 % MOD1
    pw2[i] = pw2[i - 1] * BASE2 % MOD2
  }
  return { h1, h2, pw1, pw2 }
}

function queryH(h: bigint[], pw: bigint[], mod: bigint, l: number, r: number): bigint {
  // l, r are 1-indexed
  return ((h[r] - h[l - 1] * pw[r - l + 1] % mod) % mod + mod) % mod
}

const EXAMPLES = [
  { label: 'banana', s: 'banana' },
  { label: 'abcabc', s: 'abcabc' },
  { label: 'aabbaab', s: 'aabbaab' },
  { label: 'mississi', s: 'mississi' },
]

function fmt(n: bigint): string {
  return n.toLocaleString()
}

export default function StringHashingDemo() {
  const [exIdx, setExIdx] = useState(0)
  const [selL, setSelL] = useState<number | null>(null)   // 1-indexed
  const [selR, setSelR] = useState<number | null>(null)
  const [dualHash, setDualHash] = useState(false)
  const [selStep, setSelStep] = useState<0 | 1>(0) // 0=pick L, 1=pick R

  const { s } = EXAMPLES[exIdx]
  const n = s.length
  const { h1, h2, pw1, pw2 } = useMemo(() => buildHash(s), [s])

  function handleChar(i: number) { // 1-indexed
    if (selStep === 0) { setSelL(i); setSelR(null); setSelStep(1) }
    else { if (i >= (selL ?? 1)) { setSelR(i); setSelStep(0) } else { setSelL(i); setSelR(null) } }
  }

  const queryVal1 = selL && selR ? queryH(h1, pw1, MOD1, selL, selR) : null
  const queryVal2 = selL && selR ? queryH(h2, pw2, MOD2, selL, selR) : null
  const queryLen  = selL && selR ? selR - selL + 1 : 0

  function resetEx(i: number) { setExIdx(i); setSelL(null); setSelR(null); setSelStep(0) }

  return (
    <div className="rounded-2xl overflow-hidden border border-teal-200 dark:border-teal-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-500 to-emerald-500 dark:from-teal-600 dark:to-emerald-600 px-5 py-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-white font-bold text-base">🔢 字符串哈希 — O(1) 区间查询</h3>
          <p className="text-teal-100 text-xs mt-0.5">点击两个字符位置，即时查看 hash(l, r) 计算过程</p>
        </div>
        <div className="flex gap-1 flex-wrap">
          {EXAMPLES.map((e, i) => (
            <button key={i} onClick={() => resetEx(i)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all ${exIdx === i ? 'bg-white text-teal-700 font-semibold' : 'bg-teal-400/40 text-white hover:bg-teal-400/60'}`}>
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4 space-y-4">
        {/* Dual hash toggle */}
        <div className="flex items-center gap-3">
          <button onClick={() => setDualHash(v => !v)}
            className={`px-3 py-1.5 text-xs rounded-xl border transition-all font-medium ${dualHash ? 'bg-emerald-500 text-white border-emerald-500' : 'border-gray-200 dark:border-gray-700 text-gray-500 hover:border-emerald-300'}`}>
            {dualHash ? '✅ 双模哈希 (减少碰撞)' : '单模哈希'}
          </button>
          <span className="text-[11px] text-gray-400 dark:text-gray-500">
            {dualHash ? `使用两对 (BASE, MOD)，碰撞概率 ≈ 1/(10⁹ × 10⁹)` : `BASE=${BASE1}, MOD=${MOD1.toLocaleString()}`}
          </span>
        </div>

        {/* Click instruction banner */}
        <div className={`px-3 py-2 rounded-xl text-xs border ${
          selStep === 0 ? 'bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-700 text-teal-700 dark:text-teal-300' :
                          'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300'
        }`}>
          {selStep === 0 ? '👆 点击一个字符作为查询区间的 左端点 L' : `✅ L=${selL}（已选），再点击右侧字符作为 右端点 R`}
        </div>

        {/* String characters — clickable */}
        <div className="space-y-1">
          {/* Chars */}
          <div className="flex gap-0.5 flex-wrap">
            {s.split('').map((c, idx0) => {
              const i = idx0 + 1 // 1-indexed
              const inRange = selL && selR && i >= selL && i <= selR
              const isL = i === selL
              const isR = i === selR
              return (
                <button key={i} onClick={() => handleChar(i)}
                  className={`w-9 h-9 text-sm font-mono font-bold rounded-lg border-2 transition-all ${
                    isL ? 'bg-teal-500 text-white border-teal-500 scale-110 shadow' :
                    isR ? 'bg-emerald-500 text-white border-emerald-500 scale-110 shadow' :
                    inRange ? 'bg-teal-100 dark:bg-teal-900/40 text-teal-800 dark:text-teal-200 border-teal-200 dark:border-teal-700' :
                    'bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:border-teal-300'
                  }`}>
                  {c}
                </button>
              )
            })}
          </div>
          {/* Index */}
          <div className="flex gap-0.5 flex-wrap">
            {s.split('').map((_, idx0) => (
              <div key={idx0} className="w-9 text-[10px] text-center text-gray-400">{idx0 + 1}</div>
            ))}
          </div>
        </div>

        {/* Prefix hash table */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
          <table className="text-xs border-collapse w-full min-w-max">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-800 text-gray-500">
                <th className="py-2 px-3 text-left font-medium">i</th>
                <th className="py-2 px-3 text-left font-medium">s[i]</th>
                <th className="py-2 px-3 text-left font-medium">h₁[i] = (h₁[i-1]×{BASE1.toString()}+ord) mod {MOD1.toLocaleString()}</th>
                {dualHash && <th className="py-2 px-3 text-left font-medium">h₂[i] = (h₂[i-1]×{BASE2.toString()}+ord) mod {MOD2.toLocaleString()}</th>}
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-gray-100 dark:border-gray-800">
                <td className="py-1.5 px-3 font-mono text-gray-400">0</td>
                <td className="py-1.5 px-3 font-mono text-gray-400">—</td>
                <td className="py-1.5 px-3 font-mono text-gray-400">0</td>
                {dualHash && <td className="py-1.5 px-3 font-mono text-gray-400">0</td>}
              </tr>
              {s.split('').map((c, idx0) => {
                const i = idx0 + 1
                const inRange = selL && selR && i >= selL && i <= selR
                return (
                  <tr key={i} className={`border-t border-gray-100 dark:border-gray-800 transition-colors ${
                    inRange ? 'bg-teal-50 dark:bg-teal-900/20' : ''
                  }`}>
                    <td className="py-1.5 px-3 font-mono font-bold text-teal-600 dark:text-teal-400">{i}</td>
                    <td className="py-1.5 px-3 font-mono font-bold">{c}</td>
                    <td className="py-1.5 px-3 font-mono text-[11px] text-gray-600 dark:text-gray-400">{fmt(h1[i])}</td>
                    {dualHash && <td className="py-1.5 px-3 font-mono text-[11px] text-gray-600 dark:text-gray-400">{fmt(h2[i])}</td>}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Query result */}
        {queryVal1 !== null && selL && selR && (
          <div className="p-4 rounded-2xl bg-gradient-to-br from-teal-50 to-emerald-50 dark:from-teal-900/20 dark:to-emerald-900/20 border border-teal-200 dark:border-teal-700">
            <p className="text-xs font-bold text-teal-700 dark:text-teal-300 mb-2">
              查询 hash(L={selL}, R={selR})，子串 = "{s.slice(selL - 1, selR)}"（长度 {queryLen}）
            </p>
            {/* Formula display */}
            <div className="font-mono text-[11px] text-gray-600 dark:text-gray-300 space-y-1 bg-white/60 dark:bg-black/20 rounded-xl p-3">
              <div>hash₁(L,R) = (h₁[R] − h₁[L−1] × BASE₁^(R−L+1)) mod MOD₁</div>
              <div className="text-teal-700 dark:text-teal-300">
                = ({fmt(h1[selR])} − {fmt(h1[selL - 1])} × {fmt(pw1[queryLen])}) mod {MOD1.toLocaleString()}
              </div>
              <div className="font-bold text-emerald-600 dark:text-emerald-400">= {fmt(queryVal1)}</div>
              {dualHash && (
                <>
                  <div className="mt-2">hash₂(L,R) = ({fmt(h2[selR])} − {fmt(h2[selL - 1])} × {fmt(pw2[queryLen])}) mod {MOD2.toLocaleString()}</div>
                  <div className="font-bold text-emerald-600 dark:text-emerald-400">= {fmt(queryVal2!)}</div>
                  <div className="mt-1 text-gray-400 text-[10px]">联合哈希 = ({fmt(queryVal1)}, {fmt(queryVal2!)})</div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Info boxes */}
        <div className="grid sm:grid-cols-2 gap-2 text-[11px]">
          <div className="p-3 rounded-xl bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-700">
            <p className="font-semibold text-teal-700 dark:text-teal-300 mb-1">⚡ 复杂度</p>
            <p className="text-gray-500 dark:text-gray-400">预处理 O(n)，单次查询 O(1)，适合大量子串比较</p>
          </div>
          <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-700">
            <p className="font-semibold text-rose-700 dark:text-rose-300 mb-1">⚠️ 注意事项</p>
            <p className="text-gray-500 dark:text-gray-400">单模存在哈希碰撞风险；竞赛建议使用双模或随机 BASE</p>
          </div>
        </div>
      </div>
    </div>
  )
}

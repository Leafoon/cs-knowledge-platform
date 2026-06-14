'use client'

import { useState } from 'react'

// vEB Tree for u=16: high(x)=x>>2, low(x)=x&3, sqrt(u)=4
// We'll use u=16, so clusters[0..3], each cluster u=4, summary u=4
const U = 16
const SQRTU = 4

function high(x: number) { return Math.floor(x / SQRTU) }
function low(x: number) { return x % SQRTU }

const DEMO_VALUES = [2, 3, 7, 9, 10, 14]

export function VEBTreeStructure() {
  const [selected, setSelected] = useState<number | null>(null)
  const [showFormula, setShowFormula] = useState(false)

  // Compute which clusters are active
  const clusters: Set<number>[] = Array.from({ length: SQRTU }, () => new Set())
  const summaryBits: Set<number> = new Set()
  for (const v of DEMO_VALUES) {
    clusters[high(v)].add(low(v))
    summaryBits.add(high(v))
  }

  const selHigh = selected !== null ? high(selected) : null
  const selLow  = selected !== null ? low(selected)  : null

  return (
    <div className="rounded-2xl border border-amber-200 dark:border-amber-800 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🌳 van Emde Boas Tree 结构（u=16）</h3>
        <p className="text-amber-50 text-xs mt-0.5">
          插入了 {DEMO_VALUES.join(', ')}。点击任意元素追踪 high/low 分解路径
        </p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* 全集合 0..15 可视化 */}
        <div>
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">全域 U = {U}（bit array 视图）</p>
          <div className="flex gap-1 flex-wrap">
            {Array.from({ length: U }, (_, i) => {
              const inSet = DEMO_VALUES.includes(i)
              const isSelected = i === selected
              return (
                <button key={i}
                  onClick={() => setSelected(isSelected ? null : i)}
                  className={`w-8 h-8 rounded-lg text-xs font-mono font-bold transition-all border ${
                    isSelected
                      ? 'bg-orange-500 text-white border-orange-400 scale-110 shadow-lg'
                      : inSet
                      ? 'bg-amber-400 dark:bg-amber-600 text-white border-amber-300 dark:border-amber-500 hover:scale-105'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-600 border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}>
                  {i}
                </button>
              )
            })}
          </div>
          {selected !== null && (
            <p className="text-xs text-orange-600 dark:text-orange-400 mt-2 font-mono">
              x = {selected} → high(x) = ⌊{selected}/√{U}⌋ = <strong>{selHigh}</strong>，
              low(x) = {selected} mod {SQRTU} = <strong>{selLow}</strong>
            </p>
          )}
        </div>

        {/* 两层结构 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Summary */}
          <div className="rounded-xl border border-violet-200 dark:border-violet-800 p-3 bg-violet-50 dark:bg-violet-900/10">
            <p className="text-xs font-bold text-violet-700 dark:text-violet-300 mb-2">Summary（u = {SQRTU}）</p>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 mb-2">记录哪些 cluster 非空</p>
            <div className="flex gap-1">
              {Array.from({ length: SQRTU }, (_, i) => {
                const active = summaryBits.has(i)
                const isHi = i === selHigh
                return (
                  <div key={i} className={`flex flex-col items-center gap-0.5`}>
                    <div className={`w-8 h-8 rounded-md flex items-center justify-center text-xs font-bold border transition-all ${
                      isHi
                        ? 'bg-orange-400 text-white border-orange-300 scale-110 shadow'
                        : active
                        ? 'bg-violet-500 text-white border-violet-400'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-400 border-slate-200 dark:border-slate-700'
                    }`}>{active ? '1' : '0'}</div>
                    <span className="text-[9px] text-slate-400">[{i}]</span>
                  </div>
                )
              })}
            </div>
            <p className="text-[10px] text-violet-600 dark:text-violet-400 mt-2">
              非空 cluster：{[...summaryBits].sort().join(', ')}
            </p>
          </div>

          {/* Clusters */}
          <div className="rounded-xl border border-teal-200 dark:border-teal-800 p-3 bg-teal-50 dark:bg-teal-900/10">
            <p className="text-xs font-bold text-teal-700 dark:text-teal-300 mb-2">Clusters[0..{SQRTU-1}]（各 u = {SQRTU}）</p>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 mb-2">cluster[i] 存 low(x) 值</p>
            <div className="space-y-1.5">
              {clusters.map((cset, ci) => (
                <div key={ci} className={`flex items-center gap-2 rounded-md px-2 py-1 transition-all ${
                  ci === selHigh
                    ? 'bg-orange-100 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800'
                    : cset.size > 0
                    ? 'bg-teal-100 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800'
                    : 'bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800'
                }`}>
                  <span className="text-[10px] font-mono font-bold text-slate-500 dark:text-slate-400 w-12 flex-shrink-0">
                    C[{ci}]:
                  </span>
                  <div className="flex gap-0.5">
                    {Array.from({ length: SQRTU }, (_, li) => {
                      const inCluster = cset.has(li)
                      const highlight = ci === selHigh && li === selLow
                      return (
                        <div key={li} className={`w-6 h-6 rounded text-[10px] font-bold flex items-center justify-center border transition-all ${
                          highlight
                            ? 'bg-orange-500 text-white border-orange-400 scale-110'
                            : inCluster
                            ? 'bg-teal-500 text-white border-teal-400'
                            : 'bg-white dark:bg-slate-900 text-slate-300 dark:text-slate-600 border-slate-200 dark:border-slate-700'
                        }`}>{inCluster ? li : '·'}</div>
                      )
                    })}
                  </div>
                  <span className="text-[10px] text-teal-600 dark:text-teal-400">
                    {cset.size > 0 ? `{${[...cset].sort((a,b)=>a-b).join(',')}}` : '∅'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <button onClick={() => setShowFormula(f => !f)}
          className="text-xs text-amber-600 dark:text-amber-400 hover:underline transition-colors">
          {showFormula ? '▲ 隐藏复杂度推导' : '▼ 展开复杂度推导'}
        </button>
        {showFormula && (
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800 p-4 text-xs space-y-2 border border-slate-200 dark:border-slate-700">
            <p className="font-bold text-slate-700 dark:text-slate-200">为何操作是 O(log log u)？</p>
            <p className="text-slate-600 dark:text-slate-400">
              每次递归将问题规模从 u 缩为 √u：
            </p>
            <div className="font-mono text-center py-2 text-slate-700 dark:text-slate-200">
              T(u) = T(√u) + O(1)
            </div>
            <p className="text-slate-600 dark:text-slate-400">
              令 m = log u，则 T(m) = T(m/2) + O(1) → T(m) = O(log m) = O(log log u)
            </p>
            <p className="text-slate-600 dark:text-slate-400">
              递归深度：u=16 → √16=4 → √4=2（仅 2 层），u=2^64 也只需 6 层。
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

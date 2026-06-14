'use client'
import React, { useState } from 'react'

const A = [
  [1, 2],
  [3, 4],
]
const B = [
  [5, 6],
  [7, 8],
]

const M = {
  M1: { formula: '(a11+a22)(b11+b22)', value: 65, expr: '(1+4)(5+8)' },
  M2: { formula: '(a21+a22)b11', value: 35, expr: '(3+4)×5' },
  M3: { formula: 'a11(b12-b22)', value: -2, expr: '1×(6-8)' },
  M4: { formula: 'a22(b21-b11)', value: 8, expr: '4×(7-5)' },
  M5: { formula: '(a11+a12)b22', value: 24, expr: '(1+2)×8' },
  M6: { formula: '(a21-a11)(b11+b12)', value: 22, expr: '(3-1)×(5+6)' },
  M7: { formula: '(a12-a22)(b21+b22)', value: -30, expr: '(2-4)×(7+8)' },
}

const RESULT = {
  c11: 19,
  c12: 22,
  c21: 43,
  c22: 50,
}

const TABS = [
  { key: 'split', label: '① 划分子块' },
  { key: 'm', label: '② 7 个乘积' },
  { key: 'combine', label: '③ 合并结果' },
  { key: 'compare', label: '④ 复杂度对比' },
] as const

type Tab = typeof TABS[number]['key']

export default function StrassenMatrixMult() {
  const [tab, setTab] = useState<Tab>('split')
  const [activeM, setActiveM] = useState<keyof typeof M>('M1')

  const MatrixCard = ({ title, values, highlight = false, accent = 'slate' }: { title: string; values: number[][]; highlight?: boolean; accent?: 'slate' | 'blue' | 'emerald' | 'violet' }) => {
    const accents = {
      slate: highlight ? 'border-slate-400 dark:border-slate-500 bg-slate-50 dark:bg-slate-800/70' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      blue: highlight ? 'border-blue-400 dark:border-blue-500 bg-blue-50 dark:bg-blue-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      emerald: highlight ? 'border-emerald-400 dark:border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
      violet: highlight ? 'border-violet-400 dark:border-violet-500 bg-violet-50 dark:bg-violet-950/40' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900',
    }

    return (
      <div className={`rounded-xl border p-3 transition-all ${accents[accent]}`}>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">{title}</div>
        <div className="grid grid-cols-2 gap-1 max-w-[120px]">
          {values.flat().map((v, i) => (
            <div key={i} className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 py-2 text-center text-sm font-bold text-slate-800 dark:text-slate-100">{v}</div>
          ))}
        </div>
      </div>
    )
  }

  const active = M[activeM]

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Strassen 矩阵乘法可视化</h3>
        <p className="text-emerald-100 text-sm mt-0.5">7 次子矩阵乘法替代 8 次 · 理论复杂度 O(n^2.807)</p>
      </div>

      <div className="flex flex-wrap border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40">
        {TABS.map(item => (
          <button
            key={item.key}
            onClick={() => setTab(item.key)}
            className={`px-4 py-3 text-xs font-bold transition-colors ${tab === item.key ? 'bg-teal-600 text-white' : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'}`}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-4">
        {tab === 'split' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <MatrixCard title="矩阵 A" values={A} highlight accent="blue" />
                <MatrixCard title="矩阵 B" values={B} highlight accent="emerald" />
              </div>
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                <MatrixCard title="A11" values={[[1, 0], [0, 0]]} accent="blue" />
                <MatrixCard title="A12" values={[[2, 0], [0, 0]]} accent="blue" />
                <MatrixCard title="A21" values={[[3, 0], [0, 0]]} accent="blue" />
                <MatrixCard title="A22" values={[[4, 0], [0, 0]]} accent="blue" />
                <MatrixCard title="B11" values={[[5, 0], [0, 0]]} accent="emerald" />
                <MatrixCard title="B12" values={[[6, 0], [0, 0]]} accent="emerald" />
                <MatrixCard title="B21" values={[[7, 0], [0, 0]]} accent="emerald" />
                <MatrixCard title="B22" values={[[8, 0], [0, 0]]} accent="emerald" />
              </div>
            </div>
            <div className="lg:col-span-2 space-y-3">
              <div className="rounded-2xl border border-teal-200 dark:border-teal-700/60 bg-teal-50 dark:bg-teal-950/30 p-4">
                <div className="text-[10px] uppercase tracking-wider text-teal-600 dark:text-teal-400 font-bold mb-2">分治结构</div>
                <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                  <p>把大矩阵拆成 4 个子块，是分治的 <b>Divide</b>。</p>
                  <p>普通块矩阵乘法会产生 8 个递归乘法。</p>
                  <p>Strassen 的天才之处在于：通过代数重组，只保留 7 个乘法，把多出来的一次乘法变成若干矩阵加减法。</p>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">本示例说明</div>
                <p className="text-[11px] text-slate-600 dark:text-slate-300">为了让演示直观，这里使用 2×2 标量矩阵，等价于 Strassen 在最小块上的一次展开。真正递归实现时，每个块本身仍然是矩阵。</p>
              </div>
            </div>
          </div>
        )}

        {tab === 'm' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(M).map(([key, val]) => {
                  const activeCard = activeM === key
                  return (
                    <button
                      key={key}
                      onClick={() => setActiveM(key as keyof typeof M)}
                      className={`rounded-xl border p-3 text-left transition-all ${activeCard ? 'border-teal-400 dark:border-teal-500 bg-teal-50 dark:bg-teal-950/40 shadow-sm' : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 hover:border-slate-300 dark:hover:border-slate-600'}`}
                    >
                      <div className="text-sm font-black text-slate-800 dark:text-slate-100">{key}</div>
                      <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">{val.formula}</div>
                      <div className="mt-2 text-lg font-black text-teal-600 dark:text-teal-400">{val.value}</div>
                    </button>
                  )
                })}
              </div>
              <div className="mt-4 rounded-2xl border border-teal-200 dark:border-teal-700/60 bg-white dark:bg-slate-900 p-4">
                <div className="text-[10px] uppercase tracking-wider text-teal-600 dark:text-teal-400 font-bold mb-2">当前选中：{activeM}</div>
                <div className="font-mono text-sm text-slate-700 dark:text-slate-200">{active.formula}</div>
                <div className="mt-2 text-[12px] text-slate-500 dark:text-slate-400">代入示例：{active.expr}</div>
                <div className="mt-3 inline-flex px-3 py-1.5 rounded-full bg-teal-50 dark:bg-teal-950/50 border border-teal-200 dark:border-teal-700 text-teal-700 dark:text-teal-300 text-xs font-bold">{activeM} = {active.value}</div>
              </div>
            </div>
            <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">为什么是 7 而不是 8？</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <p>普通块矩阵乘法把每个 Cij 写成两个乘积相加，因此需要 8 次乘法。</p>
                <p>Strassen 把若干交叉项通过加减法提前合成，牺牲了一些加减运算，换来一次乘法的减少。</p>
                <p>因为在大规模矩阵上，“乘法”远比“加减法”昂贵，所以这笔交易在理论上很划算。</p>
              </div>
            </div>
          </div>
        )}

        {tab === 'combine' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">C11</div>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-200">M1 + M4 - M5 + M7</div>
                  <div className="mt-2 text-lg font-black text-emerald-600 dark:text-emerald-400">65 + 8 - 24 - 30 = 19</div>
                </div>
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">C12</div>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-200">M3 + M5</div>
                  <div className="mt-2 text-lg font-black text-emerald-600 dark:text-emerald-400">-2 + 24 = 22</div>
                </div>
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">C21</div>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-200">M2 + M4</div>
                  <div className="mt-2 text-lg font-black text-emerald-600 dark:text-emerald-400">35 + 8 = 43</div>
                </div>
                <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">C22</div>
                  <div className="font-mono text-sm text-slate-700 dark:text-slate-200">M1 - M2 + M3 + M6</div>
                  <div className="mt-2 text-lg font-black text-emerald-600 dark:text-emerald-400">65 - 35 - 2 + 22 = 50</div>
                </div>
              </div>
              <div className="mt-4 rounded-2xl border border-emerald-200 dark:border-emerald-700/60 bg-emerald-50 dark:bg-emerald-950/30 p-4">
                <div className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold mb-2">最终矩阵 C</div>
                <div className="grid grid-cols-2 gap-2 max-w-[170px]">
                  {[RESULT.c11, RESULT.c12, RESULT.c21, RESULT.c22].map((v, i) => (
                    <div key={i} className="rounded-xl bg-white dark:bg-slate-900 border border-emerald-200 dark:border-emerald-700 py-3 text-center text-lg font-black text-emerald-700 dark:text-emerald-300">{v}</div>
                  ))}
                </div>
              </div>
            </div>
            <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">Combine 的工程代价</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <p>Strassen 虽然减少了一次乘法，但增加了很多矩阵加减法和临时块存储。</p>
                <p>所以它是“理论更优、实现更复杂”的代表算法。</p>
                <p>在实际高性能计算中，往往还要结合 cache blocking、SIMD、BLAS 库和阈值回退策略。</p>
              </div>
            </div>
          </div>
        )}

        {tab === 'compare' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">普通分治矩阵乘法</div>
                  <div className="text-sm text-slate-700 dark:text-slate-200 font-mono">T(n) = 8T(n/2) + O(n²)</div>
                  <div className="mt-3 text-2xl font-black text-slate-800 dark:text-slate-100">O(n³)</div>
                </div>
                <div className="rounded-2xl border border-teal-200 dark:border-teal-700/60 bg-teal-50 dark:bg-teal-950/30 p-4">
                  <div className="text-[10px] uppercase tracking-wider text-teal-600 dark:text-teal-400 font-bold mb-2">Strassen</div>
                  <div className="text-sm text-teal-700 dark:text-teal-300 font-mono">T(n) = 7T(n/2) + O(n²)</div>
                  <div className="mt-3 text-2xl font-black text-teal-700 dark:text-teal-300">O(n^2.807)</div>
                </div>
              </div>
              <div className="mt-4 rounded-2xl border border-cyan-200 dark:border-cyan-700/60 bg-cyan-50 dark:bg-cyan-950/30 p-4">
                <div className="text-[10px] uppercase tracking-wider text-cyan-600 dark:text-cyan-400 font-bold mb-2">递归树直觉</div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-xl bg-white dark:bg-slate-900 border border-cyan-200 dark:border-cyan-700 p-3"><div className="text-[10px] text-slate-400">普通</div><div className="text-xl font-black text-slate-800 dark:text-slate-100">8</div><div className="text-[10px] text-slate-500">每层分支</div></div>
                  <div className="rounded-xl bg-white dark:bg-slate-900 border border-cyan-200 dark:border-cyan-700 p-3"><div className="text-[10px] text-slate-400">Strassen</div><div className="text-xl font-black text-cyan-600 dark:text-cyan-300">7</div><div className="text-[10px] text-slate-500">每层分支</div></div>
                  <div className="rounded-xl bg-white dark:bg-slate-900 border border-cyan-200 dark:border-cyan-700 p-3"><div className="text-[10px] text-slate-400">代价</div><div className="text-xl font-black text-slate-800 dark:text-slate-100">+加减</div><div className="text-[10px] text-slate-500">实现复杂</div></div>
                </div>
              </div>
            </div>
            <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold mb-2">何时值得用？</div>
              <div className="space-y-2 text-[11px] text-slate-600 dark:text-slate-300">
                <p>✅ 理论分析、算法课程、理解“减少递归分支”的思想。</p>
                <p>✅ 超大矩阵、且底层实现高度优化时可能体现优势。</p>
                <p>⚠️ 小矩阵、教学代码、普通业务计算中，常常不如朴素乘法 + 优化常数来得实用。</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

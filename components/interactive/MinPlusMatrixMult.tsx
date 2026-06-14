'use client'
import React, { useState } from 'react'

const LABELS = ['A', 'B', 'C', 'D']
const N = 4
const INF = Infinity

// L^(1) = 原图权重矩阵 W
const L1: number[][] = [
  [0,   1,   5,   INF],
  [INF, 0,   2,   4  ],
  [INF, INF, 0,   1  ],
  [3,   INF, INF, 0  ],
]

// (min,+) 矩阵乘法：C = A ⊗ B
function minPlusMul(A: number[][], B: number[][]): number[][] {
  return Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) =>
      Math.min(...Array.from({ length: N }, (_, k) =>
        A[i][k] === INF || B[k][j] === INF ? INF : A[i][k] + B[k][j]
      ))
    )
  )
}

// 获取 (i,j) 单元格的所有 k 选项及其值
function getCellDetail(A: number[][], B: number[][], i: number, j: number) {
  return Array.from({ length: N }, (_, k) => {
    const aik = A[i][k], bkj = B[k][j]
    const val = (aik === INF || bkj === INF) ? INF : aik + bkj
    return { k, aik, bkj, val }
  })
}

const L2 = minPlusMul(L1, L1)
const L4 = minPlusMul(L2, L2)

function fmtNum(v: number) { return v === INF ? '∞' : String(v) }

const ROUNDS = [
  { label: 'L⁽¹⁾ = W', mat: L1, desc: '初始矩阵 L^(1) = W，L^(1)_{ij} = 最多 1 条边的最短路径（即直接边权）。' },
  { label: 'L⁽²⁾ = L⁽¹⁾⊗L⁽¹⁾', mat: L2, desc: 'L^(2) = L^(1) ⊗ L^(1)：允许至多 2 条边的最短路。(i,j) = min_k(L^(1)[i][k] + L^(1)[k][j])。' },
  { label: 'L⁽⁴⁾ = L⁽²⁾⊗L⁽²⁾', mat: L4, desc: 'L^(4) = L^(2) ⊗ L^(2)：允许至多 4 条边的最短路。由于 V=4，V-1=3 条边已足够，L^(4) = 最终 APSP 答案。' },
]

export default function MinPlusMatrixMult() {
  const [roundIdx, setRoundIdx] = useState(0)
  const [sel, setSel] = useState<[number, number] | null>(null)
  const [showCompute, setShowCompute] = useState(false)

  const cur = ROUNDS[roundIdx]
  const prev = roundIdx > 0 ? ROUNDS[roundIdx - 1] : null

  // For selected cell, show the detail
  const detail = sel && prev
    ? getCellDetail(prev.mat, prev.mat, sel[0], sel[1])
    : null

  const selMin = detail ? Math.min(...detail.map(d => d.val)) : INF

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-600 via-rose-600 to-pink-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">(min,+) 矩阵乘法 — 重复平方可视化</h3>
        <p className="text-orange-200 text-sm mt-0.5">
          L^(m)_{'{'}ij{'}'} = min_k(L^(m/2)_{'{'}ik{'}'} + L^(m/2)_{'{'}kj{'}'}) · 点击矩阵单元格查看计算过程
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Round selector */}
        <div className="flex gap-2">
          {ROUNDS.map((r, i) => (
            <button key={i} onClick={() => { setRoundIdx(i); setSel(null); setShowCompute(false) }}
              className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all ${
                i === roundIdx
                  ? 'bg-gradient-to-r from-orange-500 to-rose-500 text-white shadow scale-[1.03]'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}>
              {r.label}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-start">
          {/* Current matrix */}
          <div className="flex-1 min-w-0">
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                {cur.label}
                {roundIdx > 0 && (
                  <span className="ml-2 font-normal text-slate-400">（点击单元格查看计算）</span>
                )}
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="py-2 text-center text-[10px] text-slate-400 bg-slate-50 dark:bg-slate-800/50 w-8" />
                    {LABELS.map((l, j) => (
                      <th key={j} className="py-2 text-center text-xs font-bold text-slate-500 dark:text-slate-400">{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700/50">
                  {LABELS.map((rl, i) => (
                    <tr key={i} className="divide-x divide-slate-100 dark:divide-slate-700/50">
                      <td className="pl-3 text-xs font-bold text-slate-500 dark:text-slate-400">{rl}</td>
                      {LABELS.map((_, j) => {
                        const val = cur.mat[i][j]
                        const isSel = sel?.[0] === i && sel?.[1] === j
                        const isDiag = i === j
                        // Changed from prev round?
                        const changed = roundIdx > 0 && prev && val !== prev.mat[i][j]
                        return (
                          <td key={j}
                            onClick={() => {
                              if (roundIdx > 0) {
                                setSel([i, j])
                                setShowCompute(true)
                              }
                            }}
                            className={`w-12 h-10 text-center text-sm font-mono font-bold transition-all duration-200 ${
                              roundIdx > 0 ? 'cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-900/20' : ''
                            } ${
                              isSel ? 'bg-orange-100 dark:bg-orange-900/40 ring-2 ring-orange-400 ring-inset' :
                              changed ? 'bg-emerald-50 dark:bg-emerald-900/20' :
                              isDiag ? 'bg-slate-50 dark:bg-slate-800/50' : ''
                            } ${
                              isSel ? 'text-orange-700 dark:text-orange-300' :
                              changed ? 'text-emerald-700 dark:text-emerald-300' :
                              isDiag ? 'text-slate-400 dark:text-slate-500' :
                              val === INF ? 'text-slate-300 dark:text-slate-600' :
                              'text-slate-700 dark:text-slate-200'
                            }`}>
                            {fmtNum(val)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Changed cells legend */}
            {roundIdx > 0 && (
              <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-slate-500">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded bg-emerald-300 dark:bg-emerald-700" /> 较上轮改进
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded bg-orange-300 ring-2 ring-orange-400" /> 已选中（点击查看）
                </span>
              </div>
            )}
          </div>

          {/* Right: compute detail */}
          <div className="w-52 shrink-0 space-y-3">
            {/* Semiring info */}
            <div className="rounded-xl border border-orange-200 dark:border-orange-700/50 bg-orange-50 dark:bg-orange-900/20 p-3">
              <div className="text-[11px] font-bold text-orange-700 dark:text-orange-400 uppercase tracking-wide mb-2">
                (min,+) 半环
              </div>
              <div className="space-y-1 text-[11px] font-mono text-orange-800 dark:text-orange-300">
                <div>"加法" = <strong>min</strong></div>
                <div>"乘法" = <strong>+</strong></div>
                <div>加法单位元 = <strong>∞</strong></div>
                <div>乘法单位元 = <strong>0</strong></div>
              </div>
            </div>

            {/* Complexity ladder */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wide border-b border-slate-200 dark:border-slate-700">
                重复平方过程
              </div>
              <div className="p-2 space-y-1.5">
                {[
                  { step: 'L^(1) = W', note: 'O(V²)' },
                  { step: 'L^(2) = L^(1) ⊗ L^(1)', note: 'O(V³)' },
                  { step: 'L^(4) = L^(2) ⊗ L^(2)', note: 'O(V³)' },
                  { step: '···', note: '' },
                  { step: 'L^(2ᵏ)，2ᵏ ≥ V-1', note: 'O(V³ logV)' },
                ].map((item, i) => (
                  <div key={i} className={`text-[10px] font-mono px-2 py-1 rounded ${
                    (i === 0 && roundIdx === 0) || (i === 1 && roundIdx === 1) || (i === 2 && roundIdx === 2)
                      ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 font-bold'
                      : 'text-slate-500 dark:text-slate-400'
                  }`}>
                    <div>{item.step}</div>
                    {item.note && <div className="opacity-70 text-[9px]">{item.note}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Cell computation detail */}
        {showCompute && sel && detail && (
          <div className="rounded-xl border border-orange-200 dark:border-orange-700/50 bg-orange-50 dark:bg-orange-900/20 overflow-hidden">
            <div className="px-3 py-2 text-[11px] font-bold text-orange-700 dark:text-orange-400 uppercase tracking-wide border-b border-orange-200 dark:border-orange-700/30">
              {cur.label}[{LABELS[sel[0]]}][{LABELS[sel[1]]}] 的计算过程
              <span className="ml-2 font-normal">= min over k of {ROUNDS[roundIdx - 1].label}[{LABELS[sel[0]]}][k] + {ROUNDS[roundIdx - 1].label}[k][{LABELS[sel[1]]}]</span>
            </div>
            <div className="p-3 overflow-x-auto">
              <table className="text-xs w-full">
                <thead>
                  <tr className="border-b border-orange-200 dark:border-orange-700/30">
                    <th className="px-3 py-1.5 text-left text-orange-600 dark:text-orange-400 font-bold">中转 k</th>
                    <th className="px-3 py-1.5 text-center text-orange-600 dark:text-orange-400">[{LABELS[sel[0]]}][k]</th>
                    <th className="px-3 py-1.5 text-center text-orange-600 dark:text-orange-400">[k][{LABELS[sel[1]]}]</th>
                    <th className="px-3 py-1.5 text-center text-orange-600 dark:text-orange-400">和</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-orange-100 dark:divide-orange-700/20">
                  {detail.map(d => {
                    const isMin = d.val === selMin
                    return (
                      <tr key={d.k}
                        className={isMin ? 'bg-emerald-50 dark:bg-emerald-900/30' : ''}>
                        <td className="px-3 py-1.5 font-bold text-slate-700 dark:text-slate-200">{LABELS[d.k]}</td>
                        <td className="px-3 py-1.5 text-center font-mono text-slate-600 dark:text-slate-300">
                          {fmtNum(d.aik)}
                        </td>
                        <td className="px-3 py-1.5 text-center font-mono text-slate-600 dark:text-slate-300">
                          {fmtNum(d.bkj)}
                        </td>
                        <td className={`px-3 py-1.5 text-center font-mono font-bold ${
                          isMin ? 'text-emerald-700 dark:text-emerald-400' :
                          d.val === INF ? 'text-slate-300 dark:text-slate-600' :
                          'text-slate-600 dark:text-slate-300'
                        }`}>
                          {fmtNum(d.val)}
                          {isMin && <span className="ml-1 text-emerald-500">← min</span>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-3 py-2 border-t border-orange-200 dark:border-orange-700/30 text-[11px] font-mono font-bold text-orange-700 dark:text-orange-300">
              结果：{cur.label}[{LABELS[sel[0]]}][{LABELS[sel[1]]}] = <strong>{fmtNum(selMin)}</strong>
            </div>
          </div>
        )}

        {/* Description */}
        <div className="rounded-xl bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-700/40 px-4 py-2.5 text-sm text-rose-800 dark:text-rose-300 leading-relaxed">
          {cur.desc}
        </div>
      </div>
    </div>
  )
}

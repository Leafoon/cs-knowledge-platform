'use client'

import { useMemo, useState } from 'react'

const ARR = [7, 2, 9, 1, 5, 3, 8, 6]
const N = ARR.length

type QueryPreset = { l: number; r: number; label: string }
const QUERIES: QueryPreset[] = [
  { l: 1, r: 4, label: 'RMQ[1,4]' },
  { l: 3, r: 8, label: 'RMQ[3,8]' },
  { l: 2, r: 7, label: 'RMQ[2,7]' },
  { l: 5, r: 8, label: 'RMQ[5,8]' },
]

function floorLog2(x: number) {
  return Math.floor(Math.log2(x))
}

function buildSparseTable(arr: number[]) {
  const n = arr.length
  const kMax = floorLog2(n)
  const st: number[][] = Array.from({ length: kMax + 1 }, () => Array(n).fill(Infinity))

  for (let i = 0; i < n; i++) st[0][i] = arr[i]

  for (let k = 1; k <= kMax; k++) {
    const len = 1 << k
    const half = 1 << (k - 1)
    for (let i = 0; i + len - 1 < n; i++) {
      st[k][i] = Math.min(st[k - 1][i], st[k - 1][i + half])
    }
  }

  return st
}

export function SparseTableRMQ() {
  const [queryIdx, setQueryIdx] = useState(0)
  const [showBuild, setShowBuild] = useState(false)
  const [kLevel, setKLevel] = useState(0)

  const st = useMemo(() => buildSparseTable(ARR), [])
  const kMax = st.length - 1

  const query = QUERIES[queryIdx]
  const qLen = query.r - query.l + 1
  const qK = floorLog2(qLen)
  const leftStart = query.l - 1
  const rightStart = query.r - (1 << qK)
  const leftVal = st[qK][leftStart]
  const rightVal = st[qK][rightStart]
  const ans = Math.min(leftVal, rightVal)

  const visibleLevel = showBuild ? kLevel : kMax

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 px-5 py-4">
        <h3 className="text-white font-bold text-base">🟦 Sparse Table：RMQ O(1) 查询演示</h3>
        <p className="text-cyan-100 text-xs mt-1">
          预处理 O(n log n)，查询最小值仅需两段长度为 2^k 的区间
        </p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-800/60">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-2">原数组 A（1-index）</p>
            <div className="flex flex-wrap gap-1.5">
              {ARR.map((v, i) => {
                const inQuery = i + 1 >= query.l && i + 1 <= query.r
                return (
                  <div key={i} className={`px-2 py-1 rounded-lg border text-xs font-mono ${
                    inQuery
                      ? 'bg-sky-100 dark:bg-sky-900/40 border-sky-300 dark:border-sky-700 text-sky-800 dark:text-sky-200'
                      : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300'
                  }`}>
                    A[{i + 1}]={v}
                  </div>
                )
              })}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-800/60 space-y-3">
            <div>
              <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-2">查询区间</p>
              <div className="flex gap-2 flex-wrap">
                {QUERIES.map((q, i) => (
                  <button
                    key={q.label}
                    onClick={() => setQueryIdx(i)}
                    className={`px-2.5 py-1 rounded-md text-xs font-mono border transition-colors ${
                      queryIdx === i
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800'
                    }`}
                  >
                    {q.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 px-3 py-2 text-xs font-mono text-blue-800 dark:text-blue-200 leading-relaxed">
              len = {qLen}, k = floor(log2(len)) = {qK} <br />
              left = st[{qK}][{leftStart}] = {leftVal}, right = st[{qK}][{rightStart}] = {rightVal} <br />
              answer = min(left, right) = <span className="font-bold">{ans}</span>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-white dark:bg-slate-900">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">Sparse Table 构建层级（k 表示区间长度 2^k）</p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => { setShowBuild(true); setKLevel(0) }}
                className="px-3 py-1.5 text-xs rounded-md bg-cyan-600 hover:bg-cyan-700 text-white"
              >
                分层构建模式
              </button>
              <button
                onClick={() => setShowBuild(false)}
                className="px-3 py-1.5 text-xs rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700"
              >
                显示全部
              </button>
            </div>
          </div>

          {showBuild && (
            <div className="flex items-center gap-2 mb-3">
              <button
                onClick={() => setKLevel(v => Math.max(0, v - 1))}
                className="px-2.5 py-1 text-xs rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200"
              >
                ← k-1
              </button>
              <div className="text-xs font-mono text-slate-600 dark:text-slate-300">
                当前层级: k={kLevel}, 区间长度={1 << kLevel}
              </div>
              <button
                onClick={() => setKLevel(v => Math.min(kMax, v + 1))}
                className="px-2.5 py-1 text-xs rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200"
              >
                k+1 →
              </button>
            </div>
          )}

          <div className="space-y-2">
            {st.slice(0, visibleLevel + 1).map((row, k) => {
              const len = 1 << k
              return (
                <div key={k} className="flex items-center gap-2 flex-wrap">
                  <div className="w-36 text-[11px] font-mono text-slate-600 dark:text-slate-300">
                    k={k}, len={len}
                  </div>
                  {row.map((v, i) => {
                    const valid = i + len - 1 < N
                    const isLeftBlock = k === qK && i === leftStart
                    const isRightBlock = k === qK && i === rightStart
                    return (
                      <div
                        key={`${k}-${i}`}
                        className={`w-14 text-center py-1 rounded-md border text-[11px] font-mono ${
                          !valid
                            ? 'opacity-25 bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400'
                            : isLeftBlock || isRightBlock
                            ? 'bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-800 dark:text-blue-200'
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200'
                        }`}
                      >
                        {valid ? v : '-'}
                      </div>
                    )
                  })}
                </div>
              )
            })}
          </div>
        </div>

        <div className="text-center text-xs text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800 pt-3">
          RMQ 只对幂等运算（如 min/max/gcd）可 O(1) 查询；区间和通常改用前缀和或线段树。
        </div>
      </div>
    </div>
  )
}

'use client'

import { useMemo, useState } from 'react'

type FType = 'const' | 'poly' | 'polylog' | 'log'

const CASES = [
  { id: 1, title: 'Case 1', name: '子问题主导', color: 'blue' as const, cond: 'f(n) 增长明显慢于 n^d', result: 'T(n)=Θ(n^d)' },
  { id: 2, title: 'Case 2', name: '层级均衡', color: 'emerald' as const, cond: 'f(n) 与 n^d 同阶（或差一个 log^k）', result: 'T(n)=Θ(n^d log n)' },
  { id: 3, title: 'Case 3', name: '合并主导', color: 'amber' as const, cond: 'f(n) 增长明显快于 n^d', result: 'T(n)=Θ(f(n))' },
]

const PRESETS = [
  { label: '归并排序', a: 2, b: 2, ftype: 'poly' as FType, fp: 1, fk: 0 },
  { label: '二分搜索', a: 1, b: 2, ftype: 'const' as FType, fp: 0, fk: 0 },
  { label: 'Karatsuba', a: 3, b: 2, ftype: 'poly' as FType, fp: 1, fk: 0 },
  { label: 'Strassen', a: 7, b: 2, ftype: 'poly' as FType, fp: 2, fk: 0 },
  { label: '2T(n/2)+nlogn', a: 2, b: 2, ftype: 'polylog' as FType, fp: 1, fk: 1 },
]

function fDisplay(ftype: FType, fp: number, fk: number): string {
  if (ftype === 'const') return 'O(1)'
  if (ftype === 'log') return 'O(log n)'
  if (ftype === 'poly') return fp === 0 ? 'O(1)' : fp === 1 ? 'O(n)' : `O(n^${fp})`
  return fp === 1 ? `O(n log${fk > 1 ? `^${fk}` : ''} n)` : `O(n^${fp} log${fk > 1 ? `^${fk}` : ''} n)`
}

function detectCase(a: number, b: number, ftype: FType, fp: number) {
  if (b <= 1 || a < 1) return { caseNum: 0 as 0 | 1 | 2 | 3, d: 0, eps: 0 }
  const d = Math.log(a) / Math.log(b)
  const fPow = ftype === 'const' || ftype === 'log' ? 0 : fp
  const eps = 0.05
  if (fPow < d - eps) return { caseNum: 1 as const, d, eps: d - fPow }
  if (Math.abs(fPow - d) <= eps) return { caseNum: 2 as const, d, eps: 0 }
  if (fPow > d + eps) return { caseNum: 3 as const, d, eps: fPow - d }
  return { caseNum: 0 as const, d, eps: 0 }
}

function resultText(caseNum: 0 | 1 | 2 | 3, d: number, ftype: FType, fp: number, fk: number, a: number, b: number): string {
  const dStr = Number.isInteger(d) ? String(d) : d.toFixed(3)
  if (caseNum === 1) return `Θ(n^${dStr}) = Θ(n^{log_${b}(${a})})`
  if (caseNum === 2) {
    const extra = ftype === 'polylog' ? fk + 1 : 1
    const base = d === 0 ? '' : d === 1 ? 'n' : `n^${dStr}`
    if (!base) return extra === 1 ? 'Θ(log n)' : `Θ(log^${extra} n)`
    return extra === 1 ? `Θ(${base} log n)` : `Θ(${base} log^${extra} n)`
  }
  if (caseNum === 3) return `Θ(${fDisplay(ftype, fp, fk).replace('O', '').replace(/[()]/g, '')})`
  return '该输入不满足标准主定理形式'
}

export default function MasterTheoremCalculator() {
  const [a, setA] = useState(2)
  const [b, setB] = useState(2)
  const [ftype, setFtype] = useState<FType>('poly')
  const [fp, setFp] = useState(1)
  const [fk, setFk] = useState(0)

  const analysis = useMemo(() => detectCase(a, b, ftype, fp), [a, b, ftype, fp])
  const result = useMemo(() => resultText(analysis.caseNum, analysis.d, ftype, fp, fk, a, b), [analysis, ftype, fp, fk, a, b])

  const activeColor = analysis.caseNum === 1 ? 'blue' : analysis.caseNum === 2 ? 'emerald' : analysis.caseNum === 3 ? 'amber' : 'slate'

  const caseCardClass = (id: 1 | 2 | 3, color: 'blue' | 'emerald' | 'amber') => {
    const active = analysis.caseNum === id
    if (!active) return 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 opacity-60'
    if (color === 'blue') return 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/30'
    if (color === 'emerald') return 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30'
    return 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30'
  }

  const resultClass =
    activeColor === 'blue'
      ? 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300'
      : activeColor === 'emerald'
        ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
        : activeColor === 'amber'
          ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300'
          : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 text-slate-700 dark:text-slate-300'

  return (
    <div className="w-full max-w-4xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Master Theorem 交互计算器</h3>
        <p className="text-blue-100 text-sm mt-0.5">输入 T(n)=aT(n/b)+f(n)，自动判断三种情形并给出复杂度</p>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(p => (
            <button
              key={p.label}
              onClick={() => {
                setA(p.a)
                setB(p.b)
                setFtype(p.ftype)
                setFp(p.fp)
                setFk(p.fk)
              }}
              className="px-3 py-1 rounded-full text-xs font-bold border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-300 hover:border-blue-400 hover:text-blue-500"
            >
              {p.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <label className="space-y-1">
            <span className="text-xs text-slate-500 dark:text-slate-400">a（子问题数）</span>
            <input
              type="number"
              min={1}
              max={32}
              value={a}
              onChange={e => setA(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-800 dark:text-slate-100"
            />
          </label>
          <label className="space-y-1">
            <span className="text-xs text-slate-500 dark:text-slate-400">b（缩小比例）</span>
            <input
              type="number"
              min={2}
              max={32}
              value={b}
              onChange={e => setB(Math.max(2, parseInt(e.target.value) || 2))}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-800 dark:text-slate-100"
            />
          </label>
          <label className="space-y-1 md:col-span-2">
            <span className="text-xs text-slate-500 dark:text-slate-400">f(n) 类型</span>
            <select
              value={ftype}
              onChange={e => setFtype(e.target.value as FType)}
              className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-800 dark:text-slate-100"
            >
              <option value="const">O(1)</option>
              <option value="log">O(log n)</option>
              <option value="poly">O(n^p)</option>
              <option value="polylog">O(n^p log^k n)</option>
            </select>
          </label>
          {(ftype === 'poly' || ftype === 'polylog') && (
            <label className="space-y-1">
              <span className="text-xs text-slate-500 dark:text-slate-400">p 指数</span>
              <input
                type="number"
                min={0}
                max={8}
                step={0.5}
                value={fp}
                onChange={e => setFp(parseFloat(e.target.value) || 0)}
                className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-800 dark:text-slate-100"
              />
            </label>
          )}
          {ftype === 'polylog' && (
            <label className="space-y-1">
              <span className="text-xs text-slate-500 dark:text-slate-400">k 指数</span>
              <input
                type="number"
                min={0}
                max={6}
                value={fk}
                onChange={e => setFk(Math.max(0, parseInt(e.target.value) || 0))}
                className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm text-slate-800 dark:text-slate-100"
              />
            </label>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/30 px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-200">
          T(n) = {a}·T(n/{b}) + {fDisplay(ftype, fp, fk)}
          <span className="ml-3 text-xs text-slate-500 dark:text-slate-400">d = log_{b}({a}) ≈ {analysis.d.toFixed(4)}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {CASES.map(c => (
            <div key={c.id} className={`rounded-xl border p-3 transition-all ${caseCardClass(c.id as 1 | 2 | 3, c.color)}`}>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-bold">{c.title}</div>
              <div className="mt-1 text-sm font-black text-slate-800 dark:text-slate-100">{c.name}</div>
              <div className="mt-2 text-[11px] text-slate-600 dark:text-slate-300">{c.cond}</div>
              <div className="mt-2 text-xs font-mono text-slate-700 dark:text-slate-200">{c.result}</div>
            </div>
          ))}
        </div>

        <div className={`rounded-xl border px-4 py-3 ${resultClass}`}>
          <div className="text-[10px] uppercase tracking-wider font-bold opacity-80">结论</div>
          <div className="mt-1 text-base font-black font-mono">{result}</div>
          {analysis.caseNum > 0 && (
            <div className="mt-2 text-[11px] opacity-90">
              {analysis.caseNum === 1 && `f(n) 比 n^d 慢，递归树底层主导，ε ≈ ${analysis.eps.toFixed(3)}。`}
              {analysis.caseNum === 2 && '每层代价同阶，总层数约 log n，因此多乘一个 log 因子。'}
              {analysis.caseNum === 3 && `f(n) 比 n^d 快，顶层合并主导，ε ≈ ${analysis.eps.toFixed(3)}（需满足正则条件）。`}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

'use client'
import { useState, useEffect } from 'react'

/* ─── 编辑距离 DP 计算 ──────────────────────────────────────────────────── */
type Op = 'match' | 'replace' | 'delete' | 'insert' | 'init'
interface Cell { val: number; op: Op }

function computeED(s: string, t: string): { dp: Cell[][]; steps: [number, number][] } {
  const m = s.length, n = t.length
  const dp: Cell[][] = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => ({
      val: i === 0 ? j : j === 0 ? i : 0,
      op: (i === 0 || j === 0 ? 'init' : 'match') as Op
    }))
  )
  const steps: [number, number][] = []
  for (let i = 0; i <= m; i++) steps.push([i, 0])
  for (let j = 1; j <= n; j++) steps.push([0, j])
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      steps.push([i, j])
      if (s[i - 1] === t[j - 1]) {
        dp[i][j] = { val: dp[i - 1][j - 1].val, op: 'match' }
      } else {
        const del = dp[i - 1][j].val + 1
        const ins = dp[i][j - 1].val + 1
        const rep = dp[i - 1][j - 1].val + 1
        const minVal = Math.min(del, ins, rep)
        dp[i][j] = { val: minVal, op: minVal === del ? 'delete' : minVal === ins ? 'insert' : 'replace' }
      }
    }
  }
  return { dp, steps }
}

function backtrackED(dp: Cell[][], s: string, t: string) {
  const path = new Set<string>()
  const ops: { type: Op; s_i: number; t_j: number }[] = []
  let i = s.length, j = t.length
  while (i > 0 || j > 0) {
    path.add(`${i},${j}`)
    const op = dp[i][j].op
    if (op === 'match' || op === 'replace') { ops.unshift({ type: op, s_i: i, t_j: j }); i--; j-- }
    else if (op === 'delete') { ops.unshift({ type: 'delete', s_i: i, t_j: j }); i-- }
    else if (op === 'insert') { ops.unshift({ type: 'insert', s_i: i, t_j: j }); j-- }
    else { if (i > 0) i--; else j-- }
  }
  path.add('0,0')
  return { path, ops }
}

const PRESETS = [
  { s: 'kitten', t: 'sitting', label: 'kitten→sitting' },
  { s: 'intention', t: 'execution', label: 'intention→execution' },
  { s: 'sunday', t: 'saturday', label: 'sunday→saturday' },
]

const OP_STYLE: Record<Op, { bg: string; text: string; label: string; emoji: string }> = {
  match:   { bg: 'bg-emerald-500 border-emerald-400', text: 'text-white', label: '匹配（无操作）', emoji: '✓' },
  replace: { bg: 'bg-yellow-400 dark:bg-yellow-500 border-yellow-400', text: 'text-slate-900', label: '替换', emoji: '⇄' },
  delete:  { bg: 'bg-rose-500 border-rose-400', text: 'text-white', label: '删除 s[i]', emoji: '✕' },
  insert:  { bg: 'bg-sky-500 border-sky-400', text: 'text-white', label: '插入 t[j]', emoji: '+' },
  init:    { bg: 'bg-slate-500 dark:bg-zinc-600 border-slate-400', text: 'text-white', label: '初始化', emoji: '0' },
}

export default function EditDistanceTable() {
  const [s, setS] = useState('kitten')
  const [t, setT] = useState('sitting')
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [showPath, setShowPath] = useState(false)

  const sc = s.slice(0, 7).toLowerCase()
  const tc = t.slice(0, 7).toLowerCase()
  const { dp, steps } = computeED(sc, tc)
  const maxStep = steps.length - 1
  const done = step >= maxStep
  const { path: pathSet, ops } = done ? backtrackED(dp, sc, tc) : { path: new Set<string>(), ops: [] }

  useEffect(() => { setStep(0); setShowPath(false); setPlaying(false) }, [s, t])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); setShowPath(true); return }
    const id = setTimeout(() => setStep(st => st + 1), 180)
    return () => clearTimeout(id)
  }, [playing, step, maxStep])

  const visible = new Set(steps.slice(0, step + 1).map(([i, j]) => `${i},${j}`))
  const [ci, cj] = steps[Math.min(step, maxStep)]

  const cellStyle = (i: number, j: number) => {
    const vis = visible.has(`${i},${j}`)
    if (!vis) return { cls: 'bg-slate-50 dark:bg-zinc-900 border-slate-200 dark:border-zinc-700 text-slate-200 dark:text-zinc-700', val: '' }
    const isPath = showPath && pathSet.has(`${i},${j}`)
    const curr = i === ci && j === cj
    const cell = dp[i][j]
    if (curr && vis) {
      const os = OP_STYLE[cell.op]
      return { cls: `${os.bg} ${os.text} scale-110 shadow-lg ring-2 ring-white/50 z-10`, val: String(cell.val), emoji: os.emoji }
    }
    if (isPath) {
      const os = OP_STYLE[cell.op]
      return { cls: `${os.bg} ${os.text} scale-105 opacity-90`, val: String(cell.val), emoji: os.emoji }
    }
    if ((i === ci - 1 && j === cj) || (i === ci && j === cj - 1) || (i === ci - 1 && j === cj - 1)) {
      return { cls: 'bg-violet-100 dark:bg-violet-900/60 border-violet-400 text-violet-800 dark:text-violet-200', val: String(cell.val) }
    }
    return { cls: 'bg-white dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-700 dark:text-zinc-200', val: String(cell.val) }
  }

  const currentCell = dp[ci]?.[cj]
  const CELL = 36

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      {/* 头部 */}
      <div className="px-6 py-4 bg-gradient-to-r from-rose-600 to-pink-600">
        <h3 className="text-white font-bold text-base">编辑距离 DP 可视化</h3>
        <p className="text-rose-200 text-sm mt-0.5">Levenshtein Distance — 逐格填充，颜色标注操作类型 + 回溯最优操作序列</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 预设 & 输入 */}
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {PRESETS.map(p => (
              <button key={p.label} onClick={() => { setS(p.s); setT(p.t) }}
                className="px-3 py-1 text-xs rounded-lg bg-rose-50 dark:bg-rose-950 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800 hover:bg-rose-100 dark:hover:bg-rose-900 transition-colors">
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex gap-4 flex-wrap">
            {[{ label: '源字符串 s', val: s, set: setS }, { label: '目标字符串 t', val: t, set: setT }].map(({ label, val, set }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-xs text-slate-500 dark:text-zinc-400 whitespace-nowrap">{label}:</span>
                <input value={val} onChange={e => set(e.target.value.slice(0, 7))} maxLength={7}
                  className="w-28 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-1.5 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-rose-400" />
              </div>
            ))}
          </div>
        </div>

        {/* 控制 */}
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">← 上一格</button>
          <button onClick={() => setPlaying(p => !p)}
            className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium transition-colors ${playing ? 'bg-orange-500 hover:bg-orange-400' : 'bg-rose-600 hover:bg-rose-500'}`}>
            {playing ? '⏸ 暂停' : '▶ 自动填表'}
          </button>
          <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step >= maxStep}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200">下一格 →</button>
          <button onClick={() => { setStep(maxStep); setShowPath(true) }}
            className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white">⚡ 完成 + 回溯</button>
          <button onClick={() => { setStep(0); setPlaying(false); setShowPath(false) }}
            className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200">↺ 重置</button>
          <span className="text-xs text-slate-400 dark:text-zinc-500 ml-auto">格 {step + 1}/{maxStep + 1}</span>
        </div>

        {/* 当前步骤说明 */}
        {ci > 0 && cj > 0 && currentCell && (
          <div className={`px-4 py-2.5 rounded-xl text-sm font-mono border ${
            currentCell.op === 'match' ? 'bg-emerald-50 dark:bg-emerald-950 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200' :
            currentCell.op === 'replace' ? 'bg-yellow-50 dark:bg-yellow-950 border-yellow-200 dark:border-yellow-800 text-yellow-900 dark:text-yellow-200' :
            currentCell.op === 'delete' ? 'bg-rose-50 dark:bg-rose-950 border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200' :
            'bg-sky-50 dark:bg-sky-950 border-sky-200 dark:border-sky-800 text-sky-800 dark:text-sky-200'
          }`}>
            <span className="font-bold">[{ci},{cj}]</span>&nbsp;
            {currentCell.op === 'match'
              ? `s[${ci-1}]='${sc[ci-1]}' == t[${cj-1}]='${tc[cj-1]}' → dp[${ci}][${cj}] = dp[${ci-1}][${cj-1}] = ${currentCell.val}（无操作）`
              : currentCell.op === 'replace'
              ? `替换 s[${ci-1}]='${sc[ci-1]}' → '${tc[cj-1]}'：dp[${ci}][${cj}] = dp[${ci-1}][${cj-1}] + 1 = ${currentCell.val}`
              : currentCell.op === 'delete'
              ? `删除 s[${ci-1}]='${sc[ci-1]}'：dp[${ci}][${cj}] = dp[${ci-1}][${cj}] + 1 = ${currentCell.val}`
              : `插入 t[${cj-1}]='${tc[cj-1]}'：dp[${ci}][${cj}] = dp[${ci}][${cj-1}] + 1 = ${currentCell.val}`}
          </div>
        )}

        {/* DP 表格 */}
        <div className="overflow-auto rounded-xl bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 p-4">
          <table className="border-separate" style={{ borderSpacing: 3 }}>
            <thead>
              <tr>
                <td style={{ width: CELL }} /><td style={{ width: CELL }} />
                {['ε', ...tc.split('')].map((c, j) => (
                  <td key={j} style={{ width: CELL }} className="text-center pb-1">
                    <span className={`text-xs font-bold font-mono ${cj === j ? 'text-rose-600 dark:text-rose-400' : 'text-slate-500 dark:text-zinc-400'}`}>{c}</span>
                  </td>
                ))}
              </tr>
            </thead>
            <tbody>
              {['ε', ...sc.split('')].map((ch, i) => (
                <tr key={i}>
                  <td className="text-right pr-2" style={{ height: CELL }}>
                    <span className={`text-xs font-bold font-mono ${ci === i ? 'text-rose-600 dark:text-rose-400' : 'text-slate-500 dark:text-zinc-400'}`}>{ch}</span>
                  </td>
                  <td style={{ width: 4 }} />
                  {Array.from({ length: tc.length + 1 }, (_, j) => {
                    const { cls, val, emoji } = cellStyle(i, j) as { cls: string; val: string; emoji?: string }
                    return (
                      <td key={j} style={{ width: CELL, height: CELL }}>
                        <div className={`w-full h-full rounded-lg border-2 flex flex-col items-center justify-center transition-all duration-150 ${cls}`}>
                          {emoji && <span className="text-[9px] leading-none opacity-70">{emoji}</span>}
                          <span className="text-xs font-bold leading-tight">{val}</span>
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-3 text-xs">
          {(Object.entries(OP_STYLE) as [Op, typeof OP_STYLE[Op]][]).filter(([k]) => k !== 'init').map(([, { bg, label, emoji }]) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-5 h-5 rounded border-2 flex items-center justify-center text-[9px] text-white ${bg}`}>{emoji}</div>
              <span className="text-slate-600 dark:text-zinc-400">{label}</span>
            </div>
          ))}
        </div>

        {/* 回溯操作序列 */}
        {showPath && ops.length > 0 && (
          <div className="bg-slate-50 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="font-bold text-sm text-slate-700 dark:text-zinc-200">最优操作序列</span>
              <span className="text-xs text-slate-500 dark:text-zinc-400">（共 {dp[sc.length][tc.length].val} 步）</span>
              <span className="ml-auto text-xs font-mono font-bold text-rose-600 dark:text-rose-400">
                "{sc}" → "{tc}"
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {ops.map((op, idx) => {
                const os = OP_STYLE[op.type]
                return (
                  <div key={idx} className={`px-2 py-1 rounded-lg border text-xs font-mono flex items-center gap-1 ${op.type === 'match' ? 'bg-slate-100 dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-500 dark:text-zinc-400' : `${os.bg} ${os.text}`}`}>
                    <span className="font-bold">{os.emoji}</span>
                    <span>{op.type === 'match' ? `'${sc[op.s_i-1]}'` : op.type === 'delete' ? `del '${sc[op.s_i-1]}'` : op.type === 'insert' ? `ins '${tc[op.t_j-1]}'` : `'${sc[op.s_i-1]}'→'${tc[op.t_j-1]}'`}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {done && (
          <div className="bg-rose-50 dark:bg-rose-950 border border-rose-200 dark:border-rose-800 rounded-xl px-4 py-3 flex items-center gap-3">
            <span className="text-2xl">📏</span>
            <div>
              <div className="text-sm font-bold text-rose-800 dark:text-rose-200">编辑距离 = {dp[sc.length][tc.length].val}</div>
              <div className="text-xs text-rose-600 dark:text-rose-400">将 "{sc}" 转换为 "{tc}" 至少需要 {dp[sc.length][tc.length].val} 次操作</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

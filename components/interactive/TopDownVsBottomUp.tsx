'use client'
import { useState, useEffect } from 'react'

/* ─── 计算 Fibonacci 递归树数据 ─────────────────────────────────────── */
interface TreeNode {
  id: string; n: number; result: number; fromMemo: boolean
  children: string[]; x: number; y: number
}

function buildFibTree(N: number) {
  const nodes = new Map<string, TreeNode>()
  const order: string[] = []
  const memo = new Map<number, number>()

  // 先计算 leaf x 分配
  let leafCounter = 0
  const xMap = new Map<string, number>()
  function assignX(id: string, n: number): void {
    if (n <= 1) { xMap.set(id, leafCounter++); return }
    assignX(`${id}L`, n - 1); assignX(`${id}R`, n - 2)
    xMap.set(id, (xMap.get(`${id}L`)! + xMap.get(`${id}R`)!) / 2)
  }
  assignX('r', N)

  function build(id: string, n: number, depth: number): number {
    const cached = memo.has(n)
    let result: number
    const leftId = `${id}L`, rightId = `${id}R`
    if (n <= 1) { result = n }
    else if (cached) { result = memo.get(n)! }
    else {
      const lr = build(leftId, n - 1, depth + 1)
      const rr = build(rightId, n - 2, depth + 1)
      result = lr + rr
    }
    if (!cached) memo.set(n, result)
    order.push(id)
    nodes.set(id, {
      id, n, result, fromMemo: cached,
      children: (n > 1 && !cached) ? [leftId, rightId] : [],
      x: (xMap.get(id) ?? 0) * 52 + 26,
      y: depth * 70 + 28,
    })
    return result
  }
  memo.clear(); build('r', N, 0)
  return { nodes, order }
}

/* ─── Bottom-Up 打表步骤 ─────────────────────────────────────────────── */
function buildBUSteps(N: number) {
  const table: number[] = []
  const deps: number[][] = []
  for (let i = 0; i <= N; i++) {
    if (i <= 1) { table.push(i); deps.push([]) }
    else { table.push(table[i-1] + table[i-2]); deps.push([i-1, i-2]) }
  }
  return { table, deps }
}

const PRESETS = [
  { n: 4, label: 'F(4)' }, { n: 5, label: 'F(5)' },
  { n: 6, label: 'F(6)' }, { n: 7, label: 'F(7)' },
]

export default function TopDownVsBottomUp() {
  const [N, setN] = useState(5)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const { nodes, order } = buildFibTree(N)
  const { table, deps } = buildBUSteps(N)
  const maxStep = order.length - 1

  useEffect(() => { setStep(0); setPlaying(false) }, [N])
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s => s + 1), 480)
    return () => clearTimeout(id)
  }, [playing, step, maxStep])

  const visibleIds = new Set(order.slice(0, step + 1))
  const allNodes = [...nodes.values()]
  const svgW = Math.max(...allNodes.map(n => n.x)) + 40
  const svgH = Math.max(...allNodes.map(n => n.y)) + 30

  // BU 步骤与 TD 步骤同步，取最小
  const buStep = Math.min(step, N)

  const nodeStyle = (node: TreeNode) => {
    if (!visibleIds.has(node.id)) return null
    if (node.fromMemo) return { fill: '#78350f', stroke: '#f59e0b', text: '#fef3c7' }
    if (node.n <= 1)   return { fill: '#14532d', stroke: '#4ade80', text: '#d1fae5' }
    return { fill: '#1e3a5f', stroke: '#60a5fa', text: '#dbeafe' }
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 overflow-hidden">
      {/* 头部 */}
      <div className="px-6 py-4 bg-gradient-to-r from-violet-600 to-indigo-600">
        <h3 className="text-white font-bold text-base">Top-Down vs Bottom-Up 对比</h3>
        <p className="text-violet-200 text-sm mt-0.5">以 Fibonacci(n) 为例，同步演示记忆化递归树 与 迭代填表</p>
      </div>

      <div className="p-5 space-y-4">
        {/* 控制 */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-sm text-slate-500 dark:text-zinc-400">n =</span>
            {PRESETS.map(({ n, label }) => (
              <button key={n} onClick={() => setN(n)}
                className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${N === n ? 'bg-violet-600 text-white shadow' : 'bg-slate-100 dark:bg-zinc-800 text-slate-600 dark:text-zinc-300 hover:bg-slate-200 dark:hover:bg-zinc-700'}`}>
                {label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <button onClick={() => setStep(s => Math.max(0, s-1))} disabled={step === 0}
              className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一步</button>
            <button onClick={() => setPlaying(p => !p)}
              className={`px-3 py-1.5 text-xs rounded-lg text-white font-medium transition-colors ${playing ? 'bg-orange-500 hover:bg-orange-400' : 'bg-violet-600 hover:bg-violet-500'}`}>
              {playing ? '⏸ 暂停' : '▶ 播放'}
            </button>
            <button onClick={() => setStep(s => Math.min(maxStep, s+1))} disabled={step >= maxStep}
              className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">下一步 →</button>
            <button onClick={() => { setStep(0); setPlaying(false) }}
              className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-zinc-800 hover:bg-slate-200 dark:hover:bg-zinc-700 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺ 重置</button>
          </div>
          <span className="text-xs text-slate-400 dark:text-zinc-500">步骤 {step + 1} / {maxStep + 1}</span>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-4 text-xs">
          {[
            { bg: 'bg-blue-900/60 border-blue-400', label: '正常递归', textColor: 'text-blue-300' },
            { bg: 'bg-yellow-900/60 border-yellow-400', label: '⚡ 命中 Memo（跳过递归）', textColor: 'text-yellow-300' },
            { bg: 'bg-green-900/60 border-green-400', label: 'F(0)/F(1) 边界', textColor: 'text-green-300' },
          ].map(({ bg, label, textColor }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-3.5 h-3.5 rounded-full border ${bg}`} />
              <span className={`dark:${textColor} text-slate-600`}>{label}</span>
            </div>
          ))}
        </div>

        {/* 双栏 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left: 递归树 */}
          <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="px-2 py-0.5 text-xs font-bold rounded bg-violet-100 dark:bg-violet-950 text-violet-700 dark:text-violet-300">Top-Down</span>
              <span className="text-sm text-slate-700 dark:text-zinc-200 font-medium">记忆化递归树</span>
            </div>
            <div className="overflow-auto rounded-lg bg-zinc-950 p-2">
              <svg width={svgW} height={svgH} className="block mx-auto min-w-0">
                {allNodes.map(node => {
                  const st = nodeStyle(node)
                  if (!st) return null
                  return node.children.map(cid => {
                    const child = nodes.get(cid)
                    if (!child || !visibleIds.has(cid)) return null
                    return <line key={cid} x1={node.x} y1={node.y+18} x2={child.x} y2={child.y-18} stroke="#4b5563" strokeWidth={1.5} />
                  })
                })}
                {allNodes.map(node => {
                  const st = nodeStyle(node)
                  if (!st) return null
                  return (
                    <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                      <circle cx={0} cy={0} r={20} fill={st.fill} stroke={st.stroke} strokeWidth={2} />
                      <text x={0} y={-5} textAnchor="middle" fontSize={9} fill={st.text} fontWeight="bold">F({node.n})</text>
                      <text x={0} y={8} textAnchor="middle" fontSize={12} fill={st.text}>{node.result}</text>
                      {node.fromMemo && <text x={0} y={30} textAnchor="middle" fontSize={8} fill="#f59e0b">cached</text>}
                    </g>
                  )
                })}
              </svg>
            </div>
            <div className="mt-2 text-xs text-center text-slate-500 dark:text-zinc-500">
              展开节点 {visibleIds.size} 个 · 命中缓存 {[...visibleIds].filter(id => nodes.get(id)?.fromMemo).length} 次
            </div>
          </div>

          {/* Right: Bottom-Up 填表 */}
          <div className="bg-slate-50 dark:bg-zinc-900 rounded-xl border border-slate-200 dark:border-zinc-700 p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="px-2 py-0.5 text-xs font-bold rounded bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">Bottom-Up</span>
              <span className="text-sm text-slate-700 dark:text-zinc-200 font-medium">迭代填表（dp[0..n]）</span>
            </div>
            {/* dp 格子 */}
            <div className="flex flex-wrap gap-2 justify-center py-4">
              {table.map((val, idx) => {
                const filled = idx <= buStep
                const curr = idx === buStep
                const isRef = deps[buStep]?.includes(idx)
                return (
                  <div key={idx} className="flex flex-col items-center gap-1">
                    <div className={`w-12 h-14 rounded-xl flex flex-col items-center justify-center border-2 transition-all duration-300 font-mono ${
                      curr ? 'bg-violet-600 border-violet-400 text-white scale-110 shadow-lg' :
                      isRef ? 'bg-indigo-800 border-indigo-400 text-indigo-100 scale-105' :
                      filled ? 'bg-slate-700 dark:bg-zinc-700 border-slate-500 dark:border-zinc-500 text-slate-200' :
                      'bg-slate-100 dark:bg-zinc-800 border-slate-300 dark:border-zinc-600 text-slate-300 dark:text-zinc-600'
                    }`}>
                      <span className="text-[9px] opacity-60 mb-0.5">i={idx}</span>
                      <span className="text-base font-bold">{filled ? val : '?'}</span>
                    </div>
                  </div>
                )
              })}
            </div>
            {/* 当前步骤说明 */}
            <div className={`px-3 py-2 rounded-lg text-xs font-mono ${
              buStep <= 1
                ? 'bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300'
                : 'bg-indigo-50 dark:bg-indigo-950 border border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300'
            }`}>
              {buStep <= 1
                ? `dp[${buStep}] = ${table[buStep]}（边界条件，直接赋值）`
                : `dp[${buStep}] = dp[${buStep-1}] + dp[${buStep-2}] = ${table[buStep-1]} + ${table[buStep-2]} = ${table[buStep]}`}
            </div>
          </div>
        </div>

        {/* 对比总结 */}
        <div className="grid grid-cols-2 gap-3 text-xs">
          {[
            { tag: 'Top-Down', color: 'violet', pros: ['✅ 代码贴近数学定义', '✅ 只算必要子问题'], cons: ['⚠️ 递归栈深 O(n)', '⚠️ 函数调用开销大'] },
            { tag: 'Bottom-Up', color: 'indigo', pros: ['✅ 无栈溢出风险', '✅ 可滚动数组压缩空间'], cons: ['⚠️ 需确定填表顺序', '⚠️ 稀疏状态浪费时间'] },
          ].map(({ tag, color, pros, cons }) => (
            <div key={tag} className={`rounded-xl border p-3 bg-${color}-50 dark:bg-${color}-950/40 border-${color}-200 dark:border-${color}-800`}>
              <div className={`font-bold mb-1.5 text-${color}-700 dark:text-${color}-300`}>{tag}</div>
              <ul className={`space-y-0.5 text-${color}-700/80 dark:text-${color}-300/80`}>
                {[...pros, ...cons].map(t => <li key={t}>{t}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

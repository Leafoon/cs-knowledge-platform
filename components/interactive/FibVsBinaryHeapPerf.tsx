'use client'

import { useState } from 'react'

// ─── 数据定义 ──────────────────────────────────────────────────
type ComplexityClass = 'O(1)' | 'O(log n)' | 'O(n)' | 'O(n log n)' | 'O(1)*'

interface OperationRow {
  op: string
  icon: string
  desc: string
  binary: ComplexityClass
  binomial: ComplexityClass
  fibonacci: ComplexityClass
  note?: string
}

const OPERATIONS: OperationRow[] = [
  {
    op: 'INSERT',
    icon: '➕',
    desc: '插入新节点',
    binary: 'O(log n)',
    binomial: 'O(log n)',
    fibonacci: 'O(1)*',
    note: 'Fibonacci 均摊 O(1)，最坏 O(log n)',
  },
  {
    op: 'MINIMUM',
    icon: '🔍',
    desc: '查找最小值',
    binary: 'O(1)',
    binomial: 'O(log n)',
    fibonacci: 'O(1)',
    note: 'Fibonacci 维护 min 指针',
  },
  {
    op: 'EXTRACT-MIN',
    icon: '⬆️',
    desc: '提取最小值',
    binary: 'O(log n)',
    binomial: 'O(log n)',
    fibonacci: 'O(log n)',
    note: 'Fibonacci 均摊 O(log n)，触发 CONSOLIDATE',
  },
  {
    op: 'UNION',
    icon: '🔗',
    desc: '合并两个堆',
    binary: 'O(n)',
    binomial: 'O(log n)',
    fibonacci: 'O(1)',
    note: 'Fibonacci 仅拼接根链表',
  },
  {
    op: 'DECREASE-KEY',
    icon: '📉',
    desc: '减小键值',
    binary: 'O(log n)',
    binomial: 'O(log n)',
    fibonacci: 'O(1)*',
    note: 'Fibonacci 均摊 O(1)，实现 Dijkstra 关键优化',
  },
  {
    op: 'DELETE',
    icon: '🗑️',
    desc: '删除任意节点',
    binary: 'O(log n)',
    binomial: 'O(log n)',
    fibonacci: 'O(log n)',
    note: 'DECREASE-KEY(-∞) + EXTRACT-MIN',
  },
]

type ComplexityColor = 'green' | 'yellow' | 'orange' | 'red'

function getColorClass(c: ComplexityClass, element: 'bg' | 'text' | 'border' | 'badge'): string {
  const palette: Record<ComplexityColor, Record<string, string>> = {
    green: {
      bg: 'bg-emerald-50 dark:bg-emerald-900/20',
      text: 'text-emerald-700 dark:text-emerald-300',
      border: 'border-emerald-200 dark:border-emerald-700',
      badge: 'bg-emerald-100 dark:bg-emerald-800 text-emerald-700 dark:text-emerald-200 border border-emerald-200 dark:border-emerald-600',
    },
    yellow: {
      bg: 'bg-yellow-50 dark:bg-yellow-900/20',
      text: 'text-yellow-700 dark:text-yellow-300',
      border: 'border-yellow-200 dark:border-yellow-700',
      badge: 'bg-yellow-100 dark:bg-yellow-900/60 text-yellow-700 dark:text-yellow-200 border border-yellow-200 dark:border-yellow-600',
    },
    orange: {
      bg: 'bg-orange-50 dark:bg-orange-900/20',
      text: 'text-orange-700 dark:text-orange-300',
      border: 'border-orange-200 dark:border-orange-700',
      badge: 'bg-orange-100 dark:bg-orange-900/60 text-orange-700 dark:text-orange-200 border border-orange-200 dark:border-orange-600',
    },
    red: {
      bg: 'bg-red-50 dark:bg-red-900/20',
      text: 'text-red-700 dark:text-red-300',
      border: 'border-red-200 dark:border-red-700',
      badge: 'bg-red-100 dark:bg-red-900/60 text-red-700 dark:text-red-200 border border-red-200 dark:border-red-600',
    },
  }
  const color: ComplexityColor = c === 'O(1)' || c === 'O(1)*' ? 'green' : c === 'O(log n)' ? 'yellow' : c === 'O(n)' ? 'orange' : 'red'
  return palette[color][element]
}

// 简化的分数：O(1)=3, O(log n)=2, O(n)=1
function getScore(c: ComplexityClass): number {
  if (c === 'O(1)' || c === 'O(1)*') return 3
  if (c === 'O(log n)') return 2
  if (c === 'O(n)') return 1
  return 0
}

// ─── 使用场景分析 ──────────────────────────────────────────────
interface UsageScenario {
  name: string
  icon: string
  desc: string
  primaryOps: string[]
  winner: 'fibonacci' | 'binomial' | 'binary' | 'tie'
  analysis: string
}

const USAGE_SCENARIOS: UsageScenario[] = [
  {
    name: "Dijkstra 最短路",
    icon: "🗺️",
    desc: "密集图（edges >> nodes）",
    primaryOps: ['INSERT', 'DECREASE-KEY', 'EXTRACT-MIN'],
    winner: 'fibonacci',
    analysis: 'DECREASE-KEY 均摊 O(1) 使 Dijkstra 总复杂度从 O(E log V) 降至 O(E + V log V)，对稠密图是质的提升。',
  },
  {
    name: "Prim 最小生成树",
    icon: "🌲",
    desc: "稠密图中执行",
    primaryOps: ['INSERT', 'DECREASE-KEY', 'EXTRACT-MIN'],
    winner: 'fibonacci',
    analysis: '同 Dijkstra，DECREASE-KEY O(1) 使 Prim 达到最优 O(E + V log V)。',
  },
  {
    name: "纯优先队列",
    icon: "📋",
    desc: "反复 INSERT + EXTRACT-MIN",
    primaryOps: ['INSERT', 'EXTRACT-MIN'], 
    winner: 'binary',
    analysis: 'Fibonacci 堆常数因子大，仅 INSERT/EXTRACT-MIN 时实际常数开销可能反超二叉堆。',
  },
  {
    name: "堆合并操作",
    icon: "🔀",
    desc: "频繁合并不同优先队列",
    primaryOps: ['UNION'],
    winner: 'fibonacci',
    analysis: 'UNION O(1) vs 二叉堆 O(n)，场景：合并多个工作队列、网络流分区算法。',
  },
]

const HEAP_META = {
  binary:    { label: '二叉堆',   color: 'blue',  dot: 'bg-blue-500' },
  binomial:  { label: '二项堆',   color: 'violet', dot: 'bg-violet-500' },
  fibonacci: { label: 'Fibonacci 堆', color: 'emerald', dot: 'bg-emerald-500' },
}

export function FibVsBinaryHeapPerf() {
  const [highlighted, setHighlighted] = useState<string | null>(null)
  const [activeScenario, setActiveScenario] = useState<number | null>(null)

  // 总分
  const scores = {
    binary:    OPERATIONS.reduce((s, o) => s + getScore(o.binary), 0),
    binomial:  OPERATIONS.reduce((s, o) => s + getScore(o.binomial), 0),
    fibonacci: OPERATIONS.reduce((s, o) => s + getScore(o.fibonacci), 0),
  }
  const maxScore = OPERATIONS.length * 3

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部：深色仪表盘风格 */}
      <div className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 px-5 py-5 border-b border-slate-600">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-white font-bold text-base">📊 堆性能对比仪表盘</h3>
            <p className="text-slate-400 text-xs mt-0.5">二叉堆 · 二项堆 · Fibonacci 堆 — 各操作复杂度全景对比</p>
          </div>
          {/* 总分卡片 */}
          <div className="flex gap-2 shrink-0">
            {(Object.keys(scores) as Array<keyof typeof scores>).map(heap => (
              <div key={heap} className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-600 text-center min-w-[72px]">
                <div className={`text-xs font-medium mb-0.5 ${
                  heap === 'fibonacci' ? 'text-emerald-400' : heap === 'binomial' ? 'text-violet-400' : 'text-blue-400'}`}>
                  {HEAP_META[heap].label.split(' ')[0]}
                </div>
                <div className="text-white font-bold text-lg leading-tight">{scores[heap]}</div>
                <div className="text-slate-500 text-[10px]">/{maxScore} 分</div>
                {/* 迷你进度条 */}
                <div className="mt-1 h-1 rounded-full bg-slate-700">
                  <div className={`h-full rounded-full transition-all ${
                    heap === 'fibonacci' ? 'bg-emerald-400' : heap === 'binomial' ? 'bg-violet-400' : 'bg-blue-400'}`}
                    style={{ width: `${(scores[heap] / maxScore) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 图例 */}
        <div className="flex gap-4 mt-3 flex-wrap">
          {[
            { c: 'O(1)' as ComplexityClass, label: 'O(1) / O(1)* 均摊', dot: 'bg-emerald-400' },
            { c: 'O(log n)' as ComplexityClass, label: 'O(log n)', dot: 'bg-yellow-400' },
            { c: 'O(n)' as ComplexityClass, label: 'O(n)', dot: 'bg-orange-400' },
          ].map(item => (
            <div key={item.c} className="flex items-center gap-1.5">
              <div className={`w-2.5 h-2.5 rounded-full ${item.dot}`} />
              <span className="text-xs text-slate-400">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900">
        {/* ── 主对比表格 ── */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider w-36">
                  操作
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                  二叉堆
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-violet-600 dark:text-violet-400 uppercase tracking-wider">
                  二项堆
                </th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
                  Fibonacci 堆
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {OPERATIONS.map((row) => {
                const isHighlighted = highlighted === row.op
                return (
                  <tr
                    key={row.op}
                    onMouseEnter={() => setHighlighted(row.op)}
                    onMouseLeave={() => setHighlighted(null)}
                    className={`transition-colors cursor-default ${isHighlighted ? 'bg-slate-50 dark:bg-slate-800/50' : ''}`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="text-base">{row.icon}</span>
                        <div>
                          <div className="font-mono text-xs font-bold text-slate-800 dark:text-slate-100">{row.op}</div>
                          <div className="text-[11px] text-slate-400 dark:text-slate-500">{row.desc}</div>
                        </div>
                      </div>
                    </td>
                    {(['binary', 'binomial', 'fibonacci'] as const).map(heap => {
                      const val = row[heap]
                      return (
                        <td key={heap} className="px-4 py-3 text-center">
                          <span className={`inline-block px-2.5 py-1 rounded-lg text-xs font-bold font-mono ${getColorClass(val, 'badge')}`}>
                            {val}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* 悬停注释 */}
        <div className={`px-4 py-3 border-t border-slate-100 dark:border-slate-800 min-h-[44px] transition-all duration-200`}>
          {highlighted && (() => {
            const row = OPERATIONS.find(o => o.op === highlighted)
            return row?.note ? (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                <span className="font-bold text-slate-700 dark:text-slate-300">{row.icon} {row.op}：</span>
                {row.note}
              </p>
            ) : null
          })()}
          {!highlighted && (
            <p className="text-xs text-slate-400 dark:text-slate-600">↑ 将鼠标悬停在行上查看操作说明</p>
          )}
        </div>

        {/* ── 条形图可视化 ── */}
        <div className="px-5 py-5 border-t border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-4">
            各操作评分对比（O(1)=3分 · O(log n)=2分 · O(n)=1分）
          </div>
          <div className="space-y-3">
            {OPERATIONS.map(row => {
              const vals = { binary: getScore(row.binary), binomial: getScore(row.binomial), fibonacci: getScore(row.fibonacci) }
              return (
                <div key={row.op}
                  onMouseEnter={() => setHighlighted(row.op)}
                  onMouseLeave={() => setHighlighted(null)}
                  className="flex items-center gap-3 group cursor-default">
                  <div className={`text-[10px] font-mono font-bold w-24 text-right shrink-0 transition-colors ${
                    highlighted === row.op ? 'text-slate-700 dark:text-slate-200' : 'text-slate-400 dark:text-slate-500'}`}>
                    {row.op}
                  </div>
                  <div className="flex-1 space-y-1">
                    {(Object.keys(vals) as Array<keyof typeof vals>).map(heap => (
                      <div key={heap} className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full shrink-0 ${HEAP_META[heap].dot}`} />
                        <div className="flex-1 h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              heap === 'fibonacci' ? 'bg-emerald-400' : heap === 'binomial' ? 'bg-violet-400' : 'bg-blue-400'}`}
                            style={{ width: `${(vals[heap] / 3) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* ── 使用场景分析 ── */}
        <div className="px-5 py-5 border-t border-slate-100 dark:border-slate-800">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">典型使用场景分析</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {USAGE_SCENARIOS.map((scene, i) => {
              const isActive = activeScenario === i
              const winnerMeta = scene.winner === 'tie' ? HEAP_META['binary'] : HEAP_META[scene.winner]
              return (
                <div
                  key={i}
                  onClick={() => setActiveScenario(isActive ? null : i)}
                  className={`rounded-xl border-2 p-3 cursor-pointer transition-all duration-200 ${
                    isActive
                      ? `${scene.winner === 'fibonacci' ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20' :
                          scene.winner === 'binary' ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' :
                          'border-violet-400 bg-violet-50 dark:bg-violet-900/20'}`
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{scene.icon}</span>
                      <div>
                        <div className="text-sm font-bold text-slate-800 dark:text-slate-100">{scene.name}</div>
                        <div className="text-xs text-slate-400 dark:text-slate-500">{scene.desc}</div>
                      </div>
                    </div>
                    <div className={`shrink-0 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      scene.winner === 'fibonacci' ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300' :
                      scene.winner === 'binary' ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' :
                      'bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300'}`}>
                      ✓ {winnerMeta.label.split(' ')[0]}
                    </div>
                  </div>
                  {/* 关键操作 */}
                  <div className="flex gap-1 flex-wrap mb-1.5">
                    {scene.primaryOps.map(op => {
                      const row = OPERATIONS.find(r => r.op === op)
                      const val: ComplexityClass = row?.fibonacci ?? 'O(log n)'
                      return (
                        <span key={op} className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${getColorClass(val, 'badge')}`}>
                          {op}
                        </span>
                      )
                    })}
                  </div>
                  {isActive && (
                    <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed border-t border-slate-200 dark:border-slate-700 pt-2 mt-1">
                      {scene.analysis}
                    </p>
                  )}
                  {!isActive && (
                    <p className="text-[10px] text-slate-400 dark:text-slate-600">点击展开分析 →</p>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* ── 关键结论 ── */}
        <div className="px-5 pb-5">
          <div className="bg-gradient-to-r from-slate-800 to-slate-700 dark:from-slate-800 dark:to-slate-900 rounded-xl p-4 border border-slate-600 dark:border-slate-700">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">核心结论</div>
            <div className="space-y-1.5 text-xs text-slate-300">
              <div className="flex items-start gap-2">
                <span className="text-emerald-400 font-bold shrink-0 mt-0.5">①</span>
                <span>Fibonacci 堆在理论上占优：6 个操作中 5 个達到最优复杂度，特别是 DECREASE-KEY 均摊 O(1)。</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400 font-bold shrink-0 mt-0.5">②</span>
                <span>实践中，Fibonacci 堆常数因子大、缓存局部性差，纯 INSERT/EXTRACT-MIN 场景下二叉堆往往更快。</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-400 font-bold shrink-0 mt-0.5">③</span>
                <span>需要 DECREASE-KEY 的图算法（Dijkstra、Prim）配合稠密图时，Fibonacci 堆提供理论上不可超越的 O(E + V log V)。</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

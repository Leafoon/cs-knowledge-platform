'use client'
import { useState } from 'react'

/* ─── 回溯 vs DP vs 贪心 交互决策树 ─────────────────────── */
interface Question {
  id: number; text: string; hint: string
  yes: string | number; no: string | number
}
const QUESTIONS: Question[] = [
  { id: 0, text: '需要找出所有满足约束的解，或判断解是否存在？',
    hint: '如：所有排列 / 所有子集 / 能否拼出目标金额（存在性）',
    yes: 'backtracking', no: 1 },
  { id: 1, text: '每步局部最优能保证全局最优？（贪心选择性质）',
    hint: '如：活动按结束时间排序选最多 → 全局最多；硬币若面值任意则不成立',
    yes: 'greedy', no: 2 },
  { id: 2, text: '子问题是否重叠？相同状态会被重复求解？',
    hint: '如：dp[i][j] 被多条路径依赖 = 重叠；每条路径状态唯一 = 不重叠',
    yes: 'dp', no: 'bnb' },
]
const RESULTS = {
  backtracking: {
    name: '回 溯', en: 'Backtracking',
    icon: '🌲', gradient: 'from-violet-500 to-purple-600',
    bg: 'bg-violet-50 dark:bg-violet-950/40', border: 'border-violet-300 dark:border-violet-700',
    textColor: 'text-violet-700 dark:text-violet-300',
    time: 'O(kⁿ) 最坏', space: 'O(n) 递归栈',
    examples: ['N 皇后 #51', '全排列 #46', '子集 #78', '数独 #37', '组合总和 #39'],
    tip: '三步心法：① 选择  ② 递归  ③ 撤销。关键：可行性剪枝。',
  },
  greedy: {
    name: '贪 心', en: 'Greedy',
    icon: '⚡', gradient: 'from-emerald-500 to-teal-600',
    bg: 'bg-emerald-50 dark:bg-emerald-950/40', border: 'border-emerald-300 dark:border-emerald-700',
    textColor: 'text-emerald-700 dark:text-emerald-300',
    time: 'O(n log n) 通常', space: 'O(1) 通常',
    examples: ['活动选择', '分数背包', 'Huffman 编码', 'Dijkstra 最短路'],
    tip: '必须严格证明贪心选择性质，否则可能得出错误答案。',
  },
  dp: {
    name: '动态规划', en: 'Dynamic Programming',
    icon: '📊', gradient: 'from-blue-500 to-cyan-600',
    bg: 'bg-blue-50 dark:bg-blue-950/40', border: 'border-blue-300 dark:border-blue-700',
    textColor: 'text-blue-700 dark:text-blue-300',
    time: 'O(n²) 或 O(n³) 常见', space: 'O(n) 或 O(n²)',
    examples: ['0/1 背包', 'LCS / LIS', '区间 DP', '矩阵链乘法', '编辑距离'],
    tip: '状态定义是核心；找清楚「状态转移方程」后代码即水到渠成。',
  },
  bnb: {
    name: '分支限界', en: 'Branch & Bound',
    icon: '🎯', gradient: 'from-orange-500 to-red-500',
    bg: 'bg-orange-50 dark:bg-orange-950/40', border: 'border-orange-300 dark:border-orange-700',
    textColor: 'text-orange-700 dark:text-orange-300',
    time: '指数（依赖界函数质量）', space: 'O(活跃节点)',
    examples: ['TSP 精确解', '0/1 背包精确解', '整数线性规划'],
    tip: '界函数越紧，剪枝效果越好；太松则等同暴力搜索。',
  },
}

// 决策树 SVG 节点位置
const NODE_POS = [
  { x: 260, y: 40 },   // Q0
  { x: 130, y: 140 },  // ← Q0 no → Q1
  { x: 80,  y: 240 },  // ← Q1 no → Q2
  // result leaves
  { x: 390, y: 140 },   // Q0 yes → backtracking
  { x: 185, y: 240 },  // Q1 yes → greedy
  { x: 370, y: 240 },  // Q2 no → bnb
  { x: 50,  y: 340 },  // Q2 no → bnb (below Q2)
  { x: 120, y: 340 },  // Q2 yes → dp
]

export default function BacktrackVsDPChoice() {
  const [current, setCurrent] = useState<number | string>(0)
  const [path, setPath]       = useState<{ q: number; ans: boolean }[]>([])

  function answer(yes: boolean) {
    const q = QUESTIONS[current as number]
    const next = yes ? q.yes : q.no
    setPath(p => [...p, { q: current as number, ans: yes }])
    setCurrent(next)
  }
  function reset() { setCurrent(0); setPath([]) }

  const isResult = typeof current === 'string'
  const result   = isResult ? RESULTS[current as keyof typeof RESULTS] : null

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🤔</span>
        <div>
          <h3 className="text-white font-bold text-base leading-tight">算法选择向导</h3>
          <p className="text-amber-100 text-xs">回溯 · 动态规划 · 贪心 · 分支限界</p>
        </div>
        <div className="ml-auto flex gap-1.5">
          {[0,1,2].map(i => (
            <div key={i} className={`w-2 h-2 rounded-full transition-all ${
              typeof current === 'number' && current >= i ? 'bg-white' : 'bg-amber-300/50'
            } ${isResult ? 'bg-white' : ''}`}/>
          ))}
        </div>
      </div>

      <div className="p-6 flex flex-col gap-5">
        {/* ── Breadcrumb path ── */}
        {path.length > 0 && (
          <div className="flex flex-wrap gap-2 text-xs">
            {path.map((p, i) => (
              <span key={i} className="flex items-center gap-1">
                <span className="text-slate-400 dark:text-slate-500">Q{p.q+1}:</span>
                <span className={`px-2 py-0.5 rounded-full font-medium ${
                  p.ans ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                        : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                }`}>{p.ans ? '是' : '否'}</span>
                {i < path.length - 1 && <span className="text-slate-300 dark:text-slate-600">→</span>}
              </span>
            ))}
          </div>
        )}

        {/* ── Question ── */}
        {!isResult && (
          <div className="space-y-4">
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-5 border border-slate-200 dark:border-slate-700">
              <div className="flex items-start gap-3">
                <span className="bg-amber-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">
                  Q{(current as number) + 1}
                </span>
                <div>
                  <p className="font-semibold text-slate-800 dark:text-slate-100 text-sm leading-relaxed">
                    {QUESTIONS[current as number].text}
                  </p>
                  <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                    💡 {QUESTIONS[current as number].hint}
                  </p>
                </div>
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={() => answer(true)}
                className="flex-1 py-3 rounded-xl bg-green-500 hover:bg-green-600 text-white font-semibold text-sm transition-all active:scale-95 shadow-sm">
                ✓ 是
              </button>
              <button onClick={() => answer(false)}
                className="flex-1 py-3 rounded-xl bg-rose-500 hover:bg-rose-600 text-white font-semibold text-sm transition-all active:scale-95 shadow-sm">
                ✗ 否
              </button>
            </div>
          </div>
        )}

        {/* ── Result ── */}
        {isResult && result && (
          <div className={`rounded-xl border ${result.border} ${result.bg} p-5 space-y-4`}>
            <div className="flex items-center gap-3">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${result.gradient} flex items-center justify-center text-2xl shadow`}>
                {result.icon}
              </div>
              <div>
                <p className="text-xs text-slate-500 dark:text-slate-400">推荐算法</p>
                <h4 className={`text-xl font-black ${result.textColor}`}>{result.name}</h4>
                <p className="text-xs text-slate-400 dark:text-slate-500">{result.en}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white/60 dark:bg-black/20 rounded-lg p-3">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-0.5">时间复杂度</p>
                <p className={`text-sm font-semibold ${result.textColor}`}>{result.time}</p>
              </div>
              <div className="bg-white/60 dark:bg-black/20 rounded-lg p-3">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-0.5">空间复杂度</p>
                <p className={`text-sm font-semibold ${result.textColor}`}>{result.space}</p>
              </div>
            </div>

            <div className="bg-white/60 dark:bg-black/20 rounded-lg p-3">
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">典型题目</p>
              <div className="flex flex-wrap gap-1.5">
                {result.examples.map(ex => (
                  <span key={ex} className={`text-xs px-2 py-0.5 rounded-full border ${result.border} ${result.textColor} font-medium`}>{ex}</span>
                ))}
              </div>
            </div>

            <div className={`text-xs ${result.textColor} border-l-2 border-current pl-3 leading-relaxed`}>
              ⚠️ {result.tip}
            </div>

            <button onClick={reset}
              className="w-full py-2.5 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
              ↺ 重新判断
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

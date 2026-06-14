'use client'

import React, { useState } from 'react'

type Verdict = 'greedy-ok' | 'dp-needed'

interface TraceRow { step: string; action: string; state: string; note?: string }

interface Problem {
  id: string
  title: string
  verdict: Verdict
  instance: string
  greedyStrategy: string
  greedyTrace: TraceRow[]
  greedyResult: string
  optResult: string
  optTrace: TraceRow[]
  explanation: string
  complexity: string
}

const PROBLEMS: Problem[] = [
  {
    id: 'activity',
    title: '活动选择',
    verdict: 'greedy-ok',
    instance: '活动：a₁[1,4) a₂[3,5) a₃[0,6) a₄[5,7) a₅[3,9)\n目标：最多不重叠活动数',
    greedyStrategy: '按结束时间升序，每次选最早结束且与已选不冲突的活动',
    greedyTrace: [
      { step: '1', action: '排序后：a₁(f=4) a₂(f=5) a₃(f=6) a₄(f=7) a₅(f=9)', state: 'lastEnd=0', },
      { step: '2', action: '检查 a₁[1,4)：start=1 ≥ lastEnd=0 → 选中', state: 'lastEnd=4 | 已选: {a₁}', note: '✅ 选' },
      { step: '3', action: '检查 a₂[3,5)：start=3 < lastEnd=4 → 冲突', state: 'lastEnd=4 | 已选: {a₁}', note: '❌ 跳' },
      { step: '4', action: '检查 a₃[0,6)：start=0 < lastEnd=4 → 冲突', state: 'lastEnd=4 | 已选: {a₁}', note: '❌ 跳' },
      { step: '5', action: '检查 a₄[5,7)：start=5 ≥ lastEnd=4 → 选中', state: 'lastEnd=7 | 已选: {a₁,a₄}', note: '✅ 选' },
      { step: '6', action: '检查 a₅[3,9)：start=3 < lastEnd=7 → 冲突', state: 'lastEnd=7 | 已选: {a₁,a₄}', note: '❌ 跳' },
    ],
    greedyResult: '最优解：{a₁, a₄}，共 2 个活动',
    optTrace: [
      { step: 'OPT', action: '已知最优解也为 2 个活动', state: '贪心 = 最优', note: '✅' },
    ],
    optResult: '最优解：{a₁, a₄}，共 2 个 ← 与贪心一致',
    explanation: '贪心最优性：选结束最早的活动永远"不坏"（交换论证可证）。每次选最早结束的活动，剩余时间最多，可给后续活动留最大余地。最优子结构 + 贪心选择性质均满足。',
    complexity: 'O(n log n) 排序 + O(n) 扫描',
  },
  {
    id: 'knapsack01',
    title: '0/1 背包',
    verdict: 'dp-needed',
    instance: '物品：A(v=60,w=10) B(v=100,w=20) C(v=120,w=30)\n背包容量 W=50，每件物品只能取 0 或 1 件',
    greedyStrategy: '按单位价值 v/w 降序贪心选取（A:6.0 > B:5.0 > C:4.0）',
    greedyTrace: [
      { step: '1', action: '按密度排序: A(6.0) B(5.0) C(4.0)', state: '剩余容量=50, 总价值=0' },
      { step: '2', action: '选 A(w=10, v=60)，装得下', state: '剩余=40, 价值=60', note: '✅ 选' },
      { step: '3', action: '选 B(w=20, v=100)，装得下', state: '剩余=20, 价值=160', note: '✅ 选' },
      { step: '4', action: '尝试 C(w=30)：30 > 剩余=20 → 放不下', state: '剩余=20, 价值=160', note: '❌ 跳（只能整件）' },
    ],
    greedyResult: '贪心结果：选 A+B，总价值 = 160',
    optTrace: [
      { step: 'DP', action: 'DP 枚举子集：{B,C}(w=50, v=220) vs {A,B}(w=30, v=160) vs {A,C}(w=40, v=180)...', state: '', },
      { step: 'OPT', action: '最优：选 B(w=20) + C(w=30) = 50 = W', state: '价值 = 100+120 = 220', note: '✅ 最优' },
    ],
    optResult: '最优解：{B, C}，总价值 = 220 > 贪心的 160',
    explanation: '贪心失败原因：0/1 背包中物品不可分割。选了"密度最高"的 A 后，剩余空间无法容纳高价值的 C（需要 30 但只剩 20）。正确做法是 DP，对所有可行子集建表求最优。时间 O(nW)。',
    complexity: 'DP: O(nW)，贪心次优（此例差 38%）',
  },
  {
    id: 'fractional',
    title: '分数背包',
    verdict: 'greedy-ok',
    instance: '物品：A(v=60,w=10) B(v=100,w=20) C(v=120,w=30)\n背包容量 W=50，物品可按任意比例分割',
    greedyStrategy: '按单位价值 v/w 降序贪心选取，不够时取剩余比例',
    greedyTrace: [
      { step: '1', action: '按密度排序: A(6.0) B(5.0) C(4.0)', state: '剩余=50, 价值=0' },
      { step: '2', action: '取全部 A(w=10, v=60)', state: '剩余=40, 价值=60', note: '✅ 全取' },
      { step: '3', action: '取全部 B(w=20, v=100)', state: '剩余=20, 价值=160', note: '✅ 全取' },
      { step: '4', action: '取 C 的 20/30=2/3（剩余=20，C需30）', state: '剩余=0, 价值=160+80=240', note: '✅ 取 66.7%' },
    ],
    greedyResult: '贪心结果：A(全) + B(全) + C(2/3)，总价值 = 240',
    optTrace: [
      { step: 'OPT', action: '可证明：贪心 = 最优（线性规划松弛的整数解即贪心解）', state: '总价值 = 240', note: '✅' },
    ],
    optResult: '最优解 = 贪心解 = 240，两者一致',
    explanation: '分数背包 vs 0/1 背包：分数背包允许任意分割，所以高密度物品取到满再取次高密度物品，不存在"放不下"的窘境。可证明：前 k 件全取完、第 k+1 件部分取 的策略就是 LP 松弛的最优解，即全局最优。',
    complexity: 'O(n log n) 排序，O(n) 贪心扫描',
  },
  {
    id: 'coin',
    title: '找零（任意币值）',
    verdict: 'dp-needed',
    instance: '硬币面值：[1, 3, 4]，目标金额：6\n目标：使用最少硬币数凑出 6',
    greedyStrategy: '每次选不超过剩余金额的最大面值硬币',
    greedyTrace: [
      { step: '1', action: '剩余=6，可用最大: 4 → 选 4', state: '剩余=2, 已用=[4]', note: '选 4' },
      { step: '2', action: '剩余=2，可用最大: 1 → 选 1', state: '剩余=1, 已用=[4,1]', note: '选 1' },
      { step: '3', action: '剩余=1，可用最大: 1 → 选 1', state: '剩余=0, 已用=[4,1,1]', note: '选 1' },
    ],
    greedyResult: '贪心结果：4+1+1 = 3 枚硬币（次优！）',
    optTrace: [
      { step: 'DP', action: 'dp[6] = min(dp[5]+1, dp[3]+1, dp[2]+1) = min(2,1,3) = 2', state: '回溯: 6 = 3+3', note: '' },
      { step: 'OPT', action: '最优：选 3+3', state: '2 枚，总=6', note: '✅ 最优' },
    ],
    optResult: '最优解：3+3 = 2 枚，贪心多用 1 枚',
    explanation: '贪心失败：选了面值 4 之后，剩余 2 只能用两个 1，共 3 枚。而"跳过"4 直接用 3+3 只需 2 枚。当硬币面值不满足"正则性条件"（即大面值不能被小面值线性等价替换）时，贪心没有最优子结构，必须用 DP 枚举所有子问题。',
    complexity: 'DP: O(n × amount)，贪心可能次优或无解',
  },
]

const VERDICT_META: Record<Verdict, { label: string; cls: string; headerCls: string }> = {
  'greedy-ok':  { label: '✅ 贪心最优', cls: 'border-emerald-300 dark:border-emerald-700', headerCls: 'bg-emerald-600' },
  'dp-needed':  { label: '⚠️ 需要 DP',  cls: 'border-rose-300 dark:border-rose-700',    headerCls: 'bg-rose-600' },
}

function TraceTable({ rows, label, color }: { rows: TraceRow[]; label: string; color: 'blue' | 'emerald' }) {
  const cls = color === 'emerald'
    ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/20'
    : 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/20'
  const hdr = color === 'emerald' ? 'text-emerald-700 dark:text-emerald-300' : 'text-blue-700 dark:text-blue-300'
  return (
    <div className={`rounded-xl border p-3 ${cls}`}>
      <div className={`text-xs font-bold uppercase tracking-wider mb-2 ${hdr}`}>{label}</div>
      <div className="space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="flex gap-2 text-xs">
            <span className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px] ${color === 'emerald' ? 'bg-emerald-200 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200' : 'bg-blue-200 dark:bg-blue-900 text-blue-800 dark:text-blue-200'}`}>{r.step}</span>
            <div className="flex-1">
              <div className="font-medium text-slate-700 dark:text-slate-200">{r.action}</div>
              {r.state && <div className="text-slate-500 dark:text-slate-400 font-mono text-[10px]">{r.state}</div>}
            </div>
            {r.note && <span className="shrink-0 text-[11px] font-bold text-slate-500 dark:text-slate-400">{r.note}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function GreedyVsDPDecisionMap() {
  const [selected, setSelected] = useState(0)
  const p = PROBLEMS[selected]
  const meta = VERDICT_META[p.verdict]

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">贪心 vs DP 对比实验室 — 四道经典题深度解析</h3>
        <p className="text-blue-100 text-sm mt-0.5">切换题目，对比贪心策略 vs 真实最优解，理解贪心适用条件</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Problem tabs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {PROBLEMS.map((prob, i) => {
            const m = VERDICT_META[prob.verdict]
            return (
              <button key={prob.id} onClick={() => setSelected(i)} className={`px-3 py-2 rounded-xl text-xs font-bold text-left border-2 transition-all ${i === selected ? `${m.headerCls} text-white border-transparent` : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:border-slate-300 dark:hover:border-slate-600'}`}>
                <div>{prob.title}</div>
                <div className={`text-[10px] mt-0.5 ${i === selected ? 'text-white/80' : prob.verdict === 'greedy-ok' ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>{VERDICT_META[prob.verdict].label}</div>
              </button>
            )
          })}
        </div>

        {/* Verdict badge */}
        <div className={`rounded-xl border-2 ${meta.cls} px-4 py-2 flex items-center gap-3`}>
          <span className={`px-2 py-0.5 rounded-lg text-white text-xs font-bold ${meta.headerCls}`}>{meta.label}</span>
          <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{p.title}</span>
        </div>

        {/* Problem instance */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">题目实例</div>
          <pre className="text-sm text-slate-700 dark:text-slate-200 whitespace-pre-wrap font-mono leading-relaxed">{p.instance}</pre>
          <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            <span className="font-bold">贪心策略：</span>{p.greedyStrategy}
          </div>
        </div>

        {/* Traces side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-2">
            <TraceTable rows={p.greedyTrace} label="贪心执行过程" color="blue"/>
            <div className={`rounded-xl border px-3 py-2 text-sm font-bold ${p.verdict === 'greedy-ok' ? 'border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300' : 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300'}`}>
              {p.greedyResult}
            </div>
          </div>
          <div className="space-y-2">
            <TraceTable rows={p.optTrace} label="真实最优解（DP / 枚举）" color="emerald"/>
            <div className="rounded-xl border border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30 px-3 py-2 text-sm font-bold text-emerald-700 dark:text-emerald-300">
              {p.optResult}
            </div>
          </div>
        </div>

        {/* Explanation */}
        <div className={`rounded-xl border px-4 py-3 ${p.verdict === 'greedy-ok' ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/20' : 'border-rose-200 dark:border-rose-800 bg-rose-50 dark:bg-rose-950/20'}`}>
          <div className={`text-xs font-bold uppercase tracking-wider mb-1.5 ${p.verdict === 'greedy-ok' ? 'text-emerald-700 dark:text-emerald-300' : 'text-rose-700 dark:text-rose-300'}`}>
            {p.verdict === 'greedy-ok' ? '为什么贪心正确？' : '为什么贪心失败？'}
          </div>
          <div className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{p.explanation}</div>
          <div className="mt-2 text-xs text-slate-500 dark:text-slate-400"><span className="font-bold">复杂度：</span>{p.complexity}</div>
        </div>

        {/* Decision criteria summary */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 px-4 py-3">
          <div className="text-xs font-bold text-slate-700 dark:text-slate-200 mb-2">判断是否使用贪心的三个必要条件</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-slate-600 dark:text-slate-300">
            <div className="rounded-lg bg-white dark:bg-slate-900 p-2 border border-slate-200 dark:border-slate-700">
              <div className="font-bold text-indigo-600 dark:text-indigo-400">① 贪心选择性质</div>
              <div className="mt-0.5">局部最优选择不影响全局最优可行性（可通过交换论证证明）</div>
            </div>
            <div className="rounded-lg bg-white dark:bg-slate-900 p-2 border border-slate-200 dark:border-slate-700">
              <div className="font-bold text-indigo-600 dark:text-indigo-400">② 最优子结构</div>
              <div className="mt-0.5">最优解包含子问题的最优解；贪心每步缩小问题规模</div>
            </div>
            <div className="rounded-lg bg-white dark:bg-slate-900 p-2 border border-slate-200 dark:border-slate-700">
              <div className="font-bold text-indigo-600 dark:text-indigo-400">③ 无后效性</div>
              <div className="mt-0.5">当前贪心选择不影响未来选择的可行集合（或只扩大余地）</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

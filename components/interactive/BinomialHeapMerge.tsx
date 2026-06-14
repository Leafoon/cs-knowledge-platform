'use client'

import { useState } from 'react'

// ─── 数据定义 ──────────────────────────────────────────────────
interface BinomialTree {
  degree: number   // B_k
  minKey: number   // 该树最小根的 key（简化显示）
  size: number     // 2^degree
}

interface MergeScenario {
  label: string
  desc: string
  h1: BinomialTree[]  // 第一个堆
  h2: BinomialTree[]  // 第二个堆
}

const SCENARIOS: MergeScenario[] = [
  {
    label: '基本合并',
    desc: 'H1 = n=5（101₂：B₂+B₀），H2 = n=3（011₂：B₁+B₀）→ 合并后 n=8（1000₂：B₃）',
    h1: [{ degree: 2, minKey: 3, size: 4 }, { degree: 0, minKey: 11, size: 1 }],
    h2: [{ degree: 1, minKey: 5, size: 2 }, { degree: 0, minKey: 8, size: 1 }],
  },
  {
    label: '进位链',
    desc: 'H1 = n=7（111₂），H2 = n=7（111₂）→ n=14（1110₂），进位传播两次',
    h1: [{ degree: 2, minKey: 1, size: 4 }, { degree: 1, minKey: 7, size: 2 }, { degree: 0, minKey: 12, size: 1 }],
    h2: [{ degree: 2, minKey: 4, size: 4 }, { degree: 1, minKey: 9, size: 2 }, { degree: 0, minKey: 15, size: 1 }],
  },
  {
    label: '无冲突合并',
    desc: 'H1 = n=4（100₂：B₂），H2 = n=2（010₂：B₁）→ n=6（110₂），无需进位',
    h1: [{ degree: 2, minKey: 2, size: 4 }],
    h2: [{ degree: 1, minKey: 6, size: 2 }],
  },
]

// ─── 生成合并步骤 ──────────────────────────────────────────────
interface MergeStep {
  title: string
  desc: string
  bits1: (BinomialTree | null)[]   // H1 中各 degree 的树
  bits2: (BinomialTree | null)[]   // H2 中各 degree 的树
  carry: (BinomialTree | null)[]   // 进位桶
  result: (BinomialTree | null)[]  // 当前结果
  activeDeg: number
  phase: 'init' | 'process' | 'done'
}

function getBit(trees: BinomialTree[], deg: number): BinomialTree | null {
  return trees.find(t => t.degree === deg) ?? null
}

function mergeTwo(a: BinomialTree, b: BinomialTree): BinomialTree {
  return { degree: a.degree + 1, minKey: Math.min(a.minKey, b.minKey), size: a.size + b.size }
}

function generateMergeSteps(scenario: MergeScenario): MergeStep[] {
  const { h1, h2 } = scenario
  const MAX_DEG = 5
  const steps: MergeStep[] = []

  const carry: (BinomialTree | null)[] = new Array(MAX_DEG + 1).fill(null)
  const result: (BinomialTree | null)[] = new Array(MAX_DEG + 1).fill(null)

  steps.push({
    title: '初始状态：两个二项堆',
    desc: '上方：H1 的根列表（按度数升序）；下方：H2 的根列表。类比两个二进制数，准备"相加"。',
    bits1: Array.from({ length: MAX_DEG }, (_, d) => getBit(h1, d)),
    bits2: Array.from({ length: MAX_DEG }, (_, d) => getBit(h2, d)),
    carry: [...carry],
    result: [...result],
    activeDeg: -1,
    phase: 'init',
  })

  for (let d = 0; d <= MAX_DEG; d++) {
    const b1 = getBit(h1, d)
    const b2 = getBit(h2, d)
    const c = carry[d]

    const bits = [b1, b2, c].filter(Boolean) as BinomialTree[]

    if (bits.length === 0) {
      result[d] = null
    } else if (bits.length === 1) {
      result[d] = bits[0]
      steps.push({
        title: `degree=${d}：仅一棵树`,
        desc: `A[${d}] 只有 ${bits[0].minKey} 的树（${b1 ? 'H1' : b2 ? 'H2' : '进位'}），直接放入结果，无进位。`,
        bits1: Array.from({ length: MAX_DEG }, (_, i) => getBit(h1, i)),
        bits2: Array.from({ length: MAX_DEG }, (_, i) => getBit(h2, i)),
        carry: [...carry],
        result: [...result],
        activeDeg: d,
        phase: 'process',
      })
    } else if (bits.length === 2) {
      carry[d + 1] = mergeTwo(bits[0], bits[1])
      result[d] = null
      steps.push({
        title: `degree=${d}：两棵同度树 → LINK，进位到 d=${d + 1}`,
        desc: `key=${bits[0].minKey} 和 key=${bits[1].minKey} 合并：较小根保留，较大根成为子节点。新 B_${d + 1}（min=${Math.min(bits[0].minKey, bits[1].minKey)}）作为进位传入。`,
        bits1: Array.from({ length: MAX_DEG }, (_, i) => getBit(h1, i)),
        bits2: Array.from({ length: MAX_DEG }, (_, i) => getBit(h2, i)),
        carry: [...carry],
        result: [...result],
        activeDeg: d,
        phase: 'process',
      })
    } else {
      // 3棵：保留1棵，进位1棵
      carry[d + 1] = mergeTwo(bits[1], bits[2])
      result[d] = bits[0]
      steps.push({
        title: `degree=${d}：三棵同度树 → 保留1棵，LINK 另两棵进位`,
        desc: `三棵 B_${d} 树（三重进位）：key=${bits.map(b => b.minKey).join(', ')}。保留 key=${bits[0].minKey}，合并 key=${bits[1].minKey} 和 key=${bits[2].minKey} 为 B_${d + 1} 进位。`,
        bits1: Array.from({ length: MAX_DEG }, (_, i) => getBit(h1, i)),
        bits2: Array.from({ length: MAX_DEG }, (_, i) => getBit(h2, i)),
        carry: [...carry],
        result: [...result],
        activeDeg: d,
        phase: 'process',
      })
    }
  }

  const finalTrees = result.filter(Boolean) as BinomialTree[]
  steps.push({
    title: '合并完成！',
    desc: `合并后共 ${finalTrees.length} 棵树，总大小 n=${h1.reduce((s, t) => s + t.size, 0) + h2.reduce((s, t) => s + t.size, 0)}。时间复杂度：O(log n) 严格（最多 log n 个度数，每个度数 O(1)）。`,
    bits1: Array.from({ length: MAX_DEG }, (_, d) => getBit(h1, d)),
    bits2: Array.from({ length: MAX_DEG }, (_, d) => getBit(h2, d)),
    carry: [...carry],
    result: [...result],
    activeDeg: -1,
    phase: 'done',
  })

  return steps
}

// ─── 树形可视化（简化） ────────────────────────────────────────
function BinTreeIcon({ degree, minKey, size, highlighted, dimmed }: {
  degree: number; minKey: number; size: number
  highlighted?: boolean; dimmed?: boolean
}) {
  const rows = degree + 1
  return (
    <div className={`flex flex-col items-center gap-1 transition-all duration-300 ${dimmed ? 'opacity-25' : ''}`}>
      {/* 简化的树形：用点阵表示层级 */}
      <div className={`
        px-3 py-2 rounded-xl border-2 flex flex-col items-center gap-1
        ${highlighted
          ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 shadow-md shadow-emerald-200 dark:shadow-emerald-900/50'
          : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700'}
      `}>
        {Array.from({ length: Math.min(rows, 4) }, (_, r) => (
          <div key={r} className="flex gap-1">
            {Array.from({ length: Math.min(Math.pow(2, r), 8) }, (_, c) => (
              <div key={c} className={`rounded-full transition-colors ${
                r === 0
                  ? (highlighted ? 'w-4 h-4 bg-emerald-500' : 'w-4 h-4 bg-violet-500 dark:bg-violet-400')
                  : (highlighted ? 'w-2.5 h-2.5 bg-emerald-300' : 'w-2.5 h-2.5 bg-slate-300 dark:bg-slate-500')
              }`} />
            ))}
          </div>
        ))}
        {rows > 4 && <div className="text-xs text-slate-400">...</div>}
      </div>
      <div className={`text-center text-xs font-bold ${highlighted ? 'text-emerald-600 dark:text-emerald-400' : 'text-violet-600 dark:text-violet-400'}`}>
        B_{degree}
      </div>
      <div className="text-center text-xs text-slate-500 dark:text-slate-400">
        min={minKey}
      </div>
      <div className="text-center text-[10px] text-slate-400 dark:text-slate-500">
        ({size} 节点)
      </div>
    </div>
  )
}

// ─── 二进制表示格 ──────────────────────────────────────────────
function BitRow({ label, trees, maxDeg, activeDeg, color }: {
  label: string
  trees: (BinomialTree | null)[]
  maxDeg: number
  activeDeg: number
  color: string
}) {
  const totalSize = trees.filter(Boolean).reduce((s, t) => s + t!.size, 0)
  const binStr = Array.from({ length: maxDeg }, (_, d) => trees[d] ? '1' : '0').reverse().join('')

  return (
    <div className="flex items-center gap-3">
      <div className={`w-14 text-right text-xs font-bold ${color} shrink-0`}>{label}</div>
      <div className="flex gap-1.5 flex-row-reverse">
        {Array.from({ length: maxDeg }, (_, d) => (
          <div key={d} className={`
            w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold border-2 transition-all duration-300
            ${d === activeDeg ? 'scale-110 shadow-md' : ''}
            ${trees[d]
              ? (d === activeDeg
                  ? 'bg-amber-400 border-amber-500 text-slate-900'
                  : 'bg-violet-100 dark:bg-violet-900/40 border-violet-400 dark:border-violet-600 text-violet-700 dark:text-violet-300')
              : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-300 dark:text-slate-700'}
          `}>
            {trees[d] ? '1' : '0'}
          </div>
        ))}
      </div>
      <div className="text-xs font-mono text-slate-400 dark:text-slate-500 shrink-0">
        = {binStr}₂ ({totalSize})
      </div>
    </div>
  )
}

export function BinomialHeapMerge() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const scenario = SCENARIOS[scenarioIdx]
  const steps = generateMergeSteps(scenario)
  const safeStep = Math.min(stepIdx, steps.length - 1)
  const step = steps[safeStep]

  const changeScenario = (i: number) => { setScenarioIdx(i); setStepIdx(0) }

  const MAX_DEG = 5
  const activeDeg = step.activeDeg

  // 结果树
  const resultTrees = step.result.filter(Boolean) as BinomialTree[]
  const totalN = scenario.h1.reduce((s, t) => s + t.size, 0) + scenario.h2.reduce((s, t) => s + t.size, 0)
  const finalBin = Array.from({ length: MAX_DEG }, (_, d) => step.result[d] ? '1' : '0').reverse().join('')

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部：明亮清爽风格 */}
      <div className="bg-gradient-to-br from-indigo-500 via-blue-600 to-cyan-500 px-5 py-4">
        <h3 className="text-white font-bold text-base">🔢 二项堆 UNION：二进制加法类比</h3>
        <p className="text-blue-100 text-xs mt-0.5">两个二项堆的合并过程，完全类比二进制整数相加的进位规则</p>

        <div className="flex gap-2 mt-3 flex-wrap">
          {SCENARIOS.map((s, i) => (
            <button key={i} onClick={() => changeScenario(i)}
              className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all ${
                scenarioIdx === i ? 'bg-white text-blue-700 shadow-sm' : 'bg-white/25 text-white hover:bg-white/35'}`}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* 场景描述 */}
        <div className="text-xs text-slate-600 dark:text-slate-400 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg px-3 py-2">
          {scenario.desc}
        </div>

        {/* 二进制位表示 */}
        <div className="space-y-2">
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">bit 位表示（高位在左，degree↓）</div>
          
          {/* 列标 */}
          <div className="flex items-center gap-3">
            <div className="w-14" />
            <div className="flex gap-1.5 flex-row-reverse">
              {Array.from({ length: MAX_DEG }, (_, d) => (
                <div key={d} className={`w-10 text-center text-[10px] font-mono font-bold transition-colors ${
                  d === activeDeg ? 'text-amber-500 dark:text-amber-400' : 'text-slate-400 dark:text-slate-500'}`}>
                  B_{d}
                </div>
              ))}
            </div>
          </div>

          <BitRow label="H1" trees={step.bits1} maxDeg={MAX_DEG} activeDeg={activeDeg} color="text-violet-600 dark:text-violet-400" />
          <BitRow label="H2" trees={step.bits2} maxDeg={MAX_DEG} activeDeg={activeDeg} color="text-blue-600 dark:text-blue-400" />
          
          {/* 分割线 */}
          <div className="flex items-center gap-3">
            <div className="w-14 text-right text-xs text-slate-400">进位</div>
            <div className="flex gap-1.5 flex-row-reverse">
              {Array.from({ length: MAX_DEG }, (_, d) => (
                <div key={d} className={`
                  w-10 h-8 rounded flex items-center justify-center text-xs font-bold transition-all duration-300
                  ${step.carry[d] ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-600' : 'text-slate-200 dark:text-slate-700'}
                `}>
                  {step.carry[d] ? '↑' : '·'}
                </div>
              ))}
            </div>
          </div>

          {/* 结果 */}
          <div className="flex items-center gap-3 border-t-2 border-slate-300 dark:border-slate-600 pt-2 mt-2">
            <div className="w-14 text-right text-xs font-bold text-emerald-600 dark:text-emerald-400">结果</div>
            <div className="flex gap-1.5 flex-row-reverse">
              {Array.from({ length: MAX_DEG }, (_, d) => (
                <div key={d} className={`
                  w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold border-2 transition-all duration-500
                  ${step.result[d]
                    ? 'bg-emerald-400 dark:bg-emerald-500 border-emerald-500 dark:border-emerald-400 text-white shadow-sm'
                    : 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-300 dark:text-slate-700'}
                `}>
                  {step.result[d] ? '1' : '0'}
                </div>
              ))}
            </div>
            <div className="text-xs font-mono text-emerald-600 dark:text-emerald-400 shrink-0 font-bold">
              = {finalBin}₂ ({totalN})
            </div>
          </div>
        </div>

        {/* 当前步骤说明 */}
        <div className={`rounded-xl p-4 border transition-all duration-300 ${
          step.phase === 'done'
            ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700'
            : step.phase === 'process' && activeDeg >= 0
              ? 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-700'
              : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700'
        }`}>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-base">{step.phase === 'done' ? '✅' : step.phase === 'init' ? '📖' : '⚙️'}</span>
            <span className={`text-sm font-bold ${
              step.phase === 'done' ? 'text-emerald-700 dark:text-emerald-400' : 'text-slate-800 dark:text-slate-100'}`}>
              {step.title}
            </span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">{step.desc}</p>
        </div>

        {/* 结果树形展示（最终步骤） */}
        {step.phase === 'done' && resultTrees.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
              合并结果（{resultTrees.length} 棵树）
            </div>
            <div className="flex gap-4 flex-wrap justify-center">
              {resultTrees.map(tree => (
                <BinTreeIcon key={tree.degree} {...tree} highlighted />
              ))}
            </div>
          </div>
        )}

        {/* 控制按钮 */}
        <div className="flex items-center justify-between">
          <button onClick={() => setStepIdx(0)} disabled={safeStep === 0}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ⏮ 重置
          </button>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 dark:text-slate-500">{safeStep + 1}/{steps.length}</span>
            <div className="flex gap-1">
              {steps.map((_, i) => (
                <div key={i} className={`w-2 h-2 rounded-full transition-all duration-300 cursor-pointer ${
                  i === safeStep ? 'bg-blue-500 w-4' : i < safeStep ? 'bg-blue-300 dark:bg-blue-700' : 'bg-slate-200 dark:bg-slate-700'}`}
                  onClick={() => setStepIdx(i)}
                />
              ))}
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={() => setStepIdx(i => Math.max(0, i - 1))} disabled={safeStep === 0}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              ← 上一步
            </button>
            <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))} disabled={safeStep === steps.length - 1}
              className="px-3 py-1.5 text-xs rounded-lg bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              下一步 →
            </button>
          </div>
        </div>

        {/* 底部信息 */}
        <div className="text-center text-xs text-slate-400 dark:text-slate-600 border-t border-slate-100 dark:border-slate-800 pt-3">
          UNION 时间：O(log n) 严格 —— 最多 log₂n 个度数，每个度数 O(1) LINK 操作
        </div>
      </div>
    </div>
  )
}

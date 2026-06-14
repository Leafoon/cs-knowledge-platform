'use client'

import { useState, useCallback } from 'react'

// ─── 类型定义 ──────────────────────────────────────────────────
interface HeapNode {
  id: string
  key: number
  degree: number
  children: HeapNode[]
  mark?: boolean
}

interface ConsolidateStep {
  title: string
  desc: string
  roots: HeapNode[]         // 当前根列表
  buckets: (HeapNode | null)[]  // 度数桶
  highlight: string[]       // 当前操作涉及的节点 id
  merging?: [string, string]    // 正在合并的两个节点 id
  phase: 'scan' | 'link' | 'rebuild' | 'done'
  currentDeg?: number
}

// ─── 预设场景 ─────────────────────────────────────────────────
const PRESETS = [
  {
    label: '场景一：4棵同度树',
    desc: '根列表包含 4 棵度数相同的树（B_0），体验完整进位链',
    initial: [
      { id: 'a', key: 3, degree: 0, children: [] },
      { id: 'b', key: 7, degree: 0, children: [] },
      { id: 'c', key: 1, degree: 0, children: [] },
      { id: 'd', key: 9, degree: 0, children: [] },
    ] as HeapNode[],
  },
  {
    label: '场景二：混合度数',
    desc: '根列表包含度数 0、1、0 的树，演示跳过已有树',
    initial: [
      { id: 'a', key: 5, degree: 0, children: [] },
      { id: 'b', key: 2, degree: 1, children: [{ id: 'b1', key: 8, degree: 0, children: [] }] },
      { id: 'c', key: 4, degree: 0, children: [] },
      { id: 'd', key: 6, degree: 0, children: [] },
    ] as HeapNode[],
  },
  {
    label: '场景三：3棵 B1 进位',
    desc: '两棵 B1 合并为 B2，再和另一棵 B1 产生进位',
    initial: [
      { id: 'a', key: 3, degree: 1, children: [{ id: 'a1', key: 10, degree: 0, children: [] }] },
      { id: 'b', key: 1, degree: 1, children: [{ id: 'b1', key: 7, degree: 0, children: [] }] },
      { id: 'c', key: 5, degree: 1, children: [{ id: 'c1', key: 9, degree: 0, children: [] }] },
    ] as HeapNode[],
  },
]

// ─── 核心逻辑：生成所有步骤 ───────────────────────────────────
function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj))
}

function linkNodes(y: HeapNode, x: HeapNode): HeapNode {
  // 将 y 合并到 x 下面（x.key <= y.key）
  const newX = deepClone(x)
  const newY = deepClone(y)
  newY.children = []
  newX.children = [newY, ...newX.children]
  newX.degree += 1
  return newX
}

function generateSteps(initialRoots: HeapNode[]): ConsolidateStep[] {
  const steps: ConsolidateStep[] = []
  const MAX_DEG = 6

  // 初始状态
  steps.push({
    title: '初始根列表',
    desc: `根列表共 ${initialRoots.length} 棵树。CONSOLIDATE 目标：使每个度数最多一棵树（类似二进制不重复位）。`,
    roots: deepClone(initialRoots),
    buckets: new Array(MAX_DEG).fill(null),
    highlight: [],
    phase: 'scan',
  })

  // 模拟 CONSOLIDATE
  const buckets: (HeapNode | null)[] = new Array(MAX_DEG).fill(null)
  let roots = deepClone(initialRoots)
  const processed: HeapNode[] = []

  for (let i = 0; i < initialRoots.length; i++) {
    let w = deepClone(roots[i])
    let d = w.degree

    steps.push({
      title: `扫描节点 key=${w.key}，degree=${d}`,
      desc: `检查度数桶 A[${d}]：${buckets[d] ? `已有 key=${buckets[d]!.key} 的节点，需要合并！` : '为空，直接放入。'}`,
      roots: deepClone(roots),
      buckets: deepClone(buckets),
      highlight: [w.id],
      phase: 'scan',
      currentDeg: d,
    })

    while (d < MAX_DEG && buckets[d] !== null) {
      let y = buckets[d]!
      if (w.key > y.key) {
        const tmp = w; w = y; y = tmp
      }
      // y 合并到 w
      steps.push({
        title: `合并 key=${y.key} → key=${w.key}（度数 ${d}）`,
        desc: `A[${d}] 冲突：key=${y.key} 的树合并到 key=${w.key} 下面，成为其子节点。合并后 key=${w.key} 的 degree = ${d + 1}，检查 A[${d + 1}]。`,
        roots: deepClone(roots),
        buckets: deepClone(buckets),
        highlight: [w.id, y.id],
        merging: [y.id, w.id],
        phase: 'link',
        currentDeg: d,
      })
      buckets[d] = null
      w = linkNodes(y, w)
      d += 1
    }

    if (d < MAX_DEG) {
      buckets[d] = deepClone(w)
      steps.push({
        title: `key=${w.key} 放入 A[${d}]`,
        desc: `A[${d}] 为空，将 degree=${d} 的树（key=${w.key}）放入桶中。`,
        roots: deepClone(roots),
        buckets: deepClone(buckets),
        highlight: [w.id],
        phase: 'scan',
        currentDeg: d,
      })
    }
  }

  // 从桶重建根列表
  const finalRoots = buckets.filter(Boolean) as HeapNode[]
  const minRoot = finalRoots.reduce((m, r) => r.key < m.key ? r : m, finalRoots[0])

  steps.push({
    title: '重建根列表，更新 min 指针',
    desc: `从度数桶汇集所有树，组成新的根列表。共 ${finalRoots.length} 棵树，最小根 key=${minRoot?.key}（min 指针指向它）。CONSOLIDATE 完成！`,
    roots: finalRoots,
    buckets: deepClone(buckets),
    highlight: [minRoot?.id ?? ''],
    phase: 'done',
  })

  return steps
}

// ─── 树渲染组件 ───────────────────────────────────────────────
function TreeNode({ node, highlight, merging }: {
  node: HeapNode
  highlight: string[]
  merging?: [string, string]
}) {
  const isHL = highlight.includes(node.id)
  const isMerging = merging?.includes(node.id)

  return (
    <div className="flex flex-col items-center gap-1">
      <div className={`
        w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-all duration-300
        ${isMerging
          ? 'bg-amber-400 border-amber-600 dark:bg-amber-500 dark:border-amber-300 scale-110 shadow-lg shadow-amber-400/50'
          : isHL
            ? 'bg-violet-500 border-violet-700 dark:bg-violet-400 dark:border-violet-200 scale-105 text-white'
            : 'bg-white dark:bg-slate-700 border-slate-300 dark:border-slate-500 text-slate-800 dark:text-slate-100'}
      `}>
        {node.key}
      </div>
      {node.children.length > 0 && (
        <div className="flex gap-3 relative">
          {/* 连接线 */}
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-full h-px border-t border-dashed border-slate-300 dark:border-slate-600" />
          {node.children.map(child => (
            <div key={child.id} className="flex flex-col items-center pt-2">
              <div className="w-px h-2 bg-slate-300 dark:bg-slate-600" />
              <TreeNode node={child} highlight={highlight} merging={merging} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── 主组件 ───────────────────────────────────────────────────
export function FibHeapConsolidate() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const preset = PRESETS[presetIdx]
  const steps = generateSteps(preset.initial)
  const safeStep = Math.min(stepIdx, steps.length - 1)
  const step = steps[safeStep]

  const changePreset = (idx: number) => {
    setPresetIdx(idx)
    setStepIdx(0)
  }

  const MAX_DEG_SHOW = 5

  const phaseColor = {
    scan: 'text-blue-600 dark:text-blue-400',
    link: 'text-amber-600 dark:text-amber-400',
    rebuild: 'text-emerald-600 dark:text-emerald-400',
    done: 'text-emerald-600 dark:text-emerald-400',
  }

  const phaseIcon = {
    scan: '🔍',
    link: '🔗',
    rebuild: '🏗️',
    done: '✅',
  }

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部栏 */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-700 px-5 py-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h3 className="text-white font-bold text-base tracking-wide">⚙️ CONSOLIDATE 整合过程</h3>
            <p className="text-violet-200 text-xs mt-0.5">度数桶（Degree Buckets）合并同度树，类比二进制加法进位</p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => changePreset(i)}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all ${
                  presetIdx === i
                    ? 'bg-white text-purple-700 shadow-sm'
                    : 'bg-white/20 text-white hover:bg-white/30'}`}>
                {p.label.split('：')[0]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-4">
        {/* 场景说明 */}
        <div className="text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 rounded-lg px-3 py-2">
          {preset.desc}
        </div>

        {/* 步骤进度 */}
        <div className="flex items-center gap-3">
          <div className="flex-1 bg-slate-100 dark:bg-slate-800 rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-500"
              style={{ width: `${((safeStep + 1) / steps.length) * 100}%` }}
            />
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400 shrink-0">
            {safeStep + 1} / {steps.length}
          </span>
        </div>

        {/* 当前步骤说明 */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-base">{phaseIcon[step.phase]}</span>
            <span className={`font-semibold text-sm ${phaseColor[step.phase]}`}>{step.title}</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">{step.desc}</p>
        </div>

        {/* 根列表可视化 */}
        <div>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            根列表（Root List）
          </div>
          <div className="flex gap-6 flex-wrap items-end min-h-[80px] px-2 py-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-dashed border-slate-200 dark:border-slate-700">
            {step.roots.length === 0 ? (
              <span className="text-slate-400 dark:text-slate-600 text-xs">（空）</span>
            ) : (
              step.roots.map((root) => (
                <div key={root.id} className="flex flex-col items-center gap-1">
                  <div className="text-xs text-slate-400 dark:text-slate-500 mb-1">deg={root.degree}</div>
                  <TreeNode node={root} highlight={step.highlight} merging={step.merging} />
                </div>
              ))
            )}
          </div>
        </div>

        {/* 度数桶 */}
        <div>
          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            度数桶 A[0..{MAX_DEG_SHOW - 1}]
          </div>
          <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${MAX_DEG_SHOW}, 1fr)` }}>
            {Array.from({ length: MAX_DEG_SHOW }, (_, d) => {
              const node = step.buckets[d]
              const isActive = step.currentDeg === d
              return (
                <div key={d} className={`
                  rounded-xl border-2 transition-all duration-300 p-2 flex flex-col items-center gap-2 min-h-[80px] justify-center
                  ${isActive
                    ? 'border-violet-400 dark:border-violet-500 bg-violet-50 dark:bg-violet-900/20'
                    : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50'}
                `}>
                  <div className={`text-xs font-mono font-bold ${isActive ? 'text-violet-600 dark:text-violet-400' : 'text-slate-400 dark:text-slate-500'}`}>
                    A[{d}]
                  </div>
                  {node ? (
                    <TreeNode node={node} highlight={step.highlight} merging={step.merging} />
                  ) : (
                    <div className="text-slate-300 dark:text-slate-700 text-lg">∅</div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* 控制按钮 */}
        <div className="flex items-center justify-between pt-1">
          <button onClick={() => setStepIdx(0)}
            disabled={safeStep === 0}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ⏮ 重置
          </button>
          <div className="flex gap-2">
            <button onClick={() => setStepIdx(i => Math.max(0, i - 1))}
              disabled={safeStep === 0}
              className="px-4 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
              ← 上一步
            </button>
            <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))}
              disabled={safeStep === steps.length - 1}
              className="px-4 py-1.5 text-xs rounded-lg bg-violet-600 hover:bg-violet-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
              下一步 →
            </button>
          </div>
          <button onClick={() => setStepIdx(steps.length - 1)}
            disabled={safeStep === steps.length - 1}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ⏭ 末尾
          </button>
        </div>

        {/* 公式提示 */}
        <div className="text-xs text-slate-400 dark:text-slate-600 border-t border-slate-100 dark:border-slate-800 pt-3 text-center">
          CONSOLIDATE 后最多 D(n)+1 棵树，D(n) ≤ log_φ(n) ≈ 1.44 log₂n
        </div>
      </div>
    </div>
  )
}

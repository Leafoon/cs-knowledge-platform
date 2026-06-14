'use client'

import { useState } from 'react'

// ─── 场景数据 ──────────────────────────────────────────────────
// 每个场景定义一个完整的步骤序列
interface NodeState {
  id: string
  key: number
  mark: boolean
  parent: string | null
  children: string[]
  x: number   // 布局坐标（百分比位置用于 SVG）
  y: number
  isRoot?: boolean
  cutThisStep?: boolean
  highlighted?: boolean
  newKey?: number
  inRootList?: boolean
}

interface ScenarioStep {
  title: string
  subtitle: string
  nodes: NodeState[]
  edges: Array<{ from: string; to: string; dashed?: boolean; cut?: boolean }>
  rootList: string[]     // 根列表中的节点 id
  annotation?: string    // 底部注释
  potentialChange?: string
}

// ─── 场景一：简单 CUT（父节点未标记）──────────────────────────
const SCENARIO_1: ScenarioStep[] = [
  {
    title: '初始状态',
    subtitle: '节点 y（key=46）的 mark=false，x（key=35）是 y 的子节点',
    nodes: [
      { id: 'min', key: 3,  mark: false, parent: null, children: ['a'], x: 50, y: 8, isRoot: true },
      { id: 'a',   key: 17, mark: false, parent: 'min', children: ['b', 'c'], x: 50, y: 30 },
      { id: 'b',   key: 30, mark: false, parent: 'a',   children: ['d'], x: 30, y: 52 },
      { id: 'c',   key: 46, mark: false, parent: 'a',   children: ['e'], x: 70, y: 52 },  // y
      { id: 'd',   key: 52, mark: false, parent: 'b',   children: [], x: 30, y: 74 },
      { id: 'e',   key: 35, mark: false, parent: 'c',   children: [], x: 70, y: 74 },  // x (要 decrease)
    ],
    edges: [
      { from: 'min', to: 'a' }, { from: 'a', to: 'b' }, { from: 'a', to: 'c' },
      { from: 'b', to: 'd' }, { from: 'c', to: 'e' },
    ],
    rootList: ['min'],
    annotation: 'min 指向全局最小根（key=3）。即将对节点 key=35 执行 DECREASE-KEY(x, 15)',
  },
  {
    title: 'DECREASE-KEY：将 key=35 降至 15',
    subtitle: '新 key=15 < 父节点 key=46 → 违反堆性质，需要 CUT',
    nodes: [
      { id: 'min', key: 3,  mark: false, parent: null, children: ['a'], x: 50, y: 8, isRoot: true },
      { id: 'a',   key: 17, mark: false, parent: 'min', children: ['b', 'c'], x: 50, y: 30 },
      { id: 'b',   key: 30, mark: false, parent: 'a',   children: ['d'], x: 30, y: 52 },
      { id: 'c',   key: 46, mark: false, parent: 'a',   children: ['e'], x: 70, y: 52 },
      { id: 'd',   key: 52, mark: false, parent: 'b',   children: [], x: 30, y: 74 },
      { id: 'e',   key: 15, mark: false, parent: 'c',   children: [], x: 70, y: 74, highlighted: true, newKey: 15 },
    ],
    edges: [
      { from: 'min', to: 'a' }, { from: 'a', to: 'b' }, { from: 'a', to: 'c' },
      { from: 'b', to: 'd' }, { from: 'c', to: 'e', cut: true },
    ],
    rootList: ['min'],
    annotation: '检查：new_key=15 < parent.key=46 ✓  →  执行 CUT(e, c)',
    potentialChange: 'Φ 即将变化',
  },
  {
    title: 'CUT(e, c)：将 x=15 从父节点切断',
    subtitle: '节点 key=15 移入根列表；父节点 key=46 的 mark 从 false → true（首次失去孩子）',
    nodes: [
      { id: 'min', key: 3,  mark: false, parent: null, children: ['a'], x: 50, y: 8, isRoot: true },
      { id: 'a',   key: 17, mark: false, parent: 'min', children: ['b', 'c'], x: 50, y: 30 },
      { id: 'b',   key: 30, mark: false, parent: 'a',   children: ['d'], x: 30, y: 52 },
      { id: 'c',   key: 46, mark: true,  parent: 'a',   children: [], x: 70, y: 52, highlighted: true },  // mark 变 true
      { id: 'd',   key: 52, mark: false, parent: 'b',   children: [], x: 30, y: 74 },
      { id: 'e',   key: 15, mark: false, parent: null,  children: [], x: 70, y: 74, isRoot: true, inRootList: true, cutThisStep: true },
    ],
    edges: [
      { from: 'min', to: 'a' }, { from: 'a', to: 'b' }, { from: 'a', to: 'c' },
      { from: 'b', to: 'd' },
    ],
    rootList: ['min', 'e'],
    annotation: 'c.mark: false → true（第一次失去孩子，仅标记，不 CUT）。CASCADING-CUT(c) 遇到 mark=false，递归终止。',
    potentialChange: 'Δt(H)=+1, Δm(H)=+1 → ΔΦ=+3... 但下一个 CASCADING-CUT 将消耗',
  },
  {
    title: '完成！更新 min 指针',
    subtitle: '新节点 key=15 成为全局最小根，更新 min 指针',
    nodes: [
      { id: 'min', key: 3,  mark: false, parent: null, children: ['a'], x: 50, y: 8, isRoot: true },
      { id: 'a',   key: 17, mark: false, parent: 'min', children: ['b', 'c'], x: 50, y: 30 },
      { id: 'b',   key: 30, mark: false, parent: 'a',   children: ['d'], x: 30, y: 52 },
      { id: 'c',   key: 46, mark: true,  parent: 'a',   children: [], x: 70, y: 52 },
      { id: 'd',   key: 52, mark: false, parent: 'b',   children: [], x: 30, y: 74 },
      { id: 'e',   key: 15, mark: false, parent: null,  children: [], x: 70, y: 74, isRoot: true },
    ],
    edges: [
      { from: 'min', to: 'a' }, { from: 'a', to: 'b' }, { from: 'a', to: 'c' },
      { from: 'b', to: 'd' },
    ],
    rootList: ['min', 'e'],
    annotation: 'DECREASE-KEY 总摊销代价 O(1)：实际 CUT 1次（+1），Δm(H)=+1（+2），Δt(H)=+1（+1），合计 ΔΦ=+3，但 DECREASE-KEY 实际付出 O(1)，摊销包含在级联裁剪的势能预算中。',
  },
]

// ─── 场景二：级联裁剪链 ────────────────────────────────────────
const SCENARIO_2: ScenarioStep[] = [
  {
    title: '初始：含级联裁剪场景',
    subtitle: '节点 y1（key=24）和 y2（key=30）均已被标记（mark=true）',
    nodes: [
      { id: 'root', key: 7,  mark: false, parent: null, children: ['n1'], x: 50, y: 6, isRoot: true },
      { id: 'n1',   key: 18, mark: false, parent: 'root', children: ['n2', 'n3'], x: 50, y: 24 },
      { id: 'n2',   key: 24, mark: true,  parent: 'n1', children: ['n4'], x: 30, y: 44 },   // mark=true!
      { id: 'n3',   key: 38, mark: false, parent: 'n1', children: [], x: 70, y: 44 },
      { id: 'n4',   key: 30, mark: true,  parent: 'n2', children: ['n5'], x: 30, y: 63 },   // mark=true!
      { id: 'n5',   key: 41, mark: false, parent: 'n4', children: ['n6'], x: 30, y: 80 },
      { id: 'n6',   key: 55, mark: false, parent: 'n5', children: [], x: 30, y: 94 },  // x: 将被 decrease
    ],
    edges: [
      { from: 'root', to: 'n1' }, { from: 'n1', to: 'n2' }, { from: 'n1', to: 'n3' },
      { from: 'n2', to: 'n4' }, { from: 'n4', to: 'n5' }, { from: 'n5', to: 'n6' },
    ],
    rootList: ['root'],
    annotation: '深色节点表示 mark=true（已失去过一个子节点）。即将执行 DECREASE-KEY(n6, 5)',
  },
  {
    title: 'key=55 → 5：违反堆性质',
    subtitle: '新 key=5 < 父节点 key=41 → CUT(n6, n5)',
    nodes: [
      { id: 'root', key: 7,  mark: false, parent: null, children: ['n1'], x: 50, y: 6, isRoot: true },
      { id: 'n1',   key: 18, mark: false, parent: 'root', children: ['n2', 'n3'], x: 50, y: 24 },
      { id: 'n2',   key: 24, mark: true,  parent: 'n1', children: ['n4'], x: 30, y: 44 },
      { id: 'n3',   key: 38, mark: false, parent: 'n1', children: [], x: 70, y: 44 },
      { id: 'n4',   key: 30, mark: true,  parent: 'n2', children: ['n5'], x: 30, y: 63 },
      { id: 'n5',   key: 41, mark: false, parent: 'n4', children: ['n6'], x: 30, y: 80 },
      { id: 'n6',   key: 5,  mark: false, parent: 'n5', children: [], x: 30, y: 94, highlighted: true, newKey: 5 },
    ],
    edges: [
      { from: 'root', to: 'n1' }, { from: 'n1', to: 'n2' }, { from: 'n1', to: 'n3' },
      { from: 'n2', to: 'n4' }, { from: 'n4', to: 'n5' }, { from: 'n5', to: 'n6', cut: true },
    ],
    rootList: ['root'],
    annotation: 'CUT(n6, n5)：剪断 n6，加入根列表。然后调用 CASCADING-CUT(n5)',
  },
  {
    title: 'CUT(n6)；CASCADING-CUT(n5)：n5.mark=false → 标记终止',
    subtitle: 'n5.mark 原为 false，这是第一次失去孩子：只设 mark=true，不 CUT，递归终止',
    nodes: [
      { id: 'root', key: 7,  mark: false, parent: null, children: ['n1'], x: 50, y: 6, isRoot: true },
      { id: 'n1',   key: 18, mark: false, parent: 'root', children: ['n2', 'n3'], x: 50, y: 24 },
      { id: 'n2',   key: 24, mark: true,  parent: 'n1', children: ['n4'], x: 30, y: 44 },
      { id: 'n3',   key: 38, mark: false, parent: 'n1', children: [], x: 70, y: 44 },
      { id: 'n4',   key: 30, mark: true,  parent: 'n2', children: ['n5'], x: 30, y: 63 },
      { id: 'n5',   key: 41, mark: true,  parent: 'n4', children: [], x: 30, y: 80, highlighted: true },  // mark flipped!
      { id: 'n6',   key: 5,  mark: false, parent: null, children: [], x: 78, y: 94, isRoot: true, cutThisStep: true },
    ],
    edges: [
      { from: 'root', to: 'n1' }, { from: 'n1', to: 'n2' }, { from: 'n1', to: 'n3' },
      { from: 'n2', to: 'n4' }, { from: 'n4', to: 'n5' },
    ],
    rootList: ['root', 'n6'],
    annotation: 'n5.mark: false → true。递归终止（mark=false 说明第一次失去孩子，暂缓裁剪）。势能变化：Δt=+1，Δm=+1，ΔΦ=+3。',
  },
  {
    title: '完成！根列表新增 key=5，更新 min',
    subtitle: 'key=5 现在是全局最小，min 指针指向它',
    nodes: [
      { id: 'root', key: 7,  mark: false, parent: null, children: ['n1'], x: 50, y: 6, isRoot: true },
      { id: 'n1',   key: 18, mark: false, parent: 'root', children: ['n2', 'n3'], x: 50, y: 24 },
      { id: 'n2',   key: 24, mark: true,  parent: 'n1', children: ['n4'], x: 30, y: 44 },
      { id: 'n3',   key: 38, mark: false, parent: 'n1', children: [], x: 70, y: 44 },
      { id: 'n4',   key: 30, mark: true,  parent: 'n2', children: ['n5'], x: 30, y: 63 },
      { id: 'n5',   key: 41, mark: true,  parent: 'n4', children: [], x: 30, y: 80 },
      { id: 'n6',   key: 5,  mark: false, parent: null, children: [], x: 78, y: 80, isRoot: true, highlighted: true },
    ],
    edges: [
      { from: 'root', to: 'n1' }, { from: 'n1', to: 'n2' }, { from: 'n1', to: 'n3' },
      { from: 'n2', to: 'n4' }, { from: 'n4', to: 'n5' },
    ],
    rootList: ['root', 'n6'],
    annotation: 'DECREASE-KEY 完成。摊销代价 O(1)：每次 CUT 实际代价 O(1)，但 m(H) 同步减少，势能补偿使摊销代价保持常数。',
  },
]

const SCENARIOS = [
  { label: '简单 CUT（无级联）', steps: SCENARIO_1 },
  { label: '级联裁剪示例', steps: SCENARIO_2 },
]

// ─── 主组件 ───────────────────────────────────────────────────
export function FibHeapDecreaseKey() {
  const [scenarioIdx, setScenarioIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)

  const scenario = SCENARIOS[scenarioIdx]
  const steps = scenario.steps
  const safeStep = Math.min(stepIdx, steps.length - 1)
  const step = steps[safeStep]

  const changeScenario = (i: number) => { setScenarioIdx(i); setStepIdx(0) }

  const getNodeColor = (n: NodeState) => {
    if (n.cutThisStep) return { bg: 'fill-emerald-500 dark:fill-emerald-400', text: 'fill-white', stroke: 'stroke-emerald-600 dark:stroke-emerald-300' }
    if (n.highlighted) return { bg: 'fill-amber-400 dark:fill-amber-400', text: 'fill-slate-900', stroke: 'stroke-amber-600 dark:stroke-amber-300' }
    if (n.mark) return { bg: 'fill-rose-500 dark:fill-rose-600', text: 'fill-white', stroke: 'stroke-rose-700 dark:stroke-rose-400' }
    if (n.isRoot) return { bg: 'fill-violet-500 dark:fill-violet-600', text: 'fill-white', stroke: 'stroke-violet-700 dark:stroke-violet-400' }
    return { bg: 'fill-slate-200 dark:fill-slate-600', text: 'fill-slate-800 dark:fill-slate-100', stroke: 'stroke-slate-300 dark:stroke-slate-500' }
  }

  const nodeMap = Object.fromEntries(step.nodes.map(n => [n.id, n]))

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-xl font-sans">
      {/* 顶部：深色风格 */}
      <div className="bg-slate-900 dark:bg-black px-5 py-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h3 className="text-white font-bold text-base">🔪 DECREASE-KEY & 级联裁剪</h3>
            <p className="text-slate-400 text-xs mt-0.5">CUT + CASCADING-CUT 的触发路径与 mark 状态变化</p>
          </div>
          <div className="flex gap-2">
            {SCENARIOS.map((s, i) => (
              <button key={i} onClick={() => changeScenario(i)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${
                  scenarioIdx === i
                    ? 'bg-amber-400 text-slate-900'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900">
        {/* 图例 */}
        <div className="flex gap-4 flex-wrap px-5 pt-4 text-xs">
          {[
            { color: 'bg-violet-500', label: '根节点' },
            { color: 'bg-slate-300 dark:bg-slate-600', label: 'mark=false' },
            { color: 'bg-rose-500', label: 'mark=true' },
            { color: 'bg-amber-400', label: '当前操作节点' },
            { color: 'bg-emerald-500', label: '已裁剪→根列表' },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className={`w-3 h-3 rounded-full ${color}`} />
              <span className="text-slate-500 dark:text-slate-400">{label}</span>
            </div>
          ))}
        </div>

        <div className="flex gap-4 px-5 pb-5 pt-3 flex-col md:flex-row">
          {/* SVG 树形图 */}
          <div className="flex-1 min-w-0">
            <svg viewBox="0 0 340 330" className="w-full max-w-sm mx-auto" style={{ minHeight: 280 }}>
              {/* 边 */}
              {step.edges.map((e, i) => {
                const from = nodeMap[e.from]
                const to = nodeMap[e.to]
                if (!from || !to) return null
                const x1 = from.x * 3.4, y1 = from.y * 3.2 + 18
                const x2 = to.x * 3.4,   y2 = to.y * 3.2 + 18
                return (
                  <line key={i}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={e.cut ? '#f87171' : '#94a3b8'}
                    strokeWidth={e.cut ? 2.5 : 1.5}
                    strokeDasharray={e.cut ? '5,4' : undefined}
                    strokeLinecap="round"
                    opacity={e.cut ? 1 : 0.7}
                  />
                )
              })}

              {/* 节点 */}
              {step.nodes.map(n => {
                const colors = getNodeColor(n)
                const cx = n.x * 3.4
                const cy = n.y * 3.2 + 18
                return (
                  <g key={n.id}>
                    <circle cx={cx} cy={cy} r={16}
                      className={`${colors.bg} ${colors.stroke} transition-all duration-400`}
                      strokeWidth={2}
                    />
                    {n.mark && (
                      <circle cx={cx + 11} cy={cy - 11} r={5}
                        className="fill-amber-400 stroke-amber-600"
                        strokeWidth={1.5}
                      />
                    )}
                    <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fontWeight="bold"
                      className={colors.text}>
                      {n.newKey ?? n.key}
                    </text>
                    {n.isRoot && (
                      <text x={cx} y={cy + 28} textAnchor="middle" fontSize={9}
                        className="fill-violet-400 dark:fill-violet-400">
                        ROOT
                      </text>
                    )}
                    {n.mark && (
                      <text x={cx + 11} y={cy - 8} textAnchor="middle" fontSize={7}
                        className="fill-amber-200">
                        ★
                      </text>
                    )}
                  </g>
                )
              })}

              {/* 根列表指示 */}
              {step.rootList.length > 0 && (
                <text x={170} y={320} textAnchor="middle" fontSize={9}
                  className="fill-slate-400">
                  根列表: {step.rootList.map(id => `key=${nodeMap[id]?.key ?? id}`).join(' ↔ ')}
                </text>
              )}
            </svg>
          </div>

          {/* 步骤信息面板 */}
          <div className="md:w-64 space-y-3 shrink-0">
            {/* 步骤标题 */}
            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3.5">
              <div className="text-xs text-slate-400 dark:text-slate-500 mb-1">步骤 {safeStep + 1} / {steps.length}</div>
              <div className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-1">{step.title}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">{step.subtitle}</div>
            </div>

            {/* 注释 */}
            {step.annotation && (
              <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/50 p-3 text-xs text-amber-800 dark:text-amber-300 leading-relaxed">
                {step.annotation}
              </div>
            )}

            {/* 节点状态表格 */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-3 py-2 bg-slate-50 dark:bg-slate-800 text-xs font-semibold text-slate-500 dark:text-slate-400">
                节点状态
              </div>
              <div className="divide-y divide-slate-100 dark:divide-slate-700/50">
                {step.nodes.map(n => (
                  <div key={n.id} className={`flex items-center justify-between px-3 py-1.5 text-xs transition-colors ${
                    n.highlighted || n.cutThisStep ? 'bg-amber-50 dark:bg-amber-900/20' : ''}`}>
                    <span className="font-mono text-slate-700 dark:text-slate-300">
                      key={n.newKey ?? n.key}
                    </span>
                    <div className="flex gap-1">
                      {n.mark && (
                        <span className="px-1.5 py-0.5 rounded bg-rose-100 dark:bg-rose-900/40 text-rose-700 dark:text-rose-400 text-[10px] font-bold">
                          mark★
                        </span>
                      )}
                      {n.isRoot && (
                        <span className="px-1.5 py-0.5 rounded bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-400 text-[10px]">
                          root
                        </span>
                      )}
                      {n.cutThisStep && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400 text-[10px] font-bold">
                          CUT!
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 控制栏 */}
        <div className="flex items-center gap-3 px-5 pb-5">
          <button onClick={() => setStepIdx(i => Math.max(0, i - 1))}
            disabled={safeStep === 0}
            className="px-4 py-2 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            ← 上一步
          </button>

          {/* 进度条 */}
          <div className="flex-1 flex gap-1.5">
            {steps.map((_, i) => (
              <button key={i} onClick={() => setStepIdx(i)}
                className={`flex-1 h-2 rounded-full transition-all duration-300 ${
                  i === safeStep ? 'bg-amber-400' : i < safeStep ? 'bg-amber-200 dark:bg-amber-700' : 'bg-slate-200 dark:bg-slate-700'}`}
              />
            ))}
          </div>

          <button onClick={() => setStepIdx(i => Math.min(steps.length - 1, i + 1))}
            disabled={safeStep === steps.length - 1}
            className="px-4 py-2 text-xs rounded-lg bg-slate-900 dark:bg-slate-700 text-white hover:bg-slate-700 dark:hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            下一步 →
          </button>
        </div>

        {/* 势能公式提示 */}
        <div className="border-t border-slate-100 dark:border-slate-800 px-5 py-3 flex items-center gap-2 text-xs text-slate-400 dark:text-slate-600">
          <span className="font-mono bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-slate-600 dark:text-slate-400">
            Φ = t(H) + 2·m(H)
          </span>
          <span>每次 CUT：t+1, m-1 → ΔΦ=-1，摊销代价 = 实际代价 + ΔΦ = 1 + (-1) = 0</span>
        </div>
      </div>
    </div>
  )
}

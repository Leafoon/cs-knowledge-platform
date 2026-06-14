'use client'

import { useState } from 'react'

// ── 配色与数据 ────────────────────────────────────────────────
const N = 8
const INIT = [3, 2, 1, 6, 5, 4, 7, 2]

const PRESETS = [
  { ql: 3, qr: 7, val: 5, label: '区间 [3,7] 加 5' },
  { ql: 1, qr: 4, val: 2, label: '区间 [1,4] 加 2' },
  { ql: 5, qr: 8, val: 3, label: '区间 [5,8] 加 3' },
]

function nodeColor(lazy: number, active: boolean) {
  if (active) return 'bg-amber-400 dark:bg-amber-400 text-slate-900 border-amber-300 scale-110 shadow-lg shadow-amber-400/40'
  if (lazy)   return 'bg-amber-100 dark:bg-amber-900/40 text-amber-900 dark:text-amber-100 border-amber-400'
  return 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600'
}

// ── 线段树结构 ────────────────────────────────────────────────
interface Node {
  l: number
  r: number
  val: number
  lazy: number
  left?: Node
  right?: Node
}

function build(arr: number[], l: number, r: number): Node {
  if (l === r) return { l, r, val: arr[l - 1], lazy: 0 }
  const mid = Math.floor((l + r) / 2)
  const left = build(arr, l, mid)
  const right = build(arr, mid + 1, r)
  return { l, r, val: left.val + right.val, lazy: 0, left, right }
}

function clone(node: Node): Node {
  return { ...node, left: node.left && clone(node.left), right: node.right && clone(node.right) }
}

function rangeAdd(node: Node, ql: number, qr: number, val: number, steps: any[]): Node {
  node = clone(node)
  if (ql <= node.l && node.r <= qr) {
    node.val += val * (node.r - node.l + 1)
    node.lazy += val
    steps.push({ node, action: 'lazy', val })
    return node
  }
  if (node.r < ql || node.l > qr) return node
  // push-down
  if (node.lazy) {
    if (node.left) {
      node.left.val += node.lazy * (node.left.r - node.left.l + 1)
      node.left.lazy += node.lazy
    }
    if (node.right) {
      node.right.val += node.lazy * (node.right.r - node.right.l + 1)
      node.right.lazy += node.lazy
    }
    node.lazy = 0
    steps.push({ node, action: 'push-down' })
  }
  if (node.left) node.left = rangeAdd(node.left, ql, qr, val, steps)
  if (node.right) node.right = rangeAdd(node.right, ql, qr, val, steps)
  node.val = (node.left?.val ?? 0) + (node.right?.val ?? 0)
  return node
}

function collectNodes(node: Node, nodes: Node[] = []): Node[] {
  nodes.push(node)
  if (node.left) collectNodes(node.left, nodes)
  if (node.right) collectNodes(node.right, nodes)
  return nodes
}

export function SegmentTreeLazyProp() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [stepIdx, setStepIdx]     = useState(-1)

  const preset = PRESETS[presetIdx]
  const root0  = build(INIT, 1, N)
  const steps: any[] = []
  const root1 = rangeAdd(root0, preset.ql, preset.qr, preset.val, steps)

  function changePreset(i: number) { setPresetIdx(i); setStepIdx(-1) }
  function start()    { setStepIdx(0) }
  function prev()     { setStepIdx(s => Math.max(0, s - 1)) }
  function next()     { setStepIdx(s => Math.min(steps.length - 1, s + 1)) }
  function reset()    { setStepIdx(-1) }

  // 当前步骤
  const currentStep = stepIdx >= 0 && stepIdx < steps.length ? steps[stepIdx] : null

  // 展示所有节点
  const nodes = collectNodes(root0)
  const nodes1 = collectNodes(root1)

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      {/* 顶部：琥珀渐变 */}
      <div className="bg-gradient-to-r from-amber-500 via-yellow-400 to-orange-400 px-5 py-4">
        <h3 className="text-white font-bold text-base">🟧 线段树懒惰传播：区间修改步进动画</h3>
        <p className="text-amber-100 text-xs mt-0.5">
          区间修改时，先打懒标记（lazy），后续访问子节点时再 push-down 下推
        </p>
        <div className="flex gap-2 mt-3 flex-wrap">
          {PRESETS.map((p, idx) => (
            <button key={idx} onClick={() => changePreset(idx)}
              className={`px-2.5 py-1 text-xs rounded-lg font-mono font-medium transition-all ${
                presetIdx === idx ? 'bg-white text-amber-700 shadow' : 'bg-white/25 text-white hover:bg-white/35'}`}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        {/* 线段树结构可视化 */}
        <div className="flex flex-wrap gap-2">
          {nodes.map((node, idx) => {
            const isActive = currentStep && currentStep.node === node
            return (
              <div key={idx} className={`
                flex flex-col items-center rounded border text-[10px] font-mono font-bold py-1.5 px-2
                transition-all duration-300 ${nodeColor(node.lazy, isActive)}
              `}>
                <span className="font-bold text-[11px]">[{node.l},{node.r}]</span>
                <span className="text-[9px] font-normal opacity-70 leading-tight">和: {node.val}</span>
                {node.lazy > 0 && <span className="text-[9px] text-amber-600 dark:text-amber-300">lazy: {node.lazy}</span>}
                {isActive && <span className="text-amber-500 text-xs animate-bounce">★</span>}
              </div>
            )
          })}
        </div>

        {/* 步骤说明 */}
        {stepIdx >= 0 && currentStep && (
          <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-100 dark:border-amber-800 px-4 py-3">
            <p className="text-xs text-amber-800 dark:text-amber-200 leading-relaxed font-mono">
              {currentStep.action === 'lazy'
                ? `区间完全覆盖，打懒标记 lazy += ${preset.val}`
                : '访问子节点前，push-down 下推懒标记'}
            </p>
          </div>
        )}

        {stepIdx === -1 && (
          <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-center">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              线段树支持区间修改，懒惰传播可将修改延迟到访问子节点时再下推。
              点击"开始演示"查看懒标记与 push-down 的过程。
            </p>
          </div>
        )}

        {/* 控制按钮 */}
        <div className="flex items-center justify-between">
          <button onClick={reset}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
            ⏮ 重置
          </button>
          <div className="flex gap-2">
            {stepIdx === -1 ? (
              <button onClick={start}
                className="px-4 py-1.5 text-xs rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium transition-colors">
                ▶ 开始演示
              </button>
            ) : (
              <>
                <button onClick={prev} disabled={stepIdx === 0}
                  className="px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  ← 上一步
                </button>
                <button onClick={next} disabled={stepIdx === steps.length - 1}
                  className="px-3 py-1.5 text-xs rounded-lg bg-amber-600 hover:bg-amber-700 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  下一步 →
                </button>
              </>
            )}
          </div>
        </div>

        <div className="text-center text-xs text-slate-400 dark:text-slate-600 border-t border-slate-100 dark:border-slate-800 pt-3">
          懒惰传播可将区间修改复杂度降至 O(log n)，每次 push-down 只需 O(1)
        </div>
      </div>
    </div>
  )
}

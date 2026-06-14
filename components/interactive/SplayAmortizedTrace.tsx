"use client"

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, RotateCcw } from 'lucide-react'

// ─── Tree node type ───────────────────────────────────────────────────────────
interface TNode {
  key: number
  left: TNode | null
  right: TNode | null
  parent: TNode | null
  size: number        // subtree size (for rank)
}

function makeNode(key: number): TNode {
  return { key, left: null, right: null, parent: null, size: 1 }
}

function updateSize(n: TNode) {
  n.size = 1 + (n.left?.size ?? 0) + (n.right?.size ?? 0)
}

function rank(n: TNode): number {
  return Math.floor(Math.log2(n.size))
}

// Simple BST insert (no auto-splay on insert here)
function bstInsert(root: TNode | null, key: number): TNode {
  const node = makeNode(key)
  if (!root) return node
  let cur: TNode = root
  while (true) {
    cur.size++
    if (key < cur.key) {
      if (!cur.left) { cur.left = node; node.parent = cur; break }
      cur = cur.left
    } else {
      if (!cur.right) { cur.right = node; node.parent = cur; break }
      cur = cur.right
    }
  }
  return root
}

// Deep clone with parent wiring
function cloneTree(n: TNode | null, parent: TNode | null = null): TNode | null {
  if (!n) return null
  const c: TNode = { key: n.key, left: null, right: null, parent, size: n.size }
  c.left = cloneTree(n.left, c)
  c.right = cloneTree(n.right, c)
  return c
}

function findNode(root: TNode | null, key: number): TNode | null {
  if (!root) return null
  if (root.key === key) return root
  return key < root.key ? findNode(root.left, key) : findNode(root.right, key)
}

// ─── Splay rotations ──────────────────────────────────────────────────────────
function rotateRight(x: TNode, rootRef: { root: TNode }) {
  const p = x.parent!
  const g = p.parent
  const b = x.right

  // rewire
  x.right = p; p.parent = x
  p.left = b; if (b) b.parent = p
  x.parent = g
  if (g) { if (g.left === p) g.left = x; else g.right = x }
  else rootRef.root = x

  updateSize(p); updateSize(x)
}

function rotateLeft(x: TNode, rootRef: { root: TNode }) {
  const p = x.parent!
  const g = p.parent
  const b = x.left

  x.left = p; p.parent = x
  p.right = b; if (b) b.parent = p
  x.parent = g
  if (g) { if (g.left === p) g.left = x; else g.right = x }
  else rootRef.root = x

  updateSize(p); updateSize(x)
}

type CaseType = 'Zig' | 'Zig-Zig' | 'Zig-Zag'

interface SplayStep {
  case_: CaseType
  xKey: number
  pKey: number
  gKey: number | null
  rootBefore: TNode     // cloned tree before rotation
  rootAfter: TNode      // cloned tree after rotation
  phiBefore: number
  phiAfter: number
  deltaPhi: number
  realCost: number
  amortized: number
  rOfX_before: number
  rOfX_after: number
}

function phi(root: TNode | null): number {
  if (!root) return 0
  return Math.floor(Math.log2(root.size)) + phi(root.left) + phi(root.right)
}

// Perform one splay step, record it, return steps array
function splay(root: TNode, target: TNode): SplayStep[] {
  const steps: SplayStep[] = []
  const rootRef = { root }

  while (target.parent !== null) {
    const p = target.parent
    const g = p.parent

    const phiBefore = phi(rootRef.root)
    const rXBefore = rank(target)
    const cloneBefore = cloneTree(rootRef.root)!

    let case_: CaseType
    let cost = 0

    if (!g) {
      // Zig
      case_ = 'Zig'
      cost = 1
      if (p.left === target) rotateRight(target, rootRef)
      else rotateLeft(target, rootRef)
    } else if ((g.left === p && p.left === target) || (g.right === p && p.right === target)) {
      // Zig-Zig
      case_ = 'Zig-Zig'
      cost = 2
      if (p.left === target) { rotateRight(p, rootRef); rotateRight(target, rootRef) }
      else { rotateLeft(p, rootRef); rotateLeft(target, rootRef) }
    } else {
      // Zig-Zag
      case_ = 'Zig-Zag'
      cost = 2
      if (p.right === target) { rotateLeft(target, rootRef); rotateRight(target, rootRef) }
      else { rotateRight(target, rootRef); rotateLeft(target, rootRef) }
    }

    const phiAfter = phi(rootRef.root)
    const rXAfter = rank(target)
    const cloneAfter = cloneTree(rootRef.root)!
    const deltaPhi = phiAfter - phiBefore

    steps.push({
      case_,
      xKey: target.key,
      pKey: p.key,
      gKey: g ? g.key : null,
      rootBefore: cloneBefore,
      rootAfter: cloneAfter,
      phiBefore,
      phiAfter,
      deltaPhi,
      realCost: cost,
      amortized: cost + deltaPhi,
      rOfX_before: rXBefore,
      rOfX_after: rXAfter,
    })
  }

  return steps
}

// ─── SVG tree renderer ────────────────────────────────────────────────────────
interface Layout {
  x: number
  y: number
  node: TNode
}

function layoutTree(root: TNode | null, x: number, y: number, hgap: number, layouts: Layout[]) {
  if (!root) return
  layouts.push({ x, y, node: root })
  layoutTree(root.left, x - hgap, y + 56, hgap / 1.8, layouts)
  layoutTree(root.right, x + hgap, y + 56, hgap / 1.8, layouts)
}

function TreeSVG({
  root,
  highlight,
  width = 220,
  height = 180,
}: {
  root: TNode | null
  highlight?: number[]
  width?: number
  height?: number
}) {
  if (!root) return <svg width={width} height={height} />
  const layouts: Layout[] = []
  layoutTree(root, width / 2, 28, width / 4, layouts)
  const nodeMap = new Map(layouts.map(l => [l.node.key, l]))

  const edges: { x1: number; y1: number; x2: number; y2: number }[] = []
  layouts.forEach(l => {
    if (l.node.left) {
      const c = nodeMap.get(l.node.left.key)
      if (c) edges.push({ x1: l.x, y1: l.y, x2: c.x, y2: c.y })
    }
    if (l.node.right) {
      const c = nodeMap.get(l.node.right.key)
      if (c) edges.push({ x1: l.x, y1: l.y, x2: c.x, y2: c.y })
    }
  })

  return (
    <svg width={width} height={height} className="overflow-visible">
      {edges.map((e, i) => (
        <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
          stroke="#9ca3af" strokeWidth={1.5} />
      ))}
      {layouts.map(l => {
        const isHL = highlight?.includes(l.node.key)
        return (
          <g key={l.node.key}>
            <circle cx={l.x} cy={l.y} r={15}
              fill={isHL ? '#f59e0b' : '#4f46e5'}
              stroke={isHL ? '#d97706' : '#3730a3'}
              strokeWidth={2} />
            <text x={l.x} y={l.y + 4} textAnchor="middle" fontSize={11}
              fontWeight="bold" fill="white">{l.node.key}</text>
            {/* Rank label */}
            <text x={l.x + 16} y={l.y - 10} fontSize={9} fill="#6b7280">
              r={Math.floor(Math.log2(l.node.size))}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ─── Pre-built example scenarios ──────────────────────────────────────────────
interface Scenario {
  label: string
  keys: number[]
  target: number
  desc: string
}

const SCENARIOS: Scenario[] = [
  { label: 'Zig 示例', keys: [10, 5, 15], target: 5, desc: '目标节点的父节点是根 → 单旋 (Zig)' },
  { label: 'Zig-Zig 示例', keys: [20, 15, 10, 5, 8], target: 5, desc: '目标与父同侧 → Zig-Zig 双旋' },
  { label: 'Zig-Zag 示例', keys: [20, 10, 15, 5, 12], target: 12, desc: '目标与父异侧 → Zig-Zag 双旋' },
]

// ─── Main component ───────────────────────────────────────────────────────────
export function SplayAmortizedTrace() {
  const [scenIdx, setScenIdx] = useState(0)
  const [stepIdx, setStepIdx] = useState(-1)   // -1 = show initial

  const scen = SCENARIOS[scenIdx]

  // Build tree and compute splay steps
  const initialRoot = scen.keys.reduce<TNode | null>((r, k) => bstInsert(r, k), null)

  const steps: SplayStep[] = (() => {
    if (!initialRoot) return []
    const treeClone = cloneTree(initialRoot)!
    const tgt = findNode(treeClone, scen.target)
    if (!tgt) return []
    return splay(treeClone, tgt)
  })()

  const totalSteps = steps.length
  const cur = stepIdx >= 0 && stepIdx < totalSteps ? steps[stepIdx] : null
  const totalRealCost = steps.reduce((a, s) => a + s.realCost, 0)
  const totalAmortized = steps.reduce((a, s) => a + s.amortized, 0)
  const rootPhi = initialRoot ? phi(initialRoot) : 0
  const targetNode = initialRoot ? findNode(initialRoot, scen.target) : null
  const targetRank = targetNode ? rank(targetNode) : 0
  const rootRank = initialRoot ? rank(initialRoot) : 0

  // Display tree
  const displayRoot = cur
    ? (stepIdx === 0 ? cur.rootBefore : steps[stepIdx - 1]?.rootAfter ?? cur.rootBefore)
    : initialRoot

  const afterRoot = cur ? cur.rootAfter : null

  const changeScenario = (idx: number) => {
    setScenIdx(idx)
    setStepIdx(-1)
  }

  const caseColors: Record<CaseType, { bg: string; text: string; border: string }> = {
    'Zig': { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-300', border: 'border-blue-300 dark:border-blue-600' },
    'Zig-Zig': { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-300', border: 'border-orange-300 dark:border-orange-600' },
    'Zig-Zag': { bg: 'bg-purple-100 dark:bg-purple-900/30', text: 'text-purple-700 dark:text-purple-300', border: 'border-purple-300 dark:border-purple-600' },
  }

  return (
    <div className="rounded-xl border border-orange-200 dark:border-orange-800 bg-white dark:bg-gray-900 overflow-hidden shadow-sm my-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-amber-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">Splay 树摊销分析：旋转过程 & 势能跟踪</h3>
        <p className="text-orange-100 text-xs mt-0.5">
          Φ = Σ r(x)，r(x) = ⌊log₂(size(x))⌋，摊销代价 ĉ ≤ 3(r(t)−r(x))+1
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Scenario selector */}
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map((s, i) => (
            <button key={i} onClick={() => changeScenario(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                i === scenIdx
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >{s.label}</button>
          ))}
        </div>

        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
          <strong>目标：</strong>将节点 <span className="font-mono font-bold">{scen.target}</span> Splay 到根。
          {scen.desc}
        </div>

        {/* Step navigation */}
        <div className="flex flex-wrap items-center gap-3">
          <button onClick={() => setStepIdx(-1)}
            className="p-1.5 rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600">
            <RotateCcw size={16} className="text-gray-600 dark:text-gray-300" />
          </button>
          <button onClick={() => setStepIdx(p => Math.max(-1, p - 1))} disabled={stepIdx < 0}
            className="px-2 py-1 rounded text-sm bg-gray-100 dark:bg-gray-700 disabled:opacity-40 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300">‹ 上一步</button>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {stepIdx < 0 ? '初始状态' : `步骤 ${stepIdx + 1} / ${totalSteps}`}
          </span>
          <button onClick={() => setStepIdx(p => Math.min(totalSteps - 1, p + 1))} disabled={stepIdx >= totalSteps - 1}
            className="flex items-center gap-1 px-3 py-1.5 rounded bg-orange-500 text-white hover:bg-orange-600 text-sm disabled:opacity-40">
            下一步 <ChevronRight size={14} />
          </button>
        </div>

        {/* Trees side by side */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-2 text-center font-medium">
              {cur ? '旋转前' : '初始树'}
            </div>
            <div className="flex justify-center">
              <TreeSVG
                root={cur ? cur.rootBefore : initialRoot}
                highlight={cur ? [cur.xKey, cur.pKey, ...(cur.gKey != null ? [cur.gKey] : [])] : [scen.target]}
                width={200}
                height={170}
              />
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-2 text-center font-medium">
              {cur ? '旋转后' : '（等待操作）'}
            </div>
            <div className="flex justify-center">
              <TreeSVG
                root={cur ? cur.rootAfter : null}
                highlight={cur ? [cur.xKey] : []}
                width={200}
                height={170}
              />
            </div>
          </div>
        </div>

        {/* Legend for highlighted nodes */}
        <div className="flex flex-wrap gap-4 text-xs text-gray-500 dark:text-gray-400 justify-center">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full inline-block bg-amber-400" /> 参与旋转的节点 (x / p / g)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full inline-block bg-indigo-500" /> 其他节点
          </span>
          <span className="text-xs text-gray-400">每个节点右上角标注 r = ⌊log₂(size)⌋</span>
        </div>

        {/* Current step detail */}
        {cur && (() => {
          const cc = caseColors[cur.case_]
          return (
            <div className={`rounded-lg border ${cc.border} ${cc.bg} p-3 space-y-2`}>
              <div className={`font-bold text-sm ${cc.text}`}>
                {cur.case_} 旋转：x={cur.xKey}, p={cur.pKey}
                {cur.gKey != null ? `, g=${cur.gKey}` : ''}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-gray-500">实际代价 c</span>
                  <div className="font-mono font-bold text-gray-700 dark:text-gray-200">{cur.realCost}</div>
                </div>
                <div>
                  <span className="text-gray-500">ΔΦ</span>
                  <div className={`font-mono font-bold ${cur.deltaPhi < 0 ? 'text-red-500' : 'text-emerald-500'}`}>
                    {cur.deltaPhi > 0 ? '+' : ''}{cur.deltaPhi}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">ĉ = c + ΔΦ</span>
                  <div className={`font-mono font-bold ${cc.text}`}>{cur.amortized}</div>
                </div>
                <div>
                  <span className="text-gray-500">r(x) 变化</span>
                  <div className="font-mono font-bold text-gray-700 dark:text-gray-200">{cur.rOfX_before} → {cur.rOfX_after}</div>
                </div>
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Φ: {cur.phiBefore} → {cur.phiAfter}（{cur.deltaPhi >= 0 ? '增加' : '减少'} {Math.abs(cur.deltaPhi)}）
              </div>
            </div>
          )
        })()}

        {/* Initial state info */}
        {stepIdx < 0 && (
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-xs text-gray-600 dark:text-gray-400">
            <div className="font-semibold text-gray-700 dark:text-gray-300 mb-1">初始状态分析</div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div>
                <span>初始 Φ = Σr(x)</span>
                <div className="font-mono font-bold text-amber-600 dark:text-amber-400">{rootPhi}</div>
              </div>
              <div>
                <span>r(目标节点 {scen.target})</span>
                <div className="font-mono font-bold">{targetRank}</div>
              </div>
              <div>
                <span>r(根节点 {initialRoot?.key})</span>
                <div className="font-mono font-bold">{rootRank}</div>
              </div>
            </div>
            <div className="mt-2 text-gray-500">
              摊销代价上界：3(r(root) − r(target)) + 1 = 3×({rootRank}−{targetRank})+1 = {3 * (rootRank - targetRank) + 1}
            </div>
          </div>
        )}

        {/* Step history summary */}
        {steps.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-orange-100 dark:bg-orange-900/30">
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">步骤</th>
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">情形</th>
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">x</th>
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">实际 c</th>
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">ΔΦ</th>
                  <th className="px-2 py-1 text-left border border-orange-200 dark:border-orange-800">ĉ</th>
                </tr>
              </thead>
              <tbody>
                {steps.map((s, i) => {
                  const cc = caseColors[s.case_]
                  return (
                    <tr
                      key={i}
                      className={`cursor-pointer transition-colors ${i === stepIdx ? `${cc.bg} font-semibold` : 'hover:bg-gray-50 dark:hover:bg-gray-800'}`}
                      onClick={() => setStepIdx(i)}
                    >
                      <td className="px-2 py-1 border border-orange-100 dark:border-orange-900">{i + 1}</td>
                      <td className={`px-2 py-1 border border-orange-100 dark:border-orange-900 ${cc.text}`}>{s.case_}</td>
                      <td className="px-2 py-1 border border-orange-100 dark:border-orange-900 font-mono">{s.xKey}</td>
                      <td className="px-2 py-1 border border-orange-100 dark:border-orange-900 font-mono">{s.realCost}</td>
                      <td className={`px-2 py-1 border border-orange-100 dark:border-orange-900 font-mono ${s.deltaPhi < 0 ? 'text-red-500' : 'text-emerald-500'}`}>
                        {s.deltaPhi > 0 ? '+' : ''}{s.deltaPhi}
                      </td>
                      <td className={`px-2 py-1 border border-orange-100 dark:border-orange-900 font-mono font-bold ${cc.text}`}>{s.amortized}</td>
                    </tr>
                  )
                })}
                {steps.length > 0 && (
                  <tr className="bg-amber-50 dark:bg-amber-900/20 font-semibold">
                    <td className="px-2 py-1 border border-orange-200 dark:border-orange-800" colSpan={3}>合计</td>
                    <td className="px-2 py-1 border border-orange-200 dark:border-orange-800 font-mono">{totalRealCost}</td>
                    <td className="px-2 py-1 border border-orange-200 dark:border-orange-800 font-mono">{steps[steps.length-1].phiAfter - rootPhi > 0 ? '+':''}{steps[steps.length-1].phiAfter - rootPhi}</td>
                    <td className="px-2 py-1 border border-orange-200 dark:border-orange-800 font-mono text-amber-600 dark:text-amber-400">{totalAmortized}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg p-3 text-xs text-amber-800 dark:text-amber-300">
          <strong>访问引理：</strong>对节点 x 的一次 Splay 操作，摊销代价 ĉ ≤ 3(r(t)−r(x))+1 = O(log n)，
          其中 t 是树的根，r(v) = ⌊log₂(size(v))⌋。
          Zig-Zig 是关键：通过先旋祖父再旋父，使 r(x) 快速增大，从而势能降低来支付实际代价。
        </div>
      </div>
    </div>
  )
}

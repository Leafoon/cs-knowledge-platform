'use client'

import { useState, useMemo } from 'react'

const MAX_BITS = 5   // 显示 5 位二进制（值域 0~31）

// ─── 预设 ────────────────────────────────────────────────────
const PRESETS = [
  { label: '[3,10,5,25,2,8]', nums: [3, 10, 5, 25, 2, 8] },
  { label: '[14,7,3,9,1,6]',  nums: [14, 7, 3, 9, 1, 6] },
  { label: '[20,4,28,12,8]',  nums: [20, 4, 28, 12, 8] },
  { label: '[1,2,4,8,16]',    nums: [1, 2, 4, 8, 16] },
]

// ─── 二进制 Trie 节点 ─────────────────────────────────────────
interface BTNode {
  id: number
  bit: number        // 0 或 1（此节点经哪个 bit 到达）
  children: [BTNode | null, BTNode | null]
  depth: number
  pathIndex: number  // 从根到此节点的 bit 路径表示的整数（用于布局）
  parentId: number | null
}

// ─── 构建 ────────────────────────────────────────────────────
function buildBitwiseTrie(nums: number[]) {
  let idCnt = 0
  const root: BTNode = { id: idCnt++, bit: -1, children: [null, null], depth: 0, pathIndex: 0, parentId: null }
  const byId = new Map<number, BTNode>([[root.id, root]])

  for (const num of nums) {
    let node = root
    for (let d = MAX_BITS - 1; d >= 0; d--) {
      const b = (num >> d) & 1
      if (!node.children[b]) {
        const newNode: BTNode = {
          id: idCnt++, bit: b, children: [null, null],
          depth: node.depth + 1,
          pathIndex: node.pathIndex * 2 + b,
          parentId: node.id
        }
        node.children[b] = newNode
        byId.set(newNode.id, newNode)
      }
      node = node.children[b]!
    }
  }
  return { root, byId }
}

// ─── SVG 布局（完全二叉树坐标）────────────────────────────────
const SVG_W = 660
const LEVEL_H = 60
const PAD = 20

function getPos(depth: number, pathIndex: number): { x: number; y: number } {
  const slots = Math.pow(2, depth)   // 该深度的总格子数
  const x = PAD + (SVG_W - 2 * PAD) * (2 * pathIndex + 1) / (2 * slots)
  const y = depth * LEVEL_H + 32
  return { x, y }
}

// ─── 贪心 XOR-max 查询步骤 ────────────────────────────────────
interface XorStep {
  depth: number
  queryBit: number
  targetBit: number
  choseBit: number
  xorBit: number
  nodeId: number
  desc: string
}

function computeXorSteps(root: BTNode, byId: Map<number, BTNode>, num: number): XorStep[] {
  const steps: XorStep[] = []
  let node = root
  for (let d = MAX_BITS - 1; d >= 0; d--) {
    const b = (num >> d) & 1
    const target = 1 - b
    let chosen: number, xorBit: number, nextNode: BTNode

    if (node.children[target]) {
      chosen = target
      xorBit = 1
      nextNode = node.children[target]!
    } else {
      chosen = b
      xorBit = 0
      nextNode = node.children[b]!
    }

    steps.push({
      depth: MAX_BITS - d,
      queryBit: b,
      targetBit: target,
      choseBit: chosen,
      xorBit,
      nodeId: nextNode.id,
      desc: `第 ${MAX_BITS - d} 步（位 ${d}）：查询数该位 = ${b}，理想方向 = ${target}，
        ${node.children[target] ? `Trie 有 ${target} 方向 → 走 ${target}，XOR 该位 = 1 ✓` : `Trie 无 ${target} 方向 → 走 ${b}，XOR 该位 = 0`}`
    })
    node = nextNode
  }
  return steps
}

// ─── 主组件 ─────────────────────────────────────────────────
export default function BitwiseTrieXOR() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [queryIdx, setQueryIdx] = useState(0)     // 查询数组的哪个数
  const [stepIdx, setStepIdx] = useState(-1)       // -1 = 不显示查询路径

  const { nums } = PRESETS[presetIdx]
  const { root, byId } = useMemo(() => buildBitwiseTrie(nums), [nums])

  const queryNum = nums[queryIdx % nums.length]

  const xorSteps = useMemo(() => computeXorSteps(root, byId, queryNum), [root, byId, queryNum])

  // 高亮节点路径
  const pathNodeIds = useMemo(() => {
    if (stepIdx < 0) return new Set<number>()
    const ids = new Set<number>([root.id])
    xorSteps.slice(0, stepIdx + 1).forEach(s => ids.add(s.nodeId))
    return ids
  }, [stepIdx, xorSteps, root.id])

  // 累计 XOR 结果
  const partialXor = useMemo(() => {
    let x = 0
    for (let i = 0; i <= stepIdx && i < xorSteps.length; i++) {
      if (xorSteps[i].xorBit) x |= (1 << (MAX_BITS - 1 - i))
    }
    return x
  }, [stepIdx, xorSteps])

  // 计算真实最大 XOR（枚举所有对）
  const maxXor = useMemo(() => {
    let best = 0
    for (let i = 0; i < nums.length; i++)
      for (let j = i + 1; j < nums.length; j++)
        best = Math.max(best, nums[i] ^ nums[j])
    return best
  }, [nums])

  // SVG 节点列表
  const allNodes = useMemo(() => [...byId.values()], [byId])
  const svgH = (MAX_BITS + 1) * LEVEL_H + 20

  const toBin = (n: number) => n.toString(2).padStart(MAX_BITS, '0')

  return (
    <div className="rounded-2xl overflow-hidden border border-amber-200 dark:border-amber-800 shadow-lg">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 dark:from-amber-600 dark:to-orange-600 px-5 py-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-white font-bold text-base">⚡ 二进制 Trie XOR 最大值</h3>
            <p className="text-amber-100 text-xs mt-0.5">按位贪心：每位优先选使 XOR 为 1 的方向</p>
          </div>
          <div className="flex gap-1 flex-wrap">
            {PRESETS.map((p, i) => (
              <button key={i} onClick={() => { setPresetIdx(i); setQueryIdx(0); setStepIdx(-1) }}
                className={`px-2.5 py-1 text-xs rounded-lg transition-all ${
                  presetIdx === i ? 'bg-white text-amber-700 font-semibold' : 'bg-amber-400/40 text-white hover:bg-amber-400/60'}`}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-900 p-4">
        {/* 数组展示 + 二进制 */}
        <div className="mb-3 overflow-x-auto">
          <table className="text-xs border-collapse w-full min-w-max">
            <thead>
              <tr>
                <th className="text-left text-gray-500 dark:text-gray-400 font-medium pb-1 pr-4">数值</th>
                {nums.map(n => (
                  <th key={n} className={`px-2 pb-1 font-mono font-bold ${
                    n === queryNum && stepIdx >= 0 ? 'text-amber-600 dark:text-amber-400' : 'text-gray-700 dark:text-gray-200'}`}>
                    {n}
                  </th>
                ))}
              </tr>
              <tr>
                <th className="text-left text-gray-500 dark:text-gray-400 font-medium pr-4">二进制</th>
                {nums.map(n => (
                  <td key={n} className={`px-2 font-mono text-center ${
                    n === queryNum && stepIdx >= 0 ? 'text-amber-600 dark:text-amber-400 font-bold' : 'text-gray-500 dark:text-gray-400'}`}>
                    {toBin(n)}
                  </td>
                ))}
              </tr>
            </thead>
          </table>
        </div>

        {/* 查询控制 */}
        <div className="flex items-center gap-2 flex-wrap mb-3">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400">查询数：</span>
          {nums.map((n, i) => (
            <button key={i} onClick={() => { setQueryIdx(i); setStepIdx(-1) }}
              className={`px-2.5 py-1 text-xs font-mono rounded-lg border transition-all ${
                queryIdx % nums.length === i
                  ? 'bg-amber-100 border-amber-400 text-amber-700 dark:bg-amber-900/30 dark:border-amber-600 dark:text-amber-300 font-bold'
                  : 'border-gray-200 text-gray-500 dark:border-gray-700 dark:text-gray-400 hover:border-amber-300'}`}>
              {n}
            </button>
          ))}
          <span className="text-gray-400 dark:text-gray-500 text-xs">= {toBin(queryNum)}</span>
        </div>

        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <button onClick={() => setStepIdx(-1)}
            className="px-2.5 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-lg">↺ 重置</button>
          <button onClick={() => setStepIdx(v => Math.max(-1, v - 1))} disabled={stepIdx < 0}
            className="px-2.5 py-1 text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg disabled:opacity-40">← 上一位</button>
          <button onClick={() => setStepIdx(v => Math.min(MAX_BITS - 1, v + 1))} disabled={stepIdx >= MAX_BITS - 1}
            className="px-3 py-1 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-40">贪心选位 →</button>
          <button onClick={() => setStepIdx(MAX_BITS - 1)}
            className="px-2.5 py-1 text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-lg">查看最终结果 ⤵</button>
        </div>

        {/* 当前步骤卡片 */}
        {stepIdx >= 0 && (
          <div className="mb-3 grid grid-cols-5 gap-2">
            {xorSteps.slice(0, stepIdx + 1).map((s, i) => (
              <div key={i} className={`p-2 rounded-lg border text-center transition-all ${
                i === stepIdx
                  ? 'bg-amber-100 border-amber-400 dark:bg-amber-900/30 dark:border-amber-600 ring-2 ring-amber-400/50'
                  : 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700'
              }`}>
                <div className="text-[10px] text-gray-400 dark:text-gray-500">位 {MAX_BITS - 1 - i}</div>
                <div className="text-xs font-mono mt-0.5">
                  <span className="text-blue-500">{s.queryBit}</span>
                  <span className="text-gray-400 mx-0.5">→</span>
                  <span className={s.xorBit ? 'text-emerald-600 font-bold' : 'text-rose-500'}>
                    XOR={s.xorBit}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 步骤文字描述 */}
        {stepIdx >= 0 && (
          <div className="mb-3 p-2.5 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg">
            <p className="text-xs text-amber-800 dark:text-amber-200 leading-relaxed">
              {xorSteps[stepIdx].desc}
            </p>
            {stepIdx === MAX_BITS - 1 && (
              <div className="mt-2 pt-2 border-t border-amber-200 dark:border-amber-700 flex items-center gap-4">
                <div className="text-xs">
                  <span className="text-gray-500">与 {queryNum} 的最大 XOR = </span>
                  <span className="font-bold text-amber-700 dark:text-amber-300 font-mono">
                    {partialXor}（{toBin(partialXor)}）
                  </span>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500">数组全局最大 XOR = </span>
                  <span className="font-bold text-emerald-600 dark:text-emerald-400 font-mono">{maxXor}</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 二进制 Trie SVG */}
        <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
          <svg width={SVG_W} height={svgH} viewBox={`0 0 ${SVG_W} ${svgH}`} className="w-full max-w-full">
            {/* 边 */}
            {allNodes.filter(n => n.parentId !== null).map(n => {
              const par = byId.get(n.parentId!)!
              const { x: x1, y: y1 } = getPos(par.depth, par.pathIndex)
              const { x: x2, y: y2 } = getPos(n.depth, n.pathIndex)
              const inPath = pathNodeIds.has(par.id) && pathNodeIds.has(n.id)
              return (
                <g key={`e-${n.id}`}>
                  <line x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={inPath ? '#f59e0b' : '#e5e7eb'}
                    strokeWidth={inPath ? 3 : 1.5}
                    className="transition-all duration-300"
                    strokeDasharray={inPath ? 'none' : 'none'}
                  />
                  {/* 边上标注 0/1 */}
                  <text x={(x1 + x2) / 2 + (n.bit === 0 ? -8 : 8)} y={(y1 + y2) / 2}
                    textAnchor="middle" dominantBaseline="middle"
                    fill={inPath ? '#d97706' : '#9ca3af'} fontSize={11} fontWeight={inPath ? '700' : '400'}
                    className="select-none transition-all duration-300">
                    {n.bit}
                  </text>
                </g>
              )
            })}

            {/* 节点 */}
            {allNodes.map(n => {
              const { x, y } = getPos(n.depth, n.pathIndex)
              const isRoot = n.parentId === null
              const inPath = pathNodeIds.has(n.id)
              const isEnd = n.depth === MAX_BITS
              const fill = inPath
                ? (isEnd ? '#d97706' : '#f59e0b')
                : isRoot ? '#6b7280' : isEnd ? '#fed7aa' : '#fef3c7'
              const stroke = inPath ? '#b45309' : isEnd ? '#fb923c' : '#fcd34d'
              const textFill = inPath || isRoot ? '#fff' : isEnd ? '#92400e' : '#78350f'
              const r = isRoot ? 15 : isEnd ? 13 : 12
              return (
                <g key={n.id} className="transition-all duration-300">
                  <circle cx={x} cy={y} r={r} fill={fill} stroke={stroke} strokeWidth={inPath ? 2.5 : 1.5}
                    className="transition-all duration-300" />
                  {isRoot ? (
                    <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                      fill={textFill} fontSize={11} fontWeight="700" className="select-none">root</text>
                  ) : isEnd ? (
                    <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle"
                      fill={textFill} fontSize={10} fontWeight="600" fontFamily="monospace" className="select-none">
                      {n.pathIndex}
                    </text>
                  ) : null}
                </g>
              )
            })}

            {/* 深度标注（位数标签）*/}
            {Array.from({ length: MAX_BITS }, (_, i) => (
              <text key={i} x={12} y={32 + (i + 1) * LEVEL_H + 2} dominantBaseline="middle"
                fill="#9ca3af" fontSize={10} fontFamily="monospace" className="select-none">
                位{MAX_BITS - 1 - i}
              </text>
            ))}
          </svg>
        </div>

        <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
          🟡 黄色路径 = 贪心 XOR-max 查询路径 · 叶节点数字 = 存储的值（十进制）
        </p>
      </div>
    </div>
  )
}

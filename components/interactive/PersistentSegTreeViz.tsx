'use client'

import { useMemo, useState } from 'react'

type Node = {
  id: string
  l: number
  r: number
  val: number
  left?: Node
  right?: Node
}

type UpdateOp = { pos: number; value: number; label: string }

const BASE = [2, 1, 4, 3, 5, 7, 6, 8]
const OPS: UpdateOp[] = [
  { pos: 3, value: 10, label: 'v1: A[3]=10' },
  { pos: 6, value: 0, label: 'v2: A[6]=0' },
  { pos: 1, value: 9, label: 'v3: A[1]=9' },
]

let gid = 0
function newId(prefix: string) {
  gid += 1
  return `${prefix}-${gid}`
}

function resetId() {
  gid = 0
}

function build(arr: number[], l: number, r: number): Node {
  if (l === r) return { id: newId('n'), l, r, val: arr[l - 1] }
  const mid = Math.floor((l + r) / 2)
  const left = build(arr, l, mid)
  const right = build(arr, mid + 1, r)
  return { id: newId('n'), l, r, val: left.val + right.val, left, right }
}

function update(prev: Node, pos: number, value: number): Node {
  if (prev.l === prev.r) {
    return { id: newId('u'), l: prev.l, r: prev.r, val: value }
  }

  const mid = Math.floor((prev.l + prev.r) / 2)
  let left = prev.left!
  let right = prev.right!

  if (pos <= mid) left = update(prev.left!, pos, value)
  else right = update(prev.right!, pos, value)

  return {
    id: newId('u'),
    l: prev.l,
    r: prev.r,
    val: left.val + right.val,
    left,
    right,
  }
}

function collect(node: Node, list: Node[] = []): Node[] {
  list.push(node)
  if (node.left) collect(node.left, list)
  if (node.right) collect(node.right, list)
  return list
}

function query(node: Node, ql: number, qr: number): number {
  if (ql <= node.l && node.r <= qr) return node.val
  if (node.r < ql || node.l > qr) return 0
  return query(node.left!, ql, qr) + query(node.right!, ql, qr)
}

export function PersistentSegTreeViz() {
  const [currentVersion, setCurrentVersion] = useState(0)
  const [rangeL, setRangeL] = useState(2)
  const [rangeR, setRangeR] = useState(6)

  const { versions, changedIdsByVersion } = useMemo(() => {
    resetId()
    const root0 = build(BASE, 1, BASE.length)
    const roots: Node[] = [root0]
    const changed: string[][] = [[]]

    for (const op of OPS) {
      const prev = roots[roots.length - 1]
      const next = update(prev, op.pos, op.value)
      roots.push(next)

      const prevSet = new Set(collect(prev).map(n => n.id))
      const nextNodes = collect(next)
      changed.push(nextNodes.filter(n => !prevSet.has(n.id)).map(n => n.id))
    }

    return { versions: roots, changedIdsByVersion: changed }
  }, [])

  const root = versions[currentVersion]
  const nodes = collect(root)
  const changedIds = new Set(changedIdsByVersion[currentVersion])

  const safeL = Math.min(rangeL, rangeR)
  const safeR = Math.max(rangeL, rangeR)
  const answer = query(root, safeL, safeR)

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden shadow-lg font-sans">
      <div className="bg-gradient-to-r from-fuchsia-600 via-purple-600 to-indigo-600 px-5 py-4">
        <h3 className="text-white font-bold text-base">🟪 可持久化线段树：版本共享与路径复制</h3>
        <p className="text-fuchsia-100 text-xs mt-1">每次单点更新只新建 O(log n) 个节点，其余节点跨版本共享</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-5 space-y-5">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-800/60">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-2">版本选择</p>
            <div className="flex flex-wrap gap-2">
              {versions.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentVersion(i)}
                  className={`px-2.5 py-1 rounded-md text-xs font-mono border ${
                    currentVersion === i
                      ? 'bg-purple-600 text-white border-purple-600'
                      : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  v{i}
                </button>
              ))}
            </div>
            <div className="mt-3 text-xs font-mono text-slate-600 dark:text-slate-300 leading-relaxed">
              v0: base = {JSON.stringify(BASE)}
              <br />
              {OPS.map((op, i) => (
                <span key={op.label}>
                  v{i + 1}: {op.label}
                  <br />
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 bg-slate-50 dark:bg-slate-800/60">
            <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-2">版本内区间求和</p>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-700 dark:text-slate-200 flex-wrap">
              <span>sum(v{currentVersion}, [</span>
              <input
                type="number"
                min={1}
                max={BASE.length}
                value={rangeL}
                onChange={(e) => setRangeL(Number(e.target.value || 1))}
                className="w-14 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1"
              />
              <span>,</span>
              <input
                type="number"
                min={1}
                max={BASE.length}
                value={rangeR}
                onChange={(e) => setRangeR(Number(e.target.value || 1))}
                className="w-14 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 px-2 py-1"
              />
              <span>]) =</span>
              <span className="px-2 py-1 rounded bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-200 font-bold">{answer}</span>
            </div>
            <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
              每个版本都是一棵独立根节点树，但绝大多数子树引用旧节点，不重复拷贝。
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <p className="text-xs font-semibold text-slate-700 dark:text-slate-200 mb-3">当前版本节点视图（新建节点高亮）</p>
          <div className="flex flex-wrap gap-2">
            {nodes.map((node) => (
              <div
                key={node.id}
                className={`rounded-lg border px-2.5 py-1.5 text-[11px] font-mono transition-colors ${
                  changedIds.has(node.id)
                    ? 'bg-fuchsia-100 dark:bg-fuchsia-900/40 border-fuchsia-300 dark:border-fuchsia-700 text-fuchsia-800 dark:text-fuchsia-200'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300'
                }`}
              >
                [{node.l},{node.r}] sum={node.val}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-900/20 px-4 py-3 text-xs text-indigo-800 dark:text-indigo-200 leading-relaxed">
          <span className="font-mono">核心思想：</span>point update 时，仅根到叶子的路径节点被复制（path-copy），其余子树直接复用旧版本节点。<br />
          因此：单次更新时间/新增空间均为 <span className="font-mono">O(log n)</span>，历史版本可随时回溯查询。
        </div>
      </div>
    </div>
  )
}

'use client'

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Play, Pause, SkipBack, ChevronLeft, ChevronRight } from 'lucide-react'

interface HNode {
  id: string
  freq: number
  isLeaf: boolean
  leftId?: string
  rightId?: string
  code?: string
}

interface QueueState { nodes: HNode[] }

interface HStep {
  desc: string
  detail: string
  queue: HNode[]
  mergedId?: string
  leftId?: string
  rightId?: string
  allNodes: Map<string, HNode>
  encodings?: Map<string, string>
}

// Pre-compute Huffman construction for "ABRACADABRA"
// Frequencies: A=5, B=2, R=2, C=1, D=1
function buildHuffmanSteps(): HStep[] {
  const steps: HStep[] = []
  const allNodes = new Map<string, HNode>()
  let counter = 0
  const newId = () => `n${counter++}`

  const leafA: HNode = { id: 'A', freq: 5, isLeaf: true }
  const leafB: HNode = { id: 'B', freq: 2, isLeaf: true }
  const leafR: HNode = { id: 'R', freq: 2, isLeaf: true }
  const leafC: HNode = { id: 'C', freq: 1, isLeaf: true }
  const leafD: HNode = { id: 'D', freq: 1, isLeaf: true }

  for (const n of [leafA, leafB, leafR, leafC, leafD]) allNodes.set(n.id, n)

  counter = 0
  let queue: HNode[] = [leafA, leafB, leafR, leafC, leafD]
    .sort((a, b) => a.freq - b.freq || a.id.localeCompare(b.id))

  // Step 0: Initial
  steps.push({
    desc: '初始状态：构建频率优先队列（最小堆）',
    detail: '统计字符频率后，将 5 个字符按频率升序放入最小堆：C(1)、D(1)、B(2)、R(2)、A(5)。每次从堆中取出两个最小节点合并，保证高频字符得到更短的编码。',
    queue: [...queue],
    allNodes: new Map(allNodes),
  })

  // Merge 1: C(1) + D(1) → CD(2)
  const n1 = queue.shift()!
  const n2 = queue.shift()!
  const m1: HNode = { id: 'CD', freq: 2, isLeaf: false, leftId: n1.id, rightId: n2.id }
  allNodes.set(m1.id, m1)
  queue.push(m1)
  queue.sort((a, b) => a.freq - b.freq || a.id.localeCompare(b.id))
  steps.push({
    desc: '合并步骤 1：C(1) + D(1) → [CD](2)',
    detail: '从最小堆弹出最小两个节点 C(freq=1) 和 D(freq=1)，创建内部节点 [CD]，频率 = 1+1 = 2。将 [CD] 重新入堆并保持最小堆性质。队列变为：B(2)、R(2)、[CD](2)、A(5)。',
    queue: [...queue],
    mergedId: m1.id,
    leftId: n1.id,
    rightId: n2.id,
    allNodes: new Map(allNodes),
  })

  // Merge 2: B(2) + R(2) → BR(4)
  const n3 = queue.shift()!
  const n4 = queue.shift()!
  const m2: HNode = { id: 'BR', freq: 4, isLeaf: false, leftId: n3.id, rightId: n4.id }
  allNodes.set(m2.id, m2)
  queue.push(m2)
  queue.sort((a, b) => a.freq - b.freq || a.id.localeCompare(b.id))
  steps.push({
    desc: '合并步骤 2：B(2) + R(2) → [BR](4)',
    detail: '弹出 B(freq=2) 和 R(freq=2)，创建内部节点 [BR]，频率 = 2+2 = 4。注意 [CD](2) 已在前一步合并，此步处理另外两个频率为 2 的字符。队列变为：[CD](2)、A(5)、[BR](4)。',
    queue: [...queue],
    mergedId: m2.id,
    leftId: n3.id,
    rightId: n4.id,
    allNodes: new Map(allNodes),
  })

  // Merge 3: CD(2) + BR(4) → BCDR(6)
  const n5 = queue.shift()!
  const n6 = queue.shift()!
  const m3: HNode = { id: 'BCDR', freq: 6, isLeaf: false, leftId: n5.id, rightId: n6.id }
  allNodes.set(m3.id, m3)
  queue.push(m3)
  queue.sort((a, b) => a.freq - b.freq || a.id.localeCompare(b.id))
  steps.push({
    desc: '合并步骤 3：[CD](2) + [BR](4) → [BCDR](6)',
    detail: '弹出 [CD](freq=2) 和 [BR](freq=4)，创建内部节点 [BCDR]，频率 = 2+4 = 6。注意左右子树分配：频率小的 [CD] 为左子树（编码前缀 0），[BR] 为右子树（前缀 1）。队列变为：A(5)、[BCDR](6)。',
    queue: [...queue],
    mergedId: m3.id,
    leftId: n5.id,
    rightId: n6.id,
    allNodes: new Map(allNodes),
  })

  // Merge 4: A(5) + BCDR(6) → root(11)
  const n7 = queue.shift()!
  const n8 = queue.shift()!
  const root: HNode = { id: 'ROOT', freq: 11, isLeaf: false, leftId: n7.id, rightId: n8.id }
  allNodes.set(root.id, root)
  queue.push(root)

  // Compute codes
  const encodings = new Map<string, string>()
  function traverse(id: string, code: string) {
    const node = allNodes.get(id)!
    if (node.isLeaf) { encodings.set(id, code); return }
    if (node.leftId) traverse(node.leftId, code + '0')
    if (node.rightId) traverse(node.rightId, code + '1')
  }
  traverse(root.id, '')

  steps.push({
    desc: '合并步骤 4：A(5) + [BCDR](6) → 根节点(11)',
    detail: '弹出 A(freq=5) 和 [BCDR](freq=6)，创建根节点，频率 = 5+6 = 11（等于字符串总长度，正确！）。贪心策略保证了此树的总带权路径长度（WPL）最小，即是最优前缀码。',
    queue: [root],
    mergedId: root.id,
    leftId: n7.id,
    rightId: n8.id,
    allNodes: new Map(allNodes),
    encodings,
  })

  return steps
}

const HUFFMAN_STEPS = buildHuffmanSteps()

const LEAF_COLORS = ['bg-cyan-500', 'bg-blue-500', 'bg-indigo-500', 'bg-violet-500', 'bg-emerald-500']
const LEAF_IDS = ['A', 'B', 'R', 'C', 'D']

function leafColor(id: string): string {
  const idx = LEAF_IDS.indexOf(id)
  return idx >= 0 ? LEAF_COLORS[idx] : 'bg-slate-500'
}

const SPEEDS = [0.5, 1, 1.5, 2]

// Simple tree renderer: shows merge arrow chain for current step
function MergeArrow({ leftId, rightId, mergedId, allNodes }: { leftId?: string; rightId?: string; mergedId?: string; allNodes: Map<string, HNode> }) {
  if (!leftId || !rightId || !mergedId) return null
  const L = allNodes.get(leftId)
  const R = allNodes.get(rightId)
  const M = allNodes.get(mergedId)
  if (!L || !R || !M) return null
  return (
    <div className="flex items-center justify-center gap-3 py-2">
      <div className={`flex flex-col items-center px-3 py-2 rounded-lg border-2 ${L.isLeaf ? 'border-cyan-400 dark:border-cyan-500 bg-cyan-50 dark:bg-cyan-950/50' : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800'}`}>
        <div className={`text-2xl font-black ${L.isLeaf ? 'text-cyan-700 dark:text-cyan-300' : 'text-slate-700 dark:text-slate-200'}`}>{leftId}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400">freq={L.freq}</div>
        {L.isLeaf && <div className="text-[10px] text-cyan-600 dark:text-cyan-400 mt-0.5">叶节点</div>}
      </div>
      <div className="text-slate-400 dark:text-slate-500 font-bold">+</div>
      <div className={`flex flex-col items-center px-3 py-2 rounded-lg border-2 ${R.isLeaf ? 'border-blue-400 dark:border-blue-500 bg-blue-50 dark:bg-blue-950/50' : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800'}`}>
        <div className={`text-2xl font-black ${R.isLeaf ? 'text-blue-700 dark:text-blue-300' : 'text-slate-700 dark:text-slate-200'}`}>{rightId}</div>
        <div className="text-xs text-slate-500 dark:text-slate-400">freq={R.freq}</div>
        {R.isLeaf && <div className="text-[10px] text-blue-600 dark:text-blue-400 mt-0.5">叶节点</div>}
      </div>
      <div className="text-slate-400 dark:text-slate-500 font-bold">→</div>
      <div className="flex flex-col items-center px-3 py-2 rounded-lg border-2 border-amber-400 dark:border-amber-500 bg-amber-50 dark:bg-amber-950/50">
        <div className="text-2xl font-black text-amber-700 dark:text-amber-300">[{mergedId}]</div>
        <div className="text-xs text-slate-500 dark:text-slate-400">freq={M.freq}</div>
        <div className="text-[10px] text-amber-600 dark:text-amber-400 mt-0.5">内部节点</div>
      </div>
    </div>
  )
}

export default function HuffmanTreeBuilder() {
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cur = HUFFMAN_STEPS[idx]
  const total = HUFFMAN_STEPS.length - 1

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setPlaying(false)
  }, [])

  useEffect(() => {
    if (!playing) return
    timerRef.current = setInterval(() => {
      setIdx(prev => {
        if (prev >= total) { stop(); return prev }
        return prev + 1
      })
    }, 1400 / SPEEDS[speedIdx])
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [playing, speedIdx, total, stop])

  useEffect(() => { if (idx >= total && playing) stop() }, [idx, total, playing, stop])
  const goto = (n: number) => { stop(); setIdx(Math.max(0, Math.min(total, n))) }

  return (
    <div className="w-full max-w-5xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 px-5 py-4">
        <h3 className="text-white font-bold text-lg">Huffman 树逐步构建 — 最优前缀码</h3>
        <p className="text-purple-100 text-sm mt-0.5">对字符串 "ABRACADABRA" 构建 Huffman 编码，观察最小堆合并过程</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => goto(0)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><SkipBack size={16}/></button>
          <button onClick={() => goto(idx - 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronLeft size={16}/></button>
          <button onClick={playing ? stop : () => setPlaying(true)} className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-1.5 text-sm font-medium">
            {playing ? <Pause size={14}/> : <Play size={14}/>}{playing ? '暂停' : '播放'}
          </button>
          <button onClick={() => goto(idx + 1)} className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300"><ChevronRight size={16}/></button>
          <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            速度: {SPEEDS.map((s, i) => (
              <button key={i} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded ${speedIdx === i ? 'bg-purple-600 text-white' : 'border border-slate-200 dark:border-slate-700'}`}>{s}x</button>
            ))}
          </div>
        </div>

        <div className="w-full h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div className="h-1.5 rounded-full bg-purple-500 transition-all" style={{ width: `${(idx / total) * 100}%` }}/>
        </div>

        {/* Frequency table */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">字符频率表（输入："ABRACADABRA"，总长 11）</div>
          <div className="flex flex-wrap gap-2">
            {(['A','B','R','C','D'] as const).map((ch, i) => {
              const freqs = [5,2,2,1,1]
              return (
                <div key={ch} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-white text-sm font-bold ${LEAF_COLORS[i]}`}>
                  <span>{ch}</span><span className="opacity-80">: {freqs[i]}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Current queue */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40 p-3">
          <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">当前优先队列（最小堆，按频率升序）</div>
          <div className="flex flex-wrap gap-2">
            {cur.queue.map((node, i) => (
              <div key={node.id + i} className={`flex flex-col items-center px-3 py-2 rounded-xl border-2 transition-all ${
                node.id === cur.mergedId
                  ? 'border-amber-400 bg-amber-50 dark:bg-amber-950/50 scale-105 shadow-lg'
                  : node.isLeaf
                    ? 'border-purple-300 dark:border-purple-700 bg-purple-50 dark:bg-purple-950/30'
                    : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900'
              }`}>
                <div className={`text-lg font-black ${node.isLeaf ? 'text-purple-700 dark:text-purple-300' : 'text-amber-700 dark:text-amber-300'}`}>
                  {node.id}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">freq={node.freq}</div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500">{node.isLeaf ? '叶' : '内部'}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Merge visualization */}
        {cur.mergedId && (
          <div className="rounded-xl border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/20 p-3">
            <div className="text-xs font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider mb-1">本轮合并操作</div>
            <MergeArrow leftId={cur.leftId} rightId={cur.rightId} mergedId={cur.mergedId} allNodes={cur.allNodes}/>
          </div>
        )}

        {/* Step description */}
        <div className="rounded-xl border border-purple-300 dark:border-purple-700 bg-purple-50 dark:bg-purple-950/30 px-4 py-3">
          <div className="font-bold text-purple-700 dark:text-purple-300 text-sm">{cur.desc}</div>
          <div className="mt-1 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{cur.detail}</div>
        </div>

        {/* Final encodings */}
        {cur.encodings && (
          <div className="rounded-xl border border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/30 p-3">
            <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-2">最终 Huffman 编码表</div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {[...cur.encodings.entries()].sort(([a],[b]) => a.localeCompare(b)).map(([ch, code], i) => {
                const freqs: Record<string,number> = {A:5,B:2,R:2,C:1,D:1}
                return (
                  <div key={ch} className={`rounded-lg p-2 text-white text-center ${LEAF_COLORS[i % LEAF_COLORS.length]}`}>
                    <div className="text-xl font-black">{ch}</div>
                    <div className="text-xs font-mono mt-0.5">{code}</div>
                    <div className="text-[10px] opacity-80">freq={freqs[ch]}  bits={code.length}</div>
                  </div>
                )
              })}
            </div>
            <div className="mt-2 text-xs text-emerald-600 dark:text-emerald-400">
              总 WPL = 5×{[...cur.encodings.entries()].find(([c])=>c==='A')?.[1]?.length}
              + 2×{[...cur.encodings.entries()].find(([c])=>c==='B')?.[1]?.length}
              + 2×{[...cur.encodings.entries()].find(([c])=>c==='R')?.[1]?.length}
              + 1×{[...cur.encodings.entries()].find(([c])=>c==='C')?.[1]?.length}
              + 1×{[...cur.encodings.entries()].find(([c])=>c==='D')?.[1]?.length}
              {' = ' + ((5*1)+(2*3)+(2*3)+(1*3)+(1*3))} bits（最优）
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

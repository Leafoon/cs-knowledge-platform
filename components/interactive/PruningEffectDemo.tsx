'use client'
import { useState, useMemo } from 'react'

interface TreeNode {
  path: number[]; remaining: number; children: TreeNode[]; pruned: boolean; found: boolean
}
const BASE_CANDS = [2, 3, 6, 7]

function buildTree(cands: number[], target: number, prune: boolean): { root: TreeNode; visited: number; solutions: number } {
  let visited = 0, solutions = 0
  function build(start: number, path: number[], rem: number): TreeNode {
    visited++
    const node: TreeNode = { path:[...path], remaining:rem, children:[], pruned:false, found:rem===0 }
    if (rem === 0) { solutions++; return node }
    for (let i=start; i<cands.length; i++) {
      const c = cands[i]
      if (prune && c > rem) { node.pruned = true; break }
      if (!prune && c > rem) continue
      node.children.push(build(i, [...path,c], rem-c))
    }
    return node
  }
  const sorted = prune ? [...cands].sort((a,b)=>a-b) : cands
  const root = build(0, [], target)
  return { root, visited, solutions }
}

function flatNodes(n: TreeNode): { node: TreeNode; depth: number; idx: number }[] {
  const res: {node:TreeNode;depth:number;idx:number}[] = []
  let cnt = 0
  function walk(n: TreeNode, d: number) {
    res.push({node:n,depth:d,idx:cnt++})
    n.children.forEach(c=>walk(c,d+1))
  }
  walk(n, 0); return res
}

const PRESETS = [
  {label:'target=7', target:7},
  {label:'target=9', target:9},
  {label:'target=12',target:12},
]

function NodeRow({node,depth}:{node:TreeNode;depth:number}) {
  const label = node.path.length===0 ? `∅` : `[${node.path.join('+')}]=${node.path.reduce((a,b)=>a+b,0)} (rem ${node.remaining})`
  return (
    <div style={{paddingLeft:depth*14}} className="flex items-center gap-1.5 py-0.5">
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
        node.found ? 'bg-emerald-500' : node.pruned ? 'bg-red-400' :
        node.children.length===0 && !node.found ? 'bg-slate-300 dark:bg-slate-600' : 'bg-slate-400'
      }`}/>
      <span className={`text-[11px] font-mono ${
        node.found ? 'text-emerald-600 dark:text-emerald-400 font-bold' :
        node.pruned ? 'text-red-500 dark:text-red-400 line-through opacity-60' :
        'text-slate-600 dark:text-slate-400'
      }`}>{label}</span>
      {node.found && <span className="text-[10px] text-emerald-500">✓</span>}
      {node.pruned && <span className="text-[10px] text-red-500">✂</span>}
    </div>
  )
}

export default function PruningEffectDemo() {
  const [pi, setPi] = useState(0)
  const target = PRESETS[pi].target
  const noPrune = useMemo(()=>buildTree(BASE_CANDS,target,false),[target])
  const pruned  = useMemo(()=>buildTree(BASE_CANDS,target,true),[target])
  const noNodes = flatNodes(noPrune.root)
  const prNodes = flatNodes(pruned.root)
  const saved   = noPrune.visited - pruned.visited
  const savePct = noPrune.visited > 0 ? Math.round(saved/noPrune.visited*100) : 0

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">✂️</span>
        <div>
          <h3 className="text-white font-bold text-base">剪枝效果对比</h3>
          <p className="text-indigo-200 text-xs">组合总和 [2,3,6,7] — 排序+break 剪枝 vs 无剪枝的搜索节点对比</p>
        </div>
        <div className="ml-auto flex gap-1.5">
          {PRESETS.map((p,i)=>(
            <button key={i} onClick={()=>setPi(i)}
              className={`px-3 py-1 rounded-lg text-xs font-medium border ${
                pi===i ? 'bg-white text-indigo-700 border-white' : 'bg-indigo-500/30 text-white border-white/30 hover:bg-indigo-500/50'
              }`}>{p.label}</button>
          ))}
        </div>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-3 gap-3 mb-5">
          <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-xl p-3 text-center">
            <p className="text-2xl font-black text-red-500 dark:text-red-400">{noPrune.visited}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">无剪枝节点</p>
          </div>
          <div className="bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800 rounded-xl p-3 text-center">
            <p className="text-2xl font-black text-indigo-600 dark:text-indigo-400">{savePct}%</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">剪枝省去</p>
          </div>
          <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 rounded-xl p-3 text-center">
            <p className="text-2xl font-black text-emerald-600 dark:text-emerald-400">{pruned.visited}</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">剪枝后节点</p>
          </div>
        </div>

        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-4">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-3">搜索节点数对比</p>
          <div className="space-y-3">
            {[{label:'无剪枝（未排序）',val:noPrune.visited,cls:'bg-red-400 dark:bg-red-600',pct:100,textCls:'text-red-500'},
              {label:'排序+break剪枝',val:pruned.visited,cls:'bg-emerald-500 dark:bg-emerald-600',pct:Math.max(8,pruned.visited/noPrune.visited*100),textCls:'text-emerald-600'}]
              .map(({label,val,cls,pct,textCls})=>(
              <div key={label}>
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                  <span>{label}</span>
                  <span className={`font-mono ${textCls}`}>{val} 节点</span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-5 overflow-hidden">
                  <div className={`${cls} h-5 rounded-full flex items-center pl-2 text-white text-xs font-bold transition-all`}
                    style={{width:`${pct}%`}}>{val}</div>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-indigo-600 dark:text-indigo-400 font-medium">
            剪枝少访问 <span className="font-bold">{saved}</span> 个节点，效率提升 <span className="font-bold">{savePct}%</span>
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm">💔</span>
              <p className="text-xs font-semibold text-red-600 dark:text-red-400">无剪枝搜索树</p>
            </div>
            <div className="max-h-56 overflow-y-auto space-y-0.5">
              {noNodes.map(({node,depth,idx})=>(
                <NodeRow key={idx} node={node} depth={depth}/>
              ))}
            </div>
          </div>
          <div className="bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm">✂️</span>
              <p className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">剪枝后搜索树</p>
            </div>
            <div className="max-h-56 overflow-y-auto space-y-0.5">
              {prNodes.map(({node,depth,idx})=>(
                <NodeRow key={idx} node={node} depth={depth}/>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800 rounded-xl p-3">
          <p className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 mb-1">💡 剪枝原理</p>
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            候选数组<strong>排序</strong>后，当 <code className="font-mono text-indigo-600 dark:text-indigo-300">candidates[i] &gt; remaining</code> 时可直接{' '}
            <code className="font-mono text-rose-500">break</code>——后续元素更大，必然超过目标。
            注意是 <code className="font-mono">break</code> 而非 <code className="font-mono">continue</code>！
          </p>
        </div>
      </div>
    </div>
  )
}

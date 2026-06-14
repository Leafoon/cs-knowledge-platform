'use client'
import { useState, useEffect } from 'react'

const ELEMS = ['1','2','3']
const W = 520, H = 300, LEVELS = 4, R = 18

function nodePos(level: number, pos: number): [number, number] {
  const x = W * (pos + 0.5) / Math.pow(2, level)
  return [x, 28 + level * 82]
}
function nodeLabel(level: number, pos: number): string {
  if (level === 0) return '∅'
  const bits: number[] = []
  let p = pos
  for (let l = level - 1; l >= 0; l--) { bits.unshift(p % 2); p = Math.floor(p / 2) }
  const inc = ELEMS.filter((_,i) => bits[i] === 1)
  return inc.length === 0 ? '∅' : '{' + inc.join(',') + '}'
}
function dfsOrder(): [number, number][] {
  const res: [number, number][] = []
  function dfs(l: number, p: number) {
    res.push([l, p])
    if (l < LEVELS - 1) { dfs(l+1, p*2); dfs(l+1, p*2+1) }
  }
  dfs(0, 0); return res
}
const DFS_ORDER = dfsOrder()

export default function SubsetEnumerationTree() {
  const [step, setStep]       = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed]     = useState(500)
  const maxStep    = DFS_ORDER.length
  const visitedSet = new Set(DFS_ORDER.slice(0, step+1).map(([l,p])=>`${l},${p}`))
  const curNode    = DFS_ORDER[step]
  useEffect(() => {
    if (!playing) return
    if (step >= maxStep-1) { setPlaying(false); return }
    const id = setTimeout(() => setStep(s=>s+1), speed)
    return () => clearTimeout(id)
  }, [playing, step, maxStep, speed])
  const subsets = DFS_ORDER.slice(0, step+1).filter(([l])=>l===LEVELS-1).map(([l,p])=>nodeLabel(l,p))

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      <div className="bg-gradient-to-r from-sky-500 to-cyan-600 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🌳</span>
        <div>
          <h3 className="text-white font-bold text-base">子集枚举二叉树</h3>
          <p className="text-sky-100 text-xs">集合 {'{1,2,3}'} — 每层决定「选」或「不选」，DFS 遍历全部 2³=8 个子集</p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-sky-100 text-[10px]">DFS 步骤</p>
          <p className="text-white font-bold text-sm">{step+1} / {maxStep}</p>
        </div>
      </div>
      <div className="p-5">
        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3 border border-slate-200 dark:border-slate-700 mb-4">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{maxHeight:300}}>
            {ELEMS.map((e,i) => (
              <text key={e} x="10" y={28+(i+1)*82+6} fontSize="11"
                fill={['#7c3aed','#2563eb','#0891b2'][i]} fontWeight="600">元素 {e}</text>
            ))}
            {DFS_ORDER.filter(([l])=>l<LEVELS-1).map(([l,p]) => {
              const [px,py]=nodePos(l,p), [lx,ly]=nodePos(l+1,p*2), [rx,ry]=nodePos(l+1,p*2+1)
              return (
                <g key={`e-${l}-${p}`}>
                  <line x1={px} y1={py+R} x2={lx} y2={ly-R}
                    stroke={visitedSet.has(`${l+1},${p*2}`)? '#0e7490':'#cbd5e1'}
                    strokeWidth={visitedSet.has(`${l+1},${p*2}`)?2:1}
                    opacity={visitedSet.has(`${l+1},${p*2}`)?1:0.35}/>
                  <line x1={px} y1={py+R} x2={rx} y2={ry-R}
                    stroke={visitedSet.has(`${l+1},${p*2+1}`)? '#7c3aed':'#cbd5e1'}
                    strokeWidth={visitedSet.has(`${l+1},${p*2+1}`)?2:1}
                    opacity={visitedSet.has(`${l+1},${p*2+1}`)?1:0.35}/>
                  {visitedSet.has(`${l},${p}`) && (
                    <>
                      <text x={(px+lx)/2-16} y={(py+ly)/2} fontSize="9" fill="#0e7490">不选</text>
                      <text x={(px+rx)/2+2}  y={(py+ry)/2} fontSize="9" fill="#7c3aed">选</text>
                    </>
                  )}
                </g>
              )
            })}
            {DFS_ORDER.map(([l,p]) => {
              const [x,y]=nodePos(l,p), key=`${l},${p}`
              const isCur=curNode[0]===l&&curNode[1]===p
              const isLeaf=l===LEVELS-1, vis=visitedSet.has(key)
              const label=nodeLabel(l,p)
              return (
                <g key={key}>
                  <circle cx={x} cy={y} r={R}
                    fill={!vis?'#f1f5f9':isCur?'#f59e0b':isLeaf?'#10b981':'#0ea5e9'}
                    stroke={!vis?'#cbd5e1':isCur?'#d97706':isLeaf?'#059669':'#0284c7'}
                    strokeWidth={isCur?2.5:1.5} opacity={vis?1:0.35}/>
                  <text x={x} y={y+4} textAnchor="middle" fontSize={label.length>5?8:9}
                    fontWeight="600" fill={!vis?'#94a3b8':isCur?'#92400e':'#fff'} opacity={vis?1:0.4}>
                    {label}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-xl p-3">
            <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-1">当前访问节点</p>
            <p className="text-sm font-bold text-slate-700 dark:text-slate-200">第 {curNode[0]} 层 · 第 {curNode[1]} 节点</p>
            <p className="text-sm text-amber-600 dark:text-amber-400 font-mono">= {nodeLabel(curNode[0],curNode[1])}</p>
            {curNode[0]===LEVELS-1 && <p className="mt-1 text-xs text-emerald-600 dark:text-emerald-400">✓ 叶节点！收集子集</p>}
          </div>
          <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-1.5">
              <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">已收集子集</p>
              <span className="text-xs bg-emerald-200 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300 px-1.5 py-0.5 rounded-full font-bold">{subsets.length}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {subsets.length===0
                ? <span className="text-xs text-slate-400 dark:text-slate-500">暂无（最终共 8 个）</span>
                : subsets.map((s,i)=>(
                  <span key={i} className="text-xs px-2 py-0.5 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 rounded-md font-mono font-semibold border border-emerald-200 dark:border-emerald-700">{s}</span>
                ))}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={()=>{setStep(0);setPlaying(false)}}
            className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700">
            ⏮ 重置
          </button>
          <button onClick={()=>setStep(s=>Math.max(0,s-1))} disabled={step===0}
            className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700 disabled:opacity-40">
            ◄ 上一步
          </button>
          <button onClick={()=>setPlaying(p=>!p)} disabled={step>=maxStep-1}
            className={`px-5 py-1.5 rounded-lg text-xs font-bold border shadow-sm disabled:opacity-40 ${
              playing ? 'bg-amber-500 border-amber-600 text-white' : 'bg-sky-500 border-sky-600 text-white hover:bg-sky-600'
            }`}>{playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <button onClick={()=>setStep(s=>Math.min(maxStep-1,s+1))} disabled={step>=maxStep-1}
            className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700 disabled:opacity-40">
            下一步 ►
          </button>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="text-xs text-slate-400">速度</span>
            {([['慢',800],['中',500],['快',200]] as [string,number][]).map(([l,ms])=>(
              <button key={l} onClick={()=>setSpeed(ms)}
                className={`px-2.5 py-1 rounded text-xs font-medium border ${
                  speed===ms ? 'bg-sky-500 text-white border-sky-600' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700'
                }`}>{l}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

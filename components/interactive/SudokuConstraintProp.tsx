'use client'
import { useState, useMemo } from 'react'

/* ─── 数独约束可视化 ───────────────────────────────────── */
const PUZZLE: (number | 0)[][] = [
  [5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],
  [8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],
  [0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9],
]

function getCands(board: (number|0)[][], r: number, c: number): number[] {
  if (board[r][c] !== 0) return []
  const used = new Set<number>()
  for (let i = 0; i < 9; i++) { used.add(board[r][i]); used.add(board[i][c]) }
  const br = Math.floor(r/3)*3, bc = Math.floor(c/3)*3
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) used.add(board[br+i][bc+j])
  return [1,2,3,4,5,6,7,8,9].filter(n => !used.has(n))
}
function constraintSet(r: number, c: number): Set<string> {
  const s = new Set<string>()
  for (let i = 0; i < 9; i++) { s.add(`${r},${i}`); s.add(`${i},${c}`) }
  const br = Math.floor(r/3)*3, bc = Math.floor(c/3)*3
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s.add(`${br+i},${bc+j}`)
  s.delete(`${r},${c}`); return s
}

export default function SudokuConstraintProp() {
  const [board, setBoard]   = useState<(number|0)[][]>(PUZZLE.map(r=>[...r]))
  const [sel, setSel]       = useState<[number,number]|null>(null)
  const [mode, setMode]     = useState<'constraint'|'count'>('constraint')

  const cSet   = sel ? constraintSet(sel[0], sel[1]) : new Set<string>()
  const cands  = sel ? getCands(board, sel[0], sel[1]) : []
  const selKey = sel ? `${sel[0]},${sel[1]}` : ''

  const counts = useMemo(() => {
    const m: number[][] = Array.from({length:9}, ()=>new Array(9).fill(0))
    for (let r=0; r<9; r++) for (let c=0; c<9; c++)
      m[r][c] = board[r][c]===0 ? getCands(board,r,c).length : -1
    return m
  }, [board])

  function click(r: number, c: number) {
    if (PUZZLE[r][c] !== 0) { setSel(null); return }
    setSel(p => (p && p[0]===r && p[1]===c) ? null : [r,c])
  }
  function place(n: number) {
    if (!sel) return
    const next = board.map(row=>[...row]); next[sel[0]][sel[1]] = n
    setBoard(next); setSel(null)
  }

  function cellCls(r: number, c: number) {
    const key = `${r},${c}`
    const isFixed = PUZZLE[r][c] !== 0
    const isEmpty = board[r][c] === 0
    const isSel   = key === selKey
    const isCon   = mode === 'constraint' && cSet.has(key) && !!sel
    const isZero  = isEmpty && counts[r][c] === 0
    const base = 'flex items-center justify-center text-sm transition-all duration-100 cursor-pointer select-none border border-slate-200 dark:border-slate-700 '
    if (isSel)    return base + 'bg-emerald-500 dark:bg-emerald-600 text-white font-bold ring-2 ring-emerald-700 z-10'
    if (isCon)    return base + 'bg-emerald-100 dark:bg-emerald-900/50'
    if (isZero)   return base + 'bg-red-50 dark:bg-red-950/30 text-red-400'
    if (isFixed)  return base + 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-bold'
    if (isEmpty)  return base + 'bg-white dark:bg-slate-900 text-slate-300 dark:text-slate-600 hover:bg-emerald-50 dark:hover:bg-emerald-950/20'
    return base + 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-600 dark:text-emerald-400 font-semibold'
  }
  function boxCls(r: number, c: number) {
    let b = ''
    if (r%3===0 && r>0) b += 'border-t-2 border-t-slate-400 dark:border-t-slate-500 '
    if (c%3===0 && c>0) b += 'border-l-2 border-l-slate-400 dark:border-l-slate-500 '
    return b
  }

  return (
    <div className="w-full max-w-2xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl font-black text-white">9</span>
        <div>
          <h3 className="text-white font-bold text-base">数独约束可视化</h3>
          <p className="text-emerald-100 text-xs">点击空白格查看行/列/宫约束与候选数字</p>
        </div>
        <div className="ml-auto flex gap-1.5">
          {(['constraint','count'] as const).map(m=>(
            <button key={m} onClick={()=>setMode(m)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all border ${
                mode===m ? 'bg-white text-emerald-700 border-white' : 'bg-emerald-500/30 text-white border-white/30 hover:bg-emerald-500/50'
              }`}>{m==='constraint' ? '约束区' : '候选数量'}</button>
          ))}
        </div>
      </div>

      <div className="p-5 flex flex-col md:flex-row gap-5 items-start">
        {/* Grid */}
        <div className="flex-shrink-0">
          <div className="border-2 border-slate-400 dark:border-slate-500 rounded-lg overflow-hidden"
            style={{display:'grid', gridTemplateColumns:'repeat(9,1fr)'}}>
            {Array.from({length:81}, (_,i) => {
              const r = Math.floor(i/9), c = i%9
              return (
              <div key={`${r}-${c}`} onClick={()=>click(r,c)}
                style={{width:38,height:38}}
                className={`${cellCls(r,c)} ${boxCls(r,c)} relative`}>
                {board[r][c]!==0 ? board[r][c]
                  : mode==='count' && !sel
                    ? <span className="text-[9px] text-slate-400 dark:text-slate-500">{counts[r][c]>0?counts[r][c]:'×'}</span>
                    : null}
              </div>
            )})}
          </div>
          <div className="mt-2 flex gap-2 text-[10px] flex-wrap">
            {[
              {cls:'bg-slate-100 dark:bg-slate-800', label:'就知'},
              {cls:'bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-300', label:'已填'},
              {cls:'bg-emerald-500', label:'选中'},
              {cls:'bg-emerald-100 dark:bg-emerald-900/50 border border-emerald-300', label:'约束区'},
              {cls:'bg-red-50 dark:bg-red-950/30 border border-red-300', label:'无候选'},
            ].map(({cls,label})=>(
              <div key={label} className="flex items-center gap-1">
                <div className={`w-3 h-3 rounded ${cls}`}/><span className="text-slate-500 dark:text-slate-400">{label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Info panel */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {sel ? (
            <>
              <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 rounded-xl p-4">
                <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-400 mb-1">
                  格 ({sel[0]}, {sel[1]}) · 第 {Math.floor(sel[0]/3)*3+Math.floor(sel[1]/3)+1} 宫
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  空白区为与此格同行/同列/同宫的 {cSet.size} 个约束格。
                </p>
              </div>
              {cands.length > 0 ? (
                <div className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <p className="text-xs font-semibold text-slate-600 dark:text-slate-300">候选数字</p>
                    <span className="text-xs bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400 px-2 py-0.5 rounded-full font-medium">{cands.length} 个</span>
                  </div>
                  <div className="flex gap-1 mb-2">
                    {[1,2,3,4,5,6,7,8,9].map(d=>(
                      <div key={d} className={`w-7 h-7 rounded text-xs font-bold flex items-center justify-center border ${
                        cands.includes(d)
                          ? 'bg-emerald-500 text-white border-emerald-600'
                          : 'bg-white dark:bg-slate-700 text-slate-200 dark:text-slate-600 border-slate-200 dark:border-slate-600'
                      }`}>{d}</div>
                    ))}
                  </div>
                  <p className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mb-3">
                    位掉码: {[1,2,3,4,5,6,7,8,9].map(d=>cands.includes(d)?'1':'0').join('')}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {cands.map(d=>(
                      <button key={d} onClick={()=>place(d)}
                        className="w-9 h-9 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-white font-bold text-sm transition-all active:scale-90 shadow-sm">
                        {d}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-800 rounded-xl p-4">
                  <p className="text-sm font-bold text-rose-600 dark:text-rose-400">⚠ 无候选数字！</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                    前向检查（Forward Checking）触发：当前填法导致某格无候选 → 必须回溯。
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200 mb-2">使用说明</p>
              <div className="space-y-2 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                <p>① 点击任意<span className="text-emerald-500"> 空白格</span>，高亮其约束区域</p>
                <p>② 选择候选数字可填入数字</p>
                <p>③ 切换“候选数量”模式看各空格剩余候选数</p>
                <p>④ 候选数 = 1 的格可直接填入（MRV 启发式）</p>
              </div>
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-2.5 border border-slate-200 dark:border-slate-700">
              <p className="text-lg font-black text-slate-700 dark:text-slate-200">{board.flat().filter(v=>v===0).length}</p>
              <p className="text-[10px] text-slate-400 dark:text-slate-500">剩余空格</p>
            </div>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-2.5 border border-slate-200 dark:border-slate-700">
              <p className="text-lg font-black text-emerald-600 dark:text-emerald-400">{board.flat().filter((v,i)=>v!==0&&PUZZLE[Math.floor(i/9)][i%9]===0).length}</p>
              <p className="text-[10px] text-slate-400 dark:text-slate-500">已填入</p>
            </div>
          </div>
          <button onClick={()=>{setBoard(PUZZLE.map(r=>[...r]));setSel(null)}}
            className="w-full py-2 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置谜题
          </button>
        </div>
      </div>
    </div>
  )
}

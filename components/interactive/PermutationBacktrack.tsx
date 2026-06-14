'use client'
import { useState, useEffect, useMemo } from 'react'

type PStep = {
  kind: 'select'|'deselect'|'found'
  path: number[]; used: boolean[]; idx: number; msg: string; results: number[][]
}

function genSteps(nums: number[]): PStep[] {
  const n = nums.length, steps: PStep[] = []
  const path: number[] = [], used: boolean[] = new Array(n).fill(false)
  const results: number[][] = []
  function bt() {
    if (path.length === n) {
      results.push([...path])
      steps.push({ kind:'found', path:[...path], used:[...used], idx:-1,
        msg:`✓ 找到排列 [${path.join(',')}]`, results:results.map(r=>[...r]) })
      return
    }
    for (let i=0; i<n; i++) {
      if (used[i]) continue
      used[i]=true; path.push(nums[i])
      steps.push({ kind:'select', path:[...path], used:[...used], idx:i,
        msg:`选择 ${nums[i]} (i=${i}) 并加入路径`, results:results.map(r=>[...r]) })
      bt()
      path.pop(); used[i]=false
      steps.push({ kind:'deselect', path:[...path], used:[...used], idx:i,
        msg:`撤销 ${nums[i]} (i=${i})，回溯`, results:results.map(r=>[...r]) })
    }
  }
  bt(); return steps
}

const ECOLS = ['bg-violet-500','bg-blue-500','bg-emerald-500','bg-orange-500']
const ELIGHT= [
  'bg-violet-100 dark:bg-violet-900/40 border-violet-300 dark:border-violet-700 text-violet-700 dark:text-violet-300',
  'bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300',
  'bg-emerald-100 dark:bg-emerald-900/40 border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300',
  'bg-orange-100 dark:bg-orange-900/40 border-orange-300 dark:border-orange-700 text-orange-700 dark:text-orange-300',
]

export default function PermutationBacktrack() {
  const [nums, setNums]       = useState([1,2,3])
  const [step, setStep]       = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed]     = useState(400)
  const steps   = useMemo(() => genSteps(nums), [nums])
  const cur     = steps[Math.min(step, steps.length-1)]
  useEffect(()=>{ setStep(0); setPlaying(false) }, [nums])
  useEffect(()=>{
    if (!playing) return
    if (step >= steps.length-1) { setPlaying(false); return }
    const id = setTimeout(()=>setStep(s=>s+1), speed)
    return ()=>clearTimeout(id)
  }, [playing, step, steps.length, speed])
  const pct = steps.length > 1 ? Math.round(step/(steps.length-1)*100) : 0
  const fact  = nums.reduce((a,_,i)=>a*(i+1),1)

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-lg bg-white dark:bg-slate-900">
      <div className="bg-gradient-to-r from-rose-500 to-pink-600 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🔄</span>
        <div>
          <h3 className="text-white font-bold text-base">全排列回溯动画</h3>
          <p className="text-rose-100 text-xs">used[] 标记已选元素，递归 → 选择 → 撤销</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-rose-100 text-xs">n =</span>
          {[3,4].map(v=>(
            <button key={v} onClick={()=>setNums(Array.from({length:v},(_,i)=>i+1))}
              className={`w-8 h-8 rounded-lg text-sm font-bold ${
                nums.length===v ? 'bg-white text-rose-600 shadow' : 'bg-rose-400/40 text-white hover:bg-rose-400/60'
              }`}>{v}</button>
          ))}
        </div>
      </div>

      <div className="p-5 space-y-4">
        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div>
              <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2">nums[] 输入</p>
              <div className="flex gap-2">
                {nums.map((n,i)=>(
                  <div key={i} className={`w-10 h-10 rounded-lg flex flex-col items-center justify-center text-white font-bold text-sm shadow ${
                    cur.used[i] ? (ECOLS[i]??ECOLS[0]) : 'bg-slate-300 dark:bg-slate-600'
                  }`}>
                    <span>{n}</span><span className="text-[8px] opacity-70">i={i}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2">used[] 标记</p>
              <div className="flex gap-2">
                {cur.used.map((u,i)=>(
                  <div key={i} className={`w-10 h-10 rounded-lg flex flex-col items-center justify-center text-xs font-bold border transition-all ${
                    u ? (ELIGHT[i]??ELIGHT[0]) : 'bg-white dark:bg-slate-700 text-slate-400 dark:text-slate-500 border-slate-200 dark:border-slate-600'
                  }`}>
                    <span>{u?'T':'F'}</span><span className="text-[8px] opacity-60">[{i}]</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2">当前路径 path[]</p>
            <div className="flex gap-2 min-h-[44px] items-center">
              {cur.path.length===0
                ? <span className="text-xs text-slate-300 dark:text-slate-600">空</span>
                : cur.path.map((v,j)=>{
                  const oi=nums.indexOf(v)
                  return <div key={j} className={`w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm border-2 ${ELIGHT[oi]??ELIGHT[0]}`}>{v}</div>
                })}
              {cur.path.length < nums.length && (
                <div className="w-10 h-10 rounded-lg border-2 border-dashed border-slate-300 dark:border-slate-600 flex items-center justify-center">
                  <span className="text-slate-300 dark:text-slate-600 text-lg">?</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className={`rounded-xl border p-3.5 transition-all ${
          cur.kind==='found'    ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800' :
          cur.kind==='select'   ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800' :
                                  'bg-rose-50 dark:bg-rose-950/30 border-rose-200 dark:border-rose-800'
        }`}>
          <div className="flex items-center gap-2">
            <span className="text-base">{cur.kind==='found'?'✓':cur.kind==='select'?'+':'↩'}</span>
            <span className={`text-xs font-bold uppercase ${
              cur.kind==='found'   ?'text-emerald-600 dark:text-emerald-400':
              cur.kind==='select'  ?'text-blue-600 dark:text-blue-400':
                                    'text-rose-600 dark:text-rose-400'
            }`}>{cur.kind}</span>
          </div>
          <p className="text-sm font-medium text-slate-700 dark:text-slate-200 mt-1">{cur.msg}</p>
        </div>

        <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800 rounded-xl p-3">
          <div className="flex items-center gap-2 mb-2">
            <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">已找到排列</p>
            <span className="text-xs bg-emerald-200 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300 px-1.5 py-0.5 rounded-full font-bold">{cur.results.length} / {fact}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {cur.results.length===0
              ? <span className="text-xs text-slate-400 dark:text-slate-500">未找到排列…</span>
              : cur.results.map((r,i)=>(
                <span key={i} className="text-xs font-mono px-2 py-0.5 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300 rounded border border-emerald-200 dark:border-emerald-700">[{r.join(',')}]</span>
              ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 w-14">步 {step+1}/{steps.length}</span>
            <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
              <div className="bg-rose-500 h-1.5 rounded-full transition-all" style={{width:`${pct}%`}}/>
            </div>
            <span className="text-xs text-slate-400 w-8">{pct}%</span>
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
            <button onClick={()=>setPlaying(p=>!p)} disabled={step>=steps.length-1}
              className={`px-5 py-1.5 rounded-lg text-xs font-bold border shadow-sm disabled:opacity-40 ${
                playing ? 'bg-amber-500 border-amber-600 text-white' : 'bg-rose-500 border-rose-600 text-white hover:bg-rose-600'
              }`}>{playing ? '⏸ 暂停' : '▶ 播放'}</button>
            <button onClick={()=>setStep(s=>Math.min(steps.length-1,s+1))} disabled={step>=steps.length-1}
              className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-700 disabled:opacity-40">
              下一步 ►
            </button>
            <div className="ml-auto flex items-center gap-1.5">
              <span className="text-xs text-slate-400">速度</span>
              {([['慢',700],['中',400],['快',150]] as [string,number][]).map(([l,ms])=>(
                <button key={l} onClick={()=>setSpeed(ms)}
                  className={`px-2.5 py-1 rounded text-xs font-medium border ${
                    speed===ms ? 'bg-rose-500 text-white border-rose-600' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-700'
                  }`}>{l}</button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

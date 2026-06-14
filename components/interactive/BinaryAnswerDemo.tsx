'use client';
import React, { useState, useMemo } from 'react';

// ====== 二分答案可视化 ======

type ProblemMode = 'koko' | 'split' | 'custom';

// Koko 吃香蕉
function kokoCheck(piles: number[], h: number, speed: number): boolean {
  let hours = 0;
  for (const p of piles) hours += Math.ceil(p / speed);
  return hours <= h;
}

// 分割数组（最大和最小化）
function splitCheck(nums: number[], k: number, maxSum: number): boolean {
  let segs = 1, cur = 0;
  for (const x of nums) {
    if (cur + x > maxSum) { segs++; cur = x; if (segs > k) return false; }
    else cur += x;
  }
  return true;
}

interface ProblemDef {
  name: string;
  lo: number;
  hi: number;
  check: (x: number) => boolean;
  labelLo: string;
  labelHi: string;
  desc: string;
  xLabel: string;
}

const DEFAULT_PILES = [3, 6, 7, 11];
const DEFAULT_H = 8;
const DEFAULT_NUMS = [7, 2, 5, 10, 8];
const DEFAULT_K = 2;

export default function BinaryAnswerDemo() {
  const [mode, setMode] = useState<ProblemMode>('koko');
  const [pilesStr, setPilesStr] = useState(DEFAULT_PILES.join(', '));
  const [hStr, setHStr] = useState(String(DEFAULT_H));
  const [numsStr, setNumsStr] = useState(DEFAULT_NUMS.join(', '));
  const [kStr, setKStr] = useState(String(DEFAULT_K));
  const [customLoStr, setCustomLoStr] = useState('1');
  const [customHiStr, setCustomHiStr] = useState('20');
  const [customThreshStr, setCustomThreshStr] = useState('12');
  const [stepIdx, setStepIdx] = useState(0);

  const piles = pilesStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  const h = parseInt(hStr) || DEFAULT_H;
  const nums = numsStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  const k = parseInt(kStr) || DEFAULT_K;

  const problem = useMemo<ProblemDef>(() => {
    if (mode === 'koko') {
      const lo = 1, hi = Math.max(...piles, 1);
      return { name: 'Koko 吃香蕉（LeetCode 875）', lo, hi, check: x => kokoCheck(piles, h, x), labelLo: '最慢 1/h', labelHi: `最快 ${hi}/h`, desc: `${piles.length} 堆香蕉 [${piles.join(',')}]，${h} 小时内吃完，最小速度？`, xLabel: '速度 k' };
    } else if (mode === 'split') {
      const lo = Math.max(...nums, 0), hi = nums.reduce((a,b)=>a+b,0);
      return { name: '分割数组（LeetCode 410）', lo, hi, check: x => splitCheck(nums, k, x), labelLo: `最小 ${lo}`, labelHi: `最大 ${hi}`, desc: `数组 [${nums.join(',')}] 分成 ${k} 段，最小化最大段和？`, xLabel: '最大段和 m' };
    } else {
      const lo = parseInt(customLoStr) || 1;
      const hi = parseInt(customHiStr) || 20;
      const thresh = parseInt(customThreshStr) || 12;
      return { name: '自定义问题', lo, hi, check: x => x >= thresh, labelLo: String(lo), labelHi: String(hi), desc: `在 [${lo}, ${hi}] 上找第一个满足 check(x) 的 x（阈值 ≥ ${thresh}）`, xLabel: 'x 值' };
    }
  }, [mode, piles, h, nums, k, customLoStr, customHiStr, customThreshStr]);

  // Build steps
  const steps = useMemo(() => {
    const result: { lo: number; hi: number; mid: number; checkResult: boolean; desc: string }[] = [];
    let lo = problem.lo, hi = problem.hi;
    const safeMax = 60;
    let iter = 0;
    result.push({ lo, hi, mid: -1, checkResult: false, desc: `初始化：在 [${lo}, ${hi}] 上二分答案` });
    while (lo < hi && iter < safeMax) {
      iter++;
      const mid = lo + Math.floor((hi - lo) / 2);
      const ok = problem.check(mid);
      if (ok) {
        result.push({ lo, hi, mid, checkResult: true, desc: `check(${mid}) = ✅，可行，尝试更小：hi = mid = ${mid}` });
        hi = mid;
      } else {
        result.push({ lo, hi, mid, checkResult: false, desc: `check(${mid}) = ❌，不可行，必须更大：lo = mid+1 = ${mid+1}` });
        lo = mid + 1;
      }
    }
    result.push({ lo, hi, mid: lo, checkResult: problem.check(lo), desc: `lo = hi = ${lo}，答案 = ${lo}` });
    return result;
  }, [problem]);

  const safeIdx = Math.min(stepIdx, steps.length - 1);
  const step = steps[safeIdx];
  const reset = () => setStepIdx(0);

  const totalRange = problem.hi - problem.lo;
  const barW = Math.floor(Math.min(100, Math.max(4, 600 / (totalRange + 1))));
  const displayRange = Math.min(totalRange + 1, 60); // 最多显示 60 个格子

  const getBarColor = (x: number) => {
    if (!step) return '#3f3f46';
    const { lo, hi, mid, checkResult } = step;
    if (x < lo || x > hi) {
      // 已排除区域
      const wasCheck = problem.check(x);
      return wasCheck ? '#065f46' : '#7c2d12';
    }
    if (x === mid) return checkResult ? '#10b981' : '#f87171';
    return '#4f46e5'; // 当前搜索范围
  };

  const rangeStart = problem.lo;
  const rangeValues = Array.from({ length: Math.min(displayRange, 60) }, (_, i) => rangeStart + i);

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">二分答案可视化</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">Binary Search on Answer — 对答案空间直接二分</p>
      </div>

      <div className="p-6 space-y-6">
        {/* 模式切换 */}
        <div className="flex flex-wrap gap-2">
          {([['koko','Koko 香蕉 (LC 875)'],['split','数组分割 (LC 410)'],['custom','自定义问题']] as [ProblemMode,string][]).map(([m,l])=>(
            <button key={m} onClick={() => { setMode(m); reset(); }}
              className={`px-4 py-2 text-sm rounded-lg font-medium transition-colors ${mode === m ? 'bg-sky-600 text-white' : 'bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 text-slate-700 dark:text-zinc-200'}`}>{l}</button>
          ))}
        </div>

        {/* 问题参数 */}
        {mode === 'koko' && (
          <div className="flex gap-3 flex-wrap items-center">
            <div className="flex items-center gap-2"><span className="text-sm text-slate-600 dark:text-zinc-400">香蕉堆：</span>
              <input value={pilesStr} onChange={e => { setPilesStr(e.target.value); reset(); }} className="w-44 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" /></div>
            <div className="flex items-center gap-2"><span className="text-sm text-slate-600 dark:text-zinc-400">小时 H:</span>
              <input value={hStr} onChange={e => { setHStr(e.target.value); reset(); }} className="w-16 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" /></div>
          </div>
        )}
        {mode === 'split' && (
          <div className="flex gap-3 flex-wrap items-center">
            <div className="flex items-center gap-2"><span className="text-sm text-slate-600 dark:text-zinc-400">数组：</span>
              <input value={numsStr} onChange={e => { setNumsStr(e.target.value); reset(); }} className="w-44 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" /></div>
            <div className="flex items-center gap-2"><span className="text-sm text-slate-600 dark:text-zinc-400">段数 k:</span>
              <input value={kStr} onChange={e => { setKStr(e.target.value); reset(); }} className="w-16 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" /></div>
          </div>
        )}
        {mode === 'custom' && (
          <div className="flex gap-3 flex-wrap items-center">
            {[['Lo', customLoStr, setCustomLoStr],['Hi', customHiStr, setCustomHiStr],['阈值', customThreshStr, setCustomThreshStr]].map(([l,v,s])=>(
              <div key={String(l)} className="flex items-center gap-2"><span className="text-sm text-slate-600 dark:text-zinc-400">{String(l)}:</span>
                <input value={String(v)} onChange={e => { (s as (v:string)=>void)(e.target.value); reset(); }} className="w-20 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" /></div>
            ))}
          </div>
        )}

        {/* 问题描述 */}
        <div className="bg-slate-100 dark:bg-zinc-800/60 border border-slate-200 dark:border-zinc-700 rounded-xl px-5 py-4 text-sm">
          <div className="font-bold text-sky-600 dark:text-sky-400 mb-1">{problem.name}</div>
          <div className="text-slate-600 dark:text-zinc-300">{problem.desc}</div>
        </div>

        {/* 控制 */}
        <div className="flex gap-3 flex-wrap items-center">
          <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一步</button>
          <button onClick={() => setStepIdx(s => Math.min(steps.length - 1, s + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-4 py-2 text-sm bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg text-white transition-colors">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)}
            className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white transition-colors">跳到答案</button>
          <button onClick={reset} className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺</button>
          <span className="text-sm text-slate-500 dark:text-zinc-400">步骤 {safeIdx+1} / {steps.length}</span>
        </div>

        {/* 步骤描述 */}
        <div className={`px-4 py-3 rounded-xl text-sm font-medium ${step.mid === -1 ? 'bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 text-slate-600 dark:text-zinc-300' : step.checkResult ? 'bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300' : 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300'}`}>
          {step.mid !== -1 && <span className="font-bold">{step.checkResult ? '✅ check 满足' : '❌ check 不满足'}</span>}{' '}{step.desc}
        </div>

        {/* 数值范围可视化（数轴） */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-5 overflow-x-auto">
          <div className="text-sm font-medium text-slate-600 dark:text-zinc-400 mb-3">{problem.xLabel}（答案搜索空间）：</div>
          <div className="flex gap-0.5 flex-nowrap">
            {rangeValues.map(x => {
              const color = getBarColor(x);
              const isMid = x === step.mid;
              const inRange = x >= (step.lo ?? 0) && x <= (step.hi ?? 0);
              return (
                <div key={x} className="flex flex-col items-center transition-all duration-200" style={{ minWidth: Math.max(barW, 18) }}>
                  <div className="rounded text-center transition-all duration-200 flex items-center justify-center" style={{
                    width: Math.max(barW - 2, 16),
                    height: 34,
                    backgroundColor: color,
                    opacity: inRange ? 1 : 0.5,
                    outline: isMid ? '2px solid #fff' : 'none',
                    fontSize: 10,
                    color: '#fff',
                    fontWeight: isMid ? 700 : 400,
                  }}>
                    {barW >= 20 ? x : (x % 5 === 0 ? x : '')}
                  </div>
                  {isMid && <div className="text-yellow-500 dark:text-yellow-400 text-center" style={{fontSize:10}}>mid</div>}
                </div>
              );
            })}
            {totalRange >= 60 && <div className="self-center text-slate-400 dark:text-zinc-500 text-sm ml-3">...（范围过大，仅显示前60个）</div>}
          </div>
          <div className="flex justify-between text-sm text-slate-500 dark:text-zinc-500 mt-2">
            <span>{problem.labelLo}</span>
            <span>{problem.labelHi}</span>
          </div>
          <div className="flex flex-wrap gap-4 mt-3 text-sm text-slate-600 dark:text-zinc-400">
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'#4f46e5'}} />当前搜索范围</span>
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'#10b981'}} />check ✅ 可行</span>
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'#f87171'}} />mid 处 check ❌</span>
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'#065f46'}} />排除（可行）</span>
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded" style={{background:'#7c2d12'}} />排除（不可行）</span>
          </div>
        </div>

        {/* 当前状态 */}
        {step && step.mid >= 0 && (
          <div className="grid grid-cols-3 gap-3 text-sm text-center">
            {[['lo', step.lo, '#60a5fa'],['mid', step.mid, '#fbbf24'],['hi', step.hi, '#f97316']].map(([n,v,c])=>(
              <div key={String(n)} className="bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-xl p-4">
                <div className="font-bold text-base mb-1" style={{color:String(c)}}>{String(n)}</div>
                <div className="text-slate-800 dark:text-white font-mono font-bold text-xl">{String(v)}</div>
              </div>
            ))}
          </div>
        )}

        {/* 最终答案 */}
        {safeIdx >= steps.length - 1 && (
          <div className="bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700 rounded-xl p-5 text-center">
            <div className="text-sm text-slate-600 dark:text-zinc-400 mb-2">最优答案（{problem.xLabel}）：</div>
            <div className="text-4xl font-bold text-emerald-600 dark:text-emerald-400">{step.lo}</div>
            <div className="text-sm text-slate-500 dark:text-zinc-500 mt-2">共 {steps.length - 2} 次 check 调用，二分 O(log n) 次</div>
          </div>
        )}
      </div>
    </div>
  );
}

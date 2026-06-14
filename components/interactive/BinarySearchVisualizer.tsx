'use client';
import React, { useState } from 'react';

// ====== 二分搜索可视化 ======

interface BinStep {
  lo: number;
  hi: number;
  mid: number;
  relation: 'equal' | 'less' | 'greater' | 'init';
  desc: string;
}

function buildBinarySearchSteps(arr: number[], target: number): { steps: BinStep[], result: number } {
  const steps: BinStep[] = [];
  let lo = 0, hi = arr.length - 1;
  let result = -1;

  steps.push({ lo, hi, mid: Math.floor((lo + hi) / 2), relation: 'init', desc: `初始化：lo=${lo}，hi=${hi}，搜索 target=${target}` });

  while (lo <= hi) {
    const mid = lo + Math.floor((hi - lo) / 2);
    if (arr[mid] === target) {
      steps.push({ lo, hi, mid, relation: 'equal', desc: `A[mid=${mid}]=${arr[mid]} == target，找到！` });
      result = mid;
      break;
    } else if (arr[mid] < target) {
      steps.push({ lo, hi, mid, relation: 'less', desc: `A[mid=${mid}]=${arr[mid]} < target(${target})，排除左半，lo = mid+1 = ${mid+1}` });
      lo = mid + 1;
    } else {
      steps.push({ lo, hi, mid, relation: 'greater', desc: `A[mid=${mid}]=${arr[mid]} > target(${target})，排除右半，hi = mid-1 = ${mid-1}` });
      hi = mid - 1;
    }
  }

  if (result === -1 && lo > hi) {
    steps.push({ lo, hi, mid: -1, relation: 'init', desc: `lo(${lo}) > hi(${hi})，搜索区间为空，target=${target} 不存在，返回 -1` });
  }

  return { steps, result };
}

const DEFAULT_ARR = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

export default function BinarySearchVisualizer() {
  const [arrStr, setArrStr] = useState(DEFAULT_ARR.join(', '));
  const [targetStr, setTargetStr] = useState('7');
  const [stepIdx, setStepIdx] = useState(0);

  // 解析并排序
  const arr = [...new Set(arrStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)))].sort((a, b) => a - b).slice(0, 16);
  const target = parseInt(targetStr) || 7;
  const { steps, result } = buildBinarySearchSteps(arr, target);
  const safeIdx = Math.min(stepIdx, steps.length - 1);
  const step = steps[safeIdx];

  const reset = () => setStepIdx(0);

  const barColor = (idx: number) => {
    if (!step) return '#3f3f46';
    const { lo, hi, mid, relation } = step;
    if (idx < lo || idx > hi) return '#1c1917'; // 排除区域
    if (idx === mid) {
      if (relation === 'equal') return '#10b981';
      if (relation === 'less') return '#f87171';
      if (relation === 'greater') return '#f87171';
      return '#f59e0b';
    }
    return '#4f46e5'; // 搜索区间
  };

  const barBorder = (idx: number) => {
    if (!step) return 'none';
    if (idx === step.lo) return '2px solid #60a5fa';
    if (idx === step.hi) return '2px solid #f97316';
    return 'none';
  };

  const PRESETS = [
    { arr: DEFAULT_ARR, target: 7 },
    { arr: DEFAULT_ARR, target: 1 },
    { arr: DEFAULT_ARR, target: 19 },
    { arr: DEFAULT_ARR, target: 6 },
    { arr: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], target: 14 },
  ];

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">二分搜索可视化</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">Binary Search — O(log n) 的信息论最优搜索算法</p>
      </div>

      <div className="p-6 space-y-6">
        {/* 预设 */}
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setArrStr(p.arr.join(', ')); setTargetStr(String(p.target)); reset(); }}
              className="px-3 py-1.5 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">
              查找 {p.target} {!p.arr.includes(p.target) ? '（不存在）' : ''}
            </button>
          ))}
        </div>

        {/* 输入 */}
        <div className="flex gap-3 flex-wrap">
          <input value={arrStr} onChange={e => { setArrStr(e.target.value); reset(); }}
            className="flex-1 min-w-48 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-4 py-2.5 text-sm font-mono text-slate-800 dark:text-zinc-100 placeholder:text-slate-400 dark:placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
            placeholder="有序数组（自动排序）" />
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500 dark:text-zinc-400">目标：</span>
            <input value={targetStr} onChange={e => { setTargetStr(e.target.value); reset(); }}
              className="w-20 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2.5 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" />
          </div>
        </div>

        {/* 控制 */}
        <div className="flex gap-3 flex-wrap items-center">
          <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">← 上一步</button>
          <button onClick={() => setStepIdx(s => Math.min(steps.length - 1, s + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-4 py-2 text-sm bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg text-white transition-colors">下一步 →</button>
          <button onClick={() => setStepIdx(steps.length - 1)}
            className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white transition-colors">跳到结果</button>
          <button onClick={reset} className="px-4 py-2 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺ 重置</button>
          <span className="text-sm text-slate-500 dark:text-zinc-400">步骤 {safeIdx + 1} / {steps.length}</span>
        </div>

        {/* 步骤描述 */}
        {step && (
          <div className={`px-4 py-3 rounded-xl text-sm font-mono ${
            step.relation === 'equal' ? 'bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300' :
            step.relation === 'less' || step.relation === 'greater' ? 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300' :
            'bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 text-slate-700 dark:text-zinc-300'
          }`}>
            {step.desc}
          </div>
        )}

        {/* 主要可视化 */}
        <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-5 space-y-4">
          {/* 指针行 */}
          {step && step.relation !== 'init' && (
            <div className="flex gap-1.5 justify-start">
              {arr.map((_, idx) => (
                <div key={idx} className="text-center" style={{ minWidth: 46, fontSize: 11 }}>
                  {idx === step.lo && <span className="text-blue-500 dark:text-blue-400 font-bold">lo</span>}
                  {idx === step.hi && idx !== step.lo && <span className="text-orange-500 dark:text-orange-400 font-bold">hi</span>}
                  {idx === step.mid && idx !== step.lo && idx !== step.hi && <span className="text-yellow-500 dark:text-yellow-400 font-bold">mid</span>}
                  {idx === step.mid && idx === step.lo && <span className="text-yellow-500 dark:text-yellow-400 font-bold">lo/mid</span>}
                  {idx === step.mid && idx === step.hi && <span className="text-yellow-500 dark:text-yellow-400 font-bold">mid/hi</span>}
                </div>
              ))}
            </div>
          )}

          {/* 数组格子 */}
          <div className="flex gap-1.5">
            {arr.map((val, idx) => (
              <div key={idx} className="flex flex-col items-center transition-all duration-300" style={{ minWidth: 46 }}>
                <div className="rounded-lg flex items-center justify-center font-mono font-bold transition-all duration-300" style={{
                  width: 42, height: 42,
                  backgroundColor: barColor(idx),
                  border: barBorder(idx),
                  opacity: step && (idx < step.lo || idx > step.hi) ? 0.22 : 1,
                  color: barColor(idx) === '#1c1917' ? '#52525b' : '#fff',
                  fontSize: 15,
                }}>
                  {val}
                </div>
                <div className="text-center text-slate-400 dark:text-zinc-600 mt-1" style={{ fontSize: 11 }}>[{idx}]</div>
              </div>
            ))}
          </div>

          {/* 图例 */}
          <div className="flex flex-wrap gap-4 pt-1 text-sm text-slate-600 dark:text-zinc-400">
            {[['搜索区间','#4f46e5'],['mid 指针','#f59e0b'],['找到','#10b981'],['不匹配','#f87171'],['已排除','#1c1917']].map(([l,c])=>(
              <span key={l} className="flex items-center gap-1.5">
                <span className="w-3.5 h-3.5 rounded" style={{background:c}} />{l}
              </span>
            ))}
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded border-2 border-blue-400" style={{background:'transparent'}} />lo</span>
            <span className="flex items-center gap-1.5"><span className="w-3.5 h-3.5 rounded border-2 border-orange-400" style={{background:'transparent'}} />hi</span>
          </div>
        </div>

        {/* 当前指针值 */}
        {step && step.mid >= 0 && (
          <div className="grid grid-cols-3 gap-3 text-center">
            {[['lo', step.lo, arr[step.lo], '#60a5fa'],['mid', step.mid, arr[step.mid], '#fbbf24'],['hi', step.hi, arr[step.hi ?? 0], '#f97316']].map(([name, i, v, c])=>(
              <div key={String(name)} className="bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-xl p-4">
                <div className="text-sm font-bold mb-1" style={{color:String(c)}}>{String(name)}</div>
                <div className="text-xs text-slate-500 dark:text-zinc-400 mb-1">索引 {String(i)}</div>
                <div className="text-slate-800 dark:text-white font-mono font-bold text-xl">{Number(i) >= 0 && Number(i) < arr.length ? String(v) : '—'}</div>
              </div>
            ))}
          </div>
        )}

        {/* 结果 */}
        {safeIdx === steps.length - 1 && (
          <div className={`rounded-xl p-4 text-center text-sm ${result >= 0 ? 'bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-700' : 'bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700'}`}>
            {result >= 0
              ? <span className="text-emerald-700 dark:text-emerald-300">✅ 找到 <span className="font-bold">{target}</span> 在索引 <span className="font-bold">{result}</span>，共 {steps.length - 1} 次迭代（理论上界 {Math.ceil(Math.log2(arr.length + 1))} 次）</span>
              : <span className="text-slate-600 dark:text-zinc-400">❌ <span className="font-bold text-red-500">{target}</span> 不存在，共 {steps.length - 1} 次迭代</span>
            }
          </div>
        )}

        {/* 复杂度 */}
        <div className="bg-slate-100 dark:bg-zinc-800 border border-slate-200 dark:border-zinc-700 rounded-lg px-4 py-3 text-sm text-slate-500 dark:text-zinc-400">
          时间 <span className="text-green-600 dark:text-green-400 font-medium">O(log n)</span>，空间 <span className="text-yellow-600 dark:text-yellow-400 font-medium">O(1)</span>。
          n={arr.length} 时最多 <span className="text-blue-600 dark:text-blue-300 font-medium">{Math.ceil(Math.log2(arr.length + 1))}</span> 次比较（信息论最优）。
        </div>
      </div>
    </div>
  );
}

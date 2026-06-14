'use client';
import React, { useState } from 'react';

// ====== 二分边界模板对比（lower_bound vs upper_bound）======

interface BoundStep {
  lo: number;
  hi: number;
  mid: number;
  condition: boolean;  // condition true → hi=mid, false → lo=mid+1
  desc: string;
}

type BoundMode = 'lower' | 'upper';

function buildBoundSteps(arr: number[], target: number, mode: BoundMode): { steps: BoundStep[], result: number } {
  const steps: BoundStep[] = [];
  let lo = 0, hi = arr.length;

  while (lo < hi) {
    const mid = lo + Math.floor((hi - lo) / 2);
    const condition = mode === 'lower' ? arr[mid] < target : arr[mid] <= target;
    const condStr = mode === 'lower'
      ? `A[${mid}]=${arr[mid]} < target(${target}) → ${condition}`
      : `A[${mid}]=${arr[mid]} ≤ target(${target}) → ${condition}`;

    if (condition) {
      steps.push({ lo, hi, mid, condition, desc: `${condStr}：lo = mid+1 = ${mid+1}（mid 处不满足，左移下界）` });
      lo = mid + 1;
    } else {
      steps.push({ lo, hi, mid, condition, desc: `${condStr}：hi = mid = ${mid}（mid 处满足，收缩上界）` });
      hi = mid;
    }
  }

  steps.push({ lo, hi, mid: lo, condition: false, desc: lo === arr.length
    ? `lo=${lo}（所有元素 ${mode === 'lower' ? '<' : '≤'} target，返回 n=${arr.length}）`
    : `lo=${lo}=hi，结果：${mode === 'lower' ? `第一个 ≥ ${target}` : `第一个 > ${target}`} 在索引 ${lo}（值=${arr[lo] ?? '越界'}）` });

  return { steps, result: lo };
}

const DEFAULT_ARR = [1, 3, 3, 5, 5, 5, 7, 9];

export default function BinarySearchBoundaryTemplate() {
  const [arrStr, setArrStr] = useState(DEFAULT_ARR.join(', '));
  const [targetStr, setTargetStr] = useState('5');
  const [stepL, setStepL] = useState(0);
  const [stepU, setStepU] = useState(0);

  const arr = arrStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n)).sort((a, b) => a - b).slice(0, 14);
  const target = parseInt(targetStr) || 5;

  const { steps: lSteps, result: lResult } = buildBoundSteps(arr, target, 'lower');
  const { steps: uSteps, result: uResult } = buildBoundSteps(arr, target, 'upper');

  const safeSL = Math.min(stepL, lSteps.length - 1);
  const safeSU = Math.min(stepU, uSteps.length - 1);
  const sL = lSteps[safeSL];
  const sU = uSteps[safeSU];

  const reset = () => { setStepL(0); setStepU(0); };

  const renderBound = (
    mode: BoundMode,
    step: BoundStep,
    steps: BoundStep[],
    stepIdx: number,
    setStep: (fn: (s: number) => number) => void,
    result: number
  ) => {
    const getColor = (idx: number) => {
      if (!step) return '#3f3f46';
      if (idx < step.lo || idx > step.hi) return '#1c1917';
      if (idx === step.mid) return step.condition ? '#f87171' : '#60a5fa';
      return '#4f46e5';
    };

    return (
      <div className="flex-1 min-w-0 bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-xl p-5 space-y-4">
        {/* 标题 */}
        <div className={`font-bold text-base ${mode === 'lower' ? 'text-blue-600 dark:text-blue-400' : 'text-orange-600 dark:text-orange-400'}`}>
          {mode === 'lower' ? 'lower_bound（第一个 ≥ target）' : 'upper_bound（第一个 > target）'}
        </div>

        {/* 伪代码 */}
        <pre className="text-xs bg-slate-50 dark:bg-zinc-950 border border-slate-200 dark:border-zinc-800 rounded-lg p-3 overflow-x-auto text-slate-600 dark:text-zinc-300 leading-5">
{`lo, hi = 0, len(arr)
while lo < hi:
  mid = lo + (hi-lo)//2
  if arr[mid] ${mode === 'lower' ? '<' : '<='} target:
    lo = mid + 1
  else:
    hi = mid
return lo  # ${mode === 'lower' ? '第一个 ≥ target' : '第一个 > target'}`}
        </pre>

        {/* 控制 */}
        <div className="flex gap-2 items-center">
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="px-3 py-1.5 text-sm bg-slate-100 dark:bg-zinc-700 hover:bg-slate-200 dark:hover:bg-zinc-600 disabled:opacity-40 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">←</button>
          <button onClick={() => setStep(s => Math.min(steps.length - 1, s + 1))} disabled={stepIdx >= steps.length - 1}
            className="px-3 py-1.5 text-sm bg-sky-600 hover:bg-sky-500 disabled:opacity-40 rounded-lg text-white transition-colors">→</button>
          <span className="text-sm text-slate-500 dark:text-zinc-400">{stepIdx+1} / {steps.length}</span>
        </div>

        {/* 描述 */}
        {step && (
          <div className={`text-sm rounded-lg px-3 py-2 ${step.condition ? 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300' : 'bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300'}`}>
            {step.desc}
          </div>
        )}

        {/* 数组格子 */}
        <div className="flex gap-1.5 flex-wrap">
          {arr.map((val, idx) => (
            <div key={idx} className="flex flex-col items-center" style={{ minWidth: 38 }}>
              {/* 指针标签 */}
              <div className="text-center" style={{ fontSize: 10, minHeight: 16 }}>
                {idx === step?.lo && <span className="text-blue-500 dark:text-blue-400 font-bold">lo</span>}
                {idx === step?.hi && idx !== step?.lo && <span className="text-orange-500 dark:text-orange-400 font-bold">hi</span>}
                {idx === step?.mid && idx !== step?.lo && idx !== step?.hi && <span className="text-yellow-500 dark:text-yellow-400 font-bold">mid</span>}
              </div>
              <div className="flex items-center justify-center rounded-lg font-mono font-bold text-sm transition-all duration-200" style={{
                width: 34, height: 34,
                backgroundColor: getColor(idx),
                opacity: step && (idx < step.lo || idx > step.hi) ? 0.2 : 1,
                color: getColor(idx) === '#1c1917' ? '#52525b' : '#fff',
              }}>
                {val}
              </div>
              <div className="text-slate-400 dark:text-zinc-600 text-center" style={{ fontSize: 10 }}>[{idx}]</div>
            </div>
          ))}
          {/* 越界 n */}
          <div className="flex flex-col items-center" style={{ minWidth: 38 }}>
            <div className="text-center" style={{ fontSize: 10, minHeight: 16 }}>
              {step?.hi === arr.length && <span className="text-orange-500 dark:text-orange-400 font-bold">hi</span>}
              {step?.lo === arr.length && <span className="text-blue-500 dark:text-blue-400 font-bold">lo</span>}
            </div>
            <div className="flex items-center justify-center rounded-lg font-mono text-sm border-2 border-dashed border-slate-300 dark:border-zinc-700 text-slate-400 dark:text-zinc-600" style={{ width: 34, height: 34 }}>n</div>
            <div className="text-slate-400 dark:text-zinc-600 text-center" style={{ fontSize: 10 }}>[{arr.length}]</div>
          </div>
        </div>

        {/* 当前 lo/hi/mid */}
        {step && (
          <div className="grid grid-cols-3 gap-2 text-sm text-center">
            {[['lo', step.lo, '#60a5fa'],['mid', step.mid, '#fbbf24'],['hi', step.hi, '#f97316']].map(([n,v,c])=>(
              <div key={String(n)} className="bg-slate-100 dark:bg-zinc-800 rounded-lg py-2">
                <span className="font-bold" style={{color:String(c)}}>{String(n)}</span>
                <span className="text-slate-700 dark:text-zinc-300"> = {String(v)}</span>
              </div>
            ))}
          </div>
        )}

        {/* 最终结果 */}
        {stepIdx >= steps.length - 1 && (
          <div className={`rounded-xl p-3 text-sm text-center border ${mode === 'lower' ? 'border-blue-300 dark:border-blue-600 bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300' : 'border-orange-300 dark:border-orange-600 bg-orange-50 dark:bg-orange-950 text-orange-700 dark:text-orange-300'}`}>
            返回 <span className="font-bold text-xl">{result}</span>
            {result < arr.length
              ? <span className="ml-1">（A[{result}] = {arr[result]}，{mode === 'lower' ? '第一个 ≥' : '第一个 >'} {target}）</span>
              : <span className="ml-1">（= n，所有元素 {mode === 'lower' ? '<' : '≤'} target）</span>
            }
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-950 overflow-hidden text-slate-900 dark:text-white">
      {/* 头部 */}
      <div className="px-6 py-5 bg-slate-100 dark:bg-zinc-900 border-b border-slate-200 dark:border-zinc-800">
        <h3 className="text-xl font-bold text-sky-600 dark:text-sky-400">二分搜索边界模板</h3>
        <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">lower_bound vs upper_bound — 两种边界查找的对比演示</p>
      </div>

      <div className="p-6 space-y-6">
        {/* 预设 */}
        <div className="flex gap-2 flex-wrap">
          {[[1,3,3,5,5,5,7,9],[1,2,3,4,5,6,7],[5,5,5,5,5],[1,1,1,2,2,3]].map((preset, i) => (
            <button key={i} onClick={() => { setArrStr(preset.join(', ')); reset(); }}
              className="px-3 py-1.5 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">
              [{preset.join(',')}]
            </button>
          ))}
        </div>

        <div className="flex gap-3 flex-wrap">
          <input value={arrStr} onChange={e => { setArrStr(e.target.value); reset(); }}
            className="flex-1 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-4 py-2.5 text-sm font-mono text-slate-800 dark:text-zinc-100 placeholder:text-slate-400 dark:placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
            placeholder="输入有序数组（自动排序）" />
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500 dark:text-zinc-400">target:</span>
            <input value={targetStr} onChange={e => { setTargetStr(e.target.value); reset(); }}
              className="w-20 bg-white dark:bg-zinc-800 border border-slate-300 dark:border-zinc-600 rounded-lg px-3 py-2.5 text-sm font-mono text-slate-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-sky-400" />
          </div>
          <button onClick={reset} className="px-4 py-2.5 text-sm bg-slate-200 dark:bg-zinc-700 hover:bg-slate-300 dark:hover:bg-zinc-600 rounded-lg text-slate-700 dark:text-zinc-200 transition-colors">↺</button>
        </div>

        {/* 数组与 target 预览 */}
        <div className="text-sm text-slate-600 dark:text-zinc-300 bg-slate-100 dark:bg-zinc-800/60 border border-slate-200 dark:border-zinc-700 rounded-lg px-4 py-3">
          数组：<span className="font-mono">[{arr.join(', ')}]</span>，查找 target = <span className="text-yellow-600 dark:text-yellow-400 font-bold">{target}</span>
          <span className="ml-4 text-slate-500 dark:text-zinc-500">（{arr.filter(v => v === target).length} 个 {target}，占位 [{arr.indexOf(target)}..{arr.lastIndexOf(target)}]）</span>
        </div>

        {/* 并排展示 */}
        <div className="flex gap-4 flex-wrap">
          {renderBound('lower', sL, lSteps, safeSL, fn => setStepL(fn), lResult)}
          {renderBound('upper', sU, uSteps, safeSU, fn => setStepU(fn), uResult)}
        </div>

        {/* 总结表 */}
        <div className="bg-slate-100 dark:bg-zinc-800/60 border border-slate-200 dark:border-zinc-700 rounded-xl p-5">
          <div className="text-base font-semibold text-slate-700 dark:text-zinc-200 mb-3">结果对比</div>
          <div className="grid grid-cols-3 gap-3 text-sm text-center">
            {['', 'lower_bound', 'upper_bound'].map(h => <div key={h} className={`font-semibold ${h === 'lower_bound' ? 'text-blue-600 dark:text-blue-400' : h === 'upper_bound' ? 'text-orange-600 dark:text-orange-400' : 'text-slate-500 dark:text-zinc-400'}`}>{h || '说明'}</div>)}
            {[
              ['返回索引', lResult, uResult],
              ['语义', `第一个 ≥ ${target}`, `第一个 > ${target}`],
              ['当前元素', lResult < arr.length ? arr[lResult] : '越界', uResult < arr.length ? arr[uResult] : '越界'],
              [`${target}的数量`, `upper-lower = ${uResult - lResult}`, `= ${uResult - lResult}`],
            ].map(([label, lv, uv], i) => (
              <React.Fragment key={i}>
                <div className="text-slate-500 dark:text-zinc-500">{String(label)}</div>
                <div className="text-blue-600 dark:text-blue-300 font-mono">{String(lv)}</div>
                <div className="text-orange-600 dark:text-orange-300 font-mono">{String(uv)}</div>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Python bisect 对照 */}
        <div className="bg-slate-100 dark:bg-zinc-900 border border-slate-200 dark:border-zinc-700 rounded-lg px-4 py-3 text-sm text-slate-500 dark:text-zinc-400">
          <span className="text-slate-700 dark:text-zinc-300 font-medium">Python 标准库等价：</span>
          <span className="font-mono ml-2 text-green-600 dark:text-green-400">bisect.bisect_left(arr, {target})</span> → lower_bound；
          <span className="font-mono ml-2 text-orange-600 dark:text-orange-400">bisect.bisect_right(arr, {target})</span> → upper_bound
        </div>
      </div>
    </div>
  );
}

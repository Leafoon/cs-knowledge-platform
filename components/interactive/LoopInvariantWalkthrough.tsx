'use client';
import React, { useState } from 'react';

// =====================================================
// LoopInvariantWalkthrough — DSA Chapter 0
// 循环不变式三步骤动画（以插入排序为例，逐步高亮）
// =====================================================

type SortPhase = 'init' | 'pick' | 'shift' | 'insert' | 'done';

interface Step {
  phase: SortPhase;
  arr: number[];
  sortedLen: number;    // arr[0..sortedLen-1] 已排序
  keyIdx: number;       // 当前被提取的 key 的原始下标（-1 = 无）
  keyVal: number | null;
  shiftFrom: number;    // 正在向右移的格子下标（-1 = 无）
  insertAt: number;     // key 将插入的位置（-1 = 未确定）
  invariant: string;    // 当前循环不变式状态说明
  action: string;       // 本步发生了什么
  highlight: 'init' | 'maintain' | 'terminate';
}

function buildSteps(initial: number[]): Step[] {
  const steps: Step[] = [];
  const arr = [...initial];
  const n = arr.length;

  steps.push({
    phase: 'init', arr: [...arr], sortedLen: 1, keyIdx: -1, keyVal: null,
    shiftFrom: -1, insertAt: -1,
    invariant: '初始化：arr[0..0] 只有一个元素，天然有序 ✓',
    action: '算法开始：第 0 个元素单独构成有序区，循环不变式成立（初始化）。',
    highlight: 'init',
  });

  for (let i = 1; i < n; i++) {
    const key = arr[i];

    // Step: pick key
    steps.push({
      phase: 'pick', arr: [...arr], sortedLen: i, keyIdx: i, keyVal: key,
      shiftFrom: -1, insertAt: -1,
      invariant: `i=${i}：arr[0..${i - 1}] 已排序（不变式成立），准备将 arr[${i}]=${key} 插入正确位置`,
      action: `提取 key = arr[${i}] = ${key}。无序区第一個元素被"拿起"，暂存为 key。`,
      highlight: 'maintain',
    });

    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      steps.push({
        phase: 'shift', arr: [...arr], sortedLen: i, keyIdx: -1, keyVal: key,
        shiftFrom: j, insertAt: -1,
        invariant: `arr[${j}]=${arr[j]} > key=${key}，需要右移腾出空位`,
        action: `arr[${j + 1}] ← arr[${j}]（将 ${arr[j]} 向右移一格）`,
        highlight: 'maintain',
      });
      arr[j + 1] = arr[j];
      j--;
    }

    steps.push({
      phase: 'insert', arr: [...arr], sortedLen: i + 1, keyIdx: -1, keyVal: key,
      shiftFrom: -1, insertAt: j + 1,
      invariant: `arr[0..${i}] 有序 ✓（不变式在本次迭代结束后保持）`,
      action: `arr[${j + 1}] ← key = ${key}，插入完毕。arr[0..${i}] 现已排序。`,
      highlight: 'maintain',
    });
    arr[j + 1] = key;
  }

  steps.push({
    phase: 'done', arr: [...arr], sortedLen: n, keyIdx: -1, keyVal: null,
    shiftFrom: -1, insertAt: -1,
    invariant: `终止：i = n = ${n}，arr[0..${n - 1}] 即整个数组，已完全排序 ✓`,
    action: '循环结束！不变式给出结论：整个数组有序，算法正确。',
    highlight: 'terminate',
  });

  return steps;
}

const PRESETS = [
  [5, 3, 8, 1, 2],
  [4, 2, 6, 1, 9, 3],
  [1, 2, 3, 4, 5],     // 最佳情况（已有序）
  [5, 4, 3, 2, 1],     // 最坏情况（逆序）
];

const HIGHLIGHT_STYLES = {
  init: { bar: 'border-indigo-400 bg-indigo-500/20', badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30', label: '🔵 初始化' },
  maintain: { bar: 'border-amber-400 bg-amber-500/20', badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30', label: '🟡 保持' },
  terminate: { bar: 'border-emerald-400 bg-emerald-500/20', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30', label: '🟢 终止' },
};

export default function LoopInvariantWalkthrough() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [speed, setSpeed] = useState(900);

  const initial = PRESETS[presetIdx];
  const steps = buildSteps(initial);
  const safeIdx = Math.min(stepIdx, steps.length - 1);
  const step = steps[safeIdx];
  const hl = HIGHLIGHT_STYLES[step.highlight];

  // Auto-play
  React.useEffect(() => {
    if (!autoPlay) return;
    if (safeIdx >= steps.length - 1) { setAutoPlay(false); return; }
    const t = setTimeout(() => setStepIdx(s => s + 1), speed);
    return () => clearTimeout(t);
  }, [autoPlay, safeIdx, steps.length, speed]);

  const reset = () => { setStepIdx(0); setAutoPlay(false); };

  // Cell colors
  const cellStyle = (idx: number): string => {
    if (step.phase === 'done') return 'bg-emerald-500/20 border-emerald-400';
    if (idx === step.keyIdx) return 'bg-amber-500/30 border-amber-400 scale-105 shadow-lg';
    if (idx === step.insertAt) return 'bg-emerald-500/30 border-emerald-400';
    if (idx === step.shiftFrom) return 'bg-rose-500/20 border-rose-400';
    if (idx < step.sortedLen && step.phase !== 'pick') return 'bg-indigo-500/15 border-indigo-500/40';
    return 'bg-bg-tertiary border-border-subtle';
  };

  return (
    <div className="my-8 rounded-2xl border border-border-subtle bg-bg-secondary overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border-subtle bg-bg-tertiary flex items-center gap-3">
        <span className="text-2xl">🔄</span>
        <div>
          <h3 className="font-bold text-text-primary text-lg">循环不变式演示</h3>
          <p className="text-sm text-text-tertiary">以插入排序为例，逐步展示「初始化 → 保持 → 终止」三阶段</p>
        </div>
        <span className={`ml-auto px-2.5 py-1 rounded-full text-xs font-semibold border ${hl.badge}`}>
          {hl.label}
        </span>
      </div>

      {/* Presets */}
      <div className="px-6 pt-3 flex flex-wrap gap-2 items-center">
        <span className="text-xs text-text-tertiary font-semibold">示例数组：</span>
        {PRESETS.map((p, i) => (
          <button
            key={i}
            onClick={() => { setPresetIdx(i); reset(); }}
            className={`px-3 py-1 rounded-lg text-xs font-mono border transition-all ${
              presetIdx === i
                ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                : 'bg-bg-tertiary text-text-tertiary border-border-subtle hover:border-indigo-400/30 hover:text-text-secondary'
            }`}
          >
            [{p.join(', ')}]
          </button>
        ))}
      </div>

      {/* Array Visualization */}
      <div className="px-6 pt-4 pb-2">
        <div className="flex items-end gap-2 justify-center flex-wrap">
          {step.arr.map((val, idx) => (
            <div key={idx} className="flex flex-col items-center gap-1">
              {/* Key floating above */}
              {idx === step.keyIdx && step.keyVal !== null && (
                <div className="text-xs font-bold text-amber-300 bg-amber-500/20 border border-amber-400/40 rounded-md px-1.5 py-0.5 -mb-1 animate-bounce">
                  key={step.keyVal}
                </div>
              )}
              {/* Cell */}
              <div className={`w-10 h-10 flex items-center justify-center rounded-lg border-2 font-bold text-sm font-mono text-text-primary transition-all duration-300 ${cellStyle(idx)}`}>
                {val}
              </div>
              {/* Index label */}
              <span className="text-xs text-text-tertiary">{idx}</span>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-3 flex flex-wrap gap-3 justify-center text-xs text-text-tertiary">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-indigo-500/20 border border-indigo-500/40 inline-block"/>已排序区</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-500/30 border border-amber-400 inline-block"/>当前 key</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-rose-500/20 border border-rose-400 inline-block"/>右移元素</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-500/20 border border-emerald-400 inline-block"/>插入位置 / 完成</span>
        </div>
      </div>

      {/* Invariant Info Box */}
      <div className={`mx-4 mt-2 rounded-xl border p-4 transition-all ${hl.badge.replace('text-', 'border-').replace('bg-', 'bg-').split(' ')[0]} border-opacity-30`}
        style={{ background: step.highlight === 'init' ? 'rgba(99,102,241,0.06)' : step.highlight === 'terminate' ? 'rgba(52,211,153,0.06)' : 'rgba(251,191,36,0.06)' }}>
        <div className="text-xs font-semibold text-text-tertiary uppercase tracking-wide mb-1">📌 循环不变式状态</div>
        <div className="text-sm text-text-primary font-medium">{step.invariant}</div>
        <div className="mt-2 pt-2 border-t border-border-subtle text-xs text-text-secondary">
          <strong>本步操作：</strong>{step.action}
        </div>
      </div>

      {/* Controls */}
      <div className="px-6 py-4 flex items-center gap-3 flex-wrap">
        <button
          onClick={reset}
          className="px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:border-indigo-400/40 transition-colors"
        >
          ↩ 重置
        </button>
        <button
          onClick={() => setStepIdx(s => Math.max(0, s - 1))}
          disabled={safeIdx === 0}
          className="px-3 py-2 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:border-indigo-400/40 transition-colors disabled:opacity-40"
        >
          ← 上一步
        </button>
        <button
          onClick={() => setStepIdx(s => Math.min(steps.length - 1, s + 1))}
          disabled={safeIdx >= steps.length - 1}
          className="px-4 py-2 rounded-lg bg-indigo-500/20 border border-indigo-500/40 text-xs text-indigo-300 font-semibold hover:bg-indigo-500/30 transition-colors disabled:opacity-40"
        >
          下一步 →
        </button>
        <button
          onClick={() => { if (safeIdx >= steps.length - 1) reset(); setAutoPlay(a => !a); }}
          className={`px-4 py-2 rounded-lg border text-xs font-semibold transition-all ${
            autoPlay
              ? 'bg-rose-500/20 border-rose-500/40 text-rose-300'
              : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/30'
          }`}
        >
          {autoPlay ? '⏸ 暂停' : '▶ 自动播放'}
        </button>

        <div className="flex items-center gap-2 ml-auto text-xs text-text-tertiary">
          <span>速度</span>
          {[['慢', 1400], ['中', 900], ['快', 400]].map(([label, ms]) => (
            <button
              key={ms}
              onClick={() => setSpeed(ms as number)}
              className={`px-2 py-1 rounded border text-xs transition-all ${
                speed === ms
                  ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                  : 'bg-bg-tertiary border-border-subtle text-text-tertiary hover:text-text-secondary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Progress bar */}
      <div className="px-6 pb-4">
        <div className="flex items-center justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {safeIdx + 1} / {steps.length}</span>
          <span>{Math.round((safeIdx / (steps.length - 1)) * 100)}%</span>
        </div>
        <div className="w-full h-1.5 bg-bg-tertiary rounded-full overflow-hidden border border-border-subtle">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${(safeIdx / (steps.length - 1)) * 100}%` }}
          />
        </div>

        {/* Three-phase markers */}
        <div className="mt-2 flex justify-between text-xs text-text-tertiary">
          <span className="text-indigo-400">🔵 初始化</span>
          <span className="text-amber-400">🟡 保持</span>
          <span className="text-emerald-400">🟢 终止</span>
        </div>
      </div>
    </div>
  );
}

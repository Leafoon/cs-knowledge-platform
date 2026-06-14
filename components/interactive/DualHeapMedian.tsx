"use client";
import React, { useState, useCallback } from "react";

// ──────────── Dual-Heap Median Tracker (LeetCode #295) ────────────
// lo: MAX-HEAP (stores smaller half)  — simulated as max-heap with sorted-desc array
// hi: MIN-HEAP (stores larger half)   — simulated as min-heap with sorted-asc array

function pushMaxHeap(heap: number[], val: number) {
  heap.push(val);
  heap.sort((a, b) => b - a); // max-heap: descending
}
function pushMinHeap(heap: number[], val: number) {
  heap.push(val);
  heap.sort((a, b) => a - b); // min-heap: ascending
}

interface HeapState {
  lo: number[];  // max-heap (smaller half)
  hi: number[];  // min-heap (larger half)
  median: number | null;
  newVal: number;
  desc: string;
  action: "addLo" | "addHi" | "rebalance" | "none";
}

function addNumber(prevLo: number[], prevHi: number[], val: number): HeapState[] {
  const steps: HeapState[] = [];
  const lo = [...prevLo];
  const hi = [...prevHi];

  // Step 1: always add to lo first
  pushMaxHeap(lo, val);
  steps.push({ lo: [...lo], hi: [...hi], median: getMedian(lo, hi), newVal: val, desc: `插入 ${val}：先放入左堆（max-heap）中`, action: "addLo" });

  // Step 2: if hi not empty and lo's max > hi's min, move lo's max to hi
  if (hi.length > 0 && lo[0] > hi[0]) {
    const top = lo.shift()!;
    pushMinHeap(hi, top);
    steps.push({ lo: [...lo], hi: [...hi], median: getMedian(lo, hi), newVal: val, desc: `左堆最大 ${top} > 右堆最小 ${hi[0]}，将 ${top} 移入右堆`, action: "rebalance" });
  }

  // Step 3: rebalance sizes: |lo| must equal |hi| or |lo| = |hi| + 1
  if (lo.length > hi.length + 1) {
    const top = lo.shift()!;
    pushMinHeap(hi, top);
    steps.push({ lo: [...lo], hi: [...hi], median: getMedian(lo, hi), newVal: val, desc: `左堆过大（${lo.length + 1} vs ${hi.length - 1}），将 ${top} 移入右堆`, action: "rebalance" });
  } else if (hi.length > lo.length) {
    const top = hi.shift()!;
    pushMaxHeap(lo, top);
    steps.push({ lo: [...lo], hi: [...hi], median: getMedian(lo, hi), newVal: val, desc: `右堆过大（${hi.length + 1} vs ${lo.length - 1}），将 ${top} 移入左堆`, action: "rebalance" });
  }

  steps.push({ lo: [...lo], hi: [...hi], median: getMedian(lo, hi), newVal: val, desc: `插入 ${val} 完成。中位数 = ${getMedian(lo, hi)}`, action: "none" });

  return steps;
}

function getMedian(lo: number[], hi: number[]): number | null {
  if (lo.length === 0 && hi.length === 0) return null;
  if (lo.length >= hi.length) return lo[0] ?? null;
  return hi[0] ?? null;
}

const SEQUENCES: Record<string, number[]> = {
  "CLRS 示例": [5, 15, 1, 3, 2, 8, 7],
  "奇数个数": [6, 1, 4, 2, 3],
  "已排序": [1, 2, 3, 4, 5, 6, 7],
};

// ──────────── Heap Display ────────────
function HeapDisplay({ values, type, newVal }: { values: number[]; type: "max" | "min"; newVal?: number }) {
  const color = type === "max" ? "#3b82f6" : "#f97316";
  const title = type === "max" ? "左堆 lo（MAX-HEAP，存较小半）" : "右堆 hi（MIN-HEAP，存较大半）";
  const label = type === "max" ? "堆顶=最大值" : "堆顶=最小值";

  return (
    <div className="rounded-lg p-3 flex-1 min-w-0" style={{ background: "var(--color-bg-secondary)", border: `2px solid ${color}22` }}>
      <p className="text-xs font-bold mb-2" style={{ color }}>{title}</p>
      {values.length === 0 ? (
        <p className="text-xs text-center py-4" style={{ color: "var(--color-text-muted)" }}>（空）</p>
      ) : (
        <div className="flex flex-wrap gap-1 justify-center">
          {values.map((v, i) => (
            <div key={i} className="relative">
              <div className="w-10 h-10 flex items-center justify-center rounded-lg font-bold text-sm border-2 transition-all"
                style={{ background: i === 0 ? color : "var(--color-bg-card)", color: i === 0 ? "#fff" : "var(--color-text-primary)", borderColor: i === 0 ? color : "var(--color-border)", boxShadow: v === newVal ? "0 0 0 2px #f59e0b" : "none" }}>
                {v}
              </div>
            </div>
          ))}
        </div>
      )}
      {values.length > 0 && <p className="text-xs text-center mt-1" style={{ color }}>{label} = {values[0]}</p>}
      <p className="text-xs text-center" style={{ color: "var(--color-text-muted)" }}>大小: {values.length}</p>
    </div>
  );
}

// ──────────── Main ────────────
export default function DualHeapMedian() {
  const seqKeys = Object.keys(SEQUENCES);
  const [seqKey, setSeqKey] = useState(seqKeys[0]);
  const [inputIdx, setInputIdx] = useState(-1); // next number to add
  const [history, setHistory] = useState<{ lo: number[]; hi: number[] }[]>([{ lo: [], hi: [] }]);
  const [curStep, setCurStep] = useState<HeapState>({ lo: [], hi: [], median: null, newVal: 0, desc: "点击「添加下一个数」开始", action: "none" });
  const [stepLog, setStepLog] = useState<HeapState[]>([]);
  const [innerIdx, setInnerIdx] = useState(-1); // inner step within current insert

  const sequence = SEQUENCES[seqKey];

  const addNext = useCallback(() => {
    const nextIdx = inputIdx + 1;
    if (nextIdx >= sequence.length) return;
    const { lo: prevLo, hi: prevHi } = history[history.length - 1];
    const steps = addNumber(prevLo, prevHi, sequence[nextIdx]);
    const lastStep = steps[steps.length - 1];
    setHistory(h => [...h, { lo: lastStep.lo, hi: lastStep.hi }]);
    setInputIdx(nextIdx);
    setCurStep(lastStep);
    setStepLog(prev => [...prev, ...steps]);
    setInnerIdx(prev => prev + steps.length);
  }, [inputIdx, sequence, history]);

  const reset = () => {
    setInputIdx(-1);
    setHistory([{ lo: [], hi: [] }]);
    setCurStep({ lo: [], hi: [], median: null, newVal: 0, desc: "点击「添加下一个数」开始", action: "none" });
    setStepLog([]);
    setInnerIdx(-1);
  };

  const nextNum = sequence[inputIdx + 1];
  const finished = inputIdx >= sequence.length - 1;

  return (
    <div className="rounded-xl border p-4 space-y-4 select-none" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-card)" }}>
      <h3 className="font-bold text-base" style={{ color: "var(--color-text-primary)" }}>
        🎯 双堆维护数据流中位数（LeetCode #295）
      </h3>

      {/* Sequence selector */}
      <div className="flex gap-1 flex-wrap items-center">
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>数据序列：</span>
        {seqKeys.map(k => (
          <button key={k} onClick={() => { setSeqKey(k); reset(); }}
            className="px-2 py-1 rounded text-xs border"
            style={{ background: seqKey === k ? "#3b82f6" : "var(--color-bg-secondary)", color: seqKey === k ? "#fff" : "var(--color-text-primary)", borderColor: "var(--color-border)" }}>
            {k}
          </button>
        ))}
        <button onClick={reset} className="px-2 py-1 rounded text-xs border ml-2" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>↺ 重置</button>
      </div>

      {/* Sequence progress */}
      <div className="rounded-lg p-3" style={{ background: "var(--color-bg-secondary)" }}>
        <p className="text-xs mb-2" style={{ color: "var(--color-text-muted)" }}>输入序列（{inputIdx + 1}/{sequence.length} 已处理）：</p>
        <div className="flex gap-1 flex-wrap">
          {sequence.map((v, i) => (
            <div key={i} className="w-9 h-9 flex items-center justify-center rounded font-bold text-sm border-2 transition-all"
              style={{ background: i <= inputIdx ? "#22c55e22" : i === inputIdx + 1 ? "#f59e0b22" : "var(--color-bg-card)", borderColor: i <= inputIdx ? "#22c55e" : i === inputIdx + 1 ? "#f59e0b" : "var(--color-border)", color: "var(--color-text-primary)" }}>
              {v}
            </div>
          ))}
        </div>
      </div>

      {/* Dual heaps */}
      <div className="flex gap-3">
        <HeapDisplay values={curStep.lo} type="max" newVal={curStep.newVal} />
        <div className="flex flex-col items-center justify-center gap-2 px-2">
          <div className="rounded-full w-12 h-12 flex items-center justify-center text-lg font-black border-4"
            style={{ borderColor: curStep.median !== null ? "#22c55e" : "var(--color-border)", background: curStep.median !== null ? "#22c55e22" : "var(--color-bg-secondary)", color: "#22c55e" }}>
            {curStep.median ?? "?"}
          </div>
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>中位数</span>
        </div>
        <HeapDisplay values={curStep.hi} type="min" newVal={curStep.newVal} />
      </div>

      {/* Invariant display */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg p-2 text-center" style={{ background: "var(--color-bg-secondary)" }}>
          <div className="font-bold" style={{ color: "var(--color-text-primary)" }}>|lo| = {curStep.lo.length}, |hi| = {curStep.hi.length}</div>
          <div style={{ color: Math.abs(curStep.lo.length - curStep.hi.length) <= 1 ? "#22c55e" : "#ef4444" }}>
            {Math.abs(curStep.lo.length - curStep.hi.length) <= 1 ? "✓ 大小平衡（|lo|-|hi|≤1）" : "⚠️ 需要再平衡"}
          </div>
        </div>
        <div className="rounded-lg p-2 text-center" style={{ background: "var(--color-bg-secondary)" }}>
          <div className="font-bold" style={{ color: "var(--color-text-primary)" }}>
            {curStep.lo.length > 0 && curStep.hi.length > 0 ? `lo.max(${curStep.lo[0]}) ≤ hi.min(${curStep.hi[0]})` : "等待数据..."}
          </div>
          <div style={{ color: (curStep.lo.length === 0 || curStep.hi.length === 0 || curStep.lo[0] <= curStep.hi[0]) ? "#22c55e" : "#ef4444" }}>
            {(curStep.lo.length === 0 || curStep.hi.length === 0 || curStep.lo[0] <= curStep.hi[0]) ? "✓ 顺序不变量" : "⚠️ 违反"}
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="rounded-lg px-4 py-3 text-sm" style={{ background: "rgba(59,130,246,0.08)", color: "var(--color-text-primary)", borderLeft: "4px solid #3b82f6" }}>
        {curStep.desc}
      </div>

      {/* Controls */}
      <div className="flex justify-center">
        <button onClick={addNext} disabled={finished}
          className="px-5 py-2 rounded-lg font-bold text-sm disabled:opacity-40 transition-all"
          style={{ background: finished ? "var(--color-bg-secondary)" : "#3b82f6", color: finished ? "var(--color-text-muted)" : "#fff" }}>
          {finished ? "✅ 序列处理完毕" : `添加下一个数 → ${nextNum}`}
        </button>
      </div>

      {/* Step log */}
      {stepLog.length > 0 && (
        <div className="rounded-lg p-3 max-h-32 overflow-y-auto text-xs space-y-1" style={{ background: "var(--color-bg-secondary)" }}>
          <p className="font-semibold mb-1" style={{ color: "var(--color-text-muted)" }}>操作日志：</p>
          {stepLog.map((s, i) => (
            <p key={i} style={{ color: "var(--color-text-muted)" }}>
              {i + 1}. 中位数={s.median ?? "?"} — {s.desc}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

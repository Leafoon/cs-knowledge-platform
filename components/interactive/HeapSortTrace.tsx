"use client";
import React, { useState, useCallback } from "react";

// ──────────────── types ────────────────
interface SortStep {
  arr: number[];
  heapSize: number;      // effective heap boundary
  phase: "build" | "extract";
  swapA: number;
  swapB: number;
  heapifyAt: number;    // -1 if not heapifying
  desc: string;
  sorted: number[];     // indices already in final position
}

// ──────────────── helpers ────────────────
function heapSortSteps(input: number[]): SortStep[] {
  const steps: SortStep[] = [];
  const arr = [...input];
  const n = arr.length;

  function addStep(heapSize: number, phase: "build" | "extract", swapA: number, swapB: number, heapifyAt: number, desc: string, sorted: number[]) {
    steps.push({ arr: [...arr], heapSize, phase, swapA, swapB, heapifyAt, desc, sorted: [...sorted] });
  }

  function heapify(i: number, size: number, phase: "build" | "extract", sorted: number[]) {
    let largest = i;
    const l = 2 * i + 1;
    const r = 2 * i + 2;
    if (l < size && arr[l] > arr[largest]) largest = l;
    if (r < size && arr[r] > arr[largest]) largest = r;
    if (largest !== i) {
      addStep(size, phase, i, largest, i, `交换 [${i}]=${arr[i]} ↔ [${largest}]=${arr[largest]}，继续下沉`, sorted);
      [arr[i], arr[largest]] = [arr[largest], arr[i]];
      addStep(size, phase, -1, -1, largest, `已交换，递归 HEAPIFY([${largest}])`, sorted);
      heapify(largest, size, phase, sorted);
    } else {
      addStep(size, phase, -1, -1, i, `节点 [${i}]=${arr[i]} 已是局部最大，停止`, sorted);
    }
  }

  const sorted: number[] = [];

  // Phase 1: BUILD
  addStep(n, "build", -1, -1, -1, `开始 BUILD-MAX-HEAP，从 i=${Math.floor(n / 2) - 1} 倒序调用 HEAPIFY`, sorted);
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    addStep(n, "build", -1, -1, i, `BUILD 阶段：对节点 [${i}]=${arr[i]} 调用 MAX-HEAPIFY`, sorted);
    heapify(i, n, "build", sorted);
  }
  addStep(n, "build", -1, -1, -1, `BUILD-MAX-HEAP 完成！根节点 [0]=${arr[0]} 是最大值`, sorted);

  // Phase 2: EXTRACT
  for (let i = n - 1; i >= 1; i--) {
    addStep(i + 1, "extract", 0, i, -1, `EXTRACT 阶段：将根 [0]=${arr[0]} 与末元素 [${i}]=${arr[i]} 交换`, sorted);
    [arr[0], arr[i]] = [arr[i], arr[0]];
    sorted.push(i);
    addStep(i, "extract", -1, -1, -1, `${arr[i]} 已确定（放入有序区），堆大小缩为 ${i}，重新 HEAPIFY 根`, [...sorted]);
    if (i > 1) heapify(0, i, "extract", [...sorted]);
  }
  sorted.push(0);
  addStep(0, "extract", -1, -1, -1, `排序完成！数组：[${arr.join(", ")}]`, [...sorted]);

  return steps;
}

const PRESETS: Record<string, number[]> = {
  "6 元素示例": [3, 9, 2, 1, 4, 5],
  "逆序输入": [5, 4, 3, 2, 1],
  "已排序": [1, 2, 3, 4, 5, 6],
};

// ──────────────── Main ────────────────
export default function HeapSortTrace() {
  const presetKeys = Object.keys(PRESETS);
  const [presetKey, setPresetKey] = useState(presetKeys[0]);
  const [steps, setSteps] = useState<SortStep[]>([]);
  const [idx, setIdx] = useState(0);
  const [generated, setGenerated] = useState(false);

  const generate = useCallback(() => {
    const s = heapSortSteps(PRESETS[presetKey]);
    setSteps(s);
    setIdx(0);
    setGenerated(true);
  }, [presetKey]);

  const cur = steps[idx];

  return (
    <div className="rounded-xl border p-4 space-y-4 select-none" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-card)" }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-bold text-base" style={{ color: "var(--color-text-primary)" }}>
          🔢 堆排序完整执行追踪
        </h3>
        <div className="flex gap-1 flex-wrap">
          {presetKeys.map(k => (
            <button key={k} onClick={() => { setPresetKey(k); setGenerated(false); }}
              className="px-2 py-1 rounded text-xs font-medium border transition-colors"
              style={{ background: presetKey === k ? "#6366f1" : "var(--color-bg-secondary)", color: presetKey === k ? "#fff" : "var(--color-text-primary)", borderColor: "var(--color-border)" }}>
              {k}
            </button>
          ))}
          <button onClick={generate}
            className="px-3 py-1 rounded text-xs font-bold"
            style={{ background: "#6366f1", color: "#fff" }}>
            ▶ 开始
          </button>
        </div>
      </div>

      {!generated && (
        <div className="rounded-lg p-4 text-center text-sm" style={{ background: "var(--color-bg-secondary)", color: "var(--color-text-muted)" }}>
          初始数组：[{PRESETS[presetKey].join(", ")}]<br />点击"开始"生成步骤追踪
        </div>
      )}

      {generated && cur && (
        <>
          {/* Phase badge */}
          <div className="flex gap-2 items-center">
            <span className="px-2 py-1 rounded-full text-xs font-bold"
              style={{ background: cur.phase === "build" ? "rgba(99,102,241,0.15)" : "rgba(249,115,22,0.15)", color: cur.phase === "build" ? "#6366f1" : "#f97316" }}>
              {cur.phase === "build" ? "🏗 BUILD-MAX-HEAP 阶段" : "⬆️ EXTRACT 排序阶段"}
            </span>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>步骤 {idx + 1} / {steps.length}</span>
          </div>

          {/* Array view */}
          <div className="rounded-lg p-4" style={{ background: "var(--color-bg-secondary)" }}>
            <div className="flex gap-1 flex-wrap justify-center">
              {cur.arr.map((val, i) => {
                const isSorted = cur.sorted.includes(i);
                const isSwapA = i === cur.swapA;
                const isSwapB = i === cur.swapB;
                const isHeapifyAt = i === cur.heapifyAt;
                const isOutOfHeap = cur.heapSize >= 0 && i >= cur.heapSize;

                let bg = "var(--color-bg-card)";
                let textCol = "var(--color-text-primary)";
                let border = "var(--color-border)";

                if (isSorted) { bg = "#22c55e"; textCol = "#fff"; border = "#16a34a"; }
                else if (isSwapA || isSwapB) { bg = "#f59e0b"; textCol = "#fff"; border = "#b45309"; }
                else if (isHeapifyAt) { bg = "#6366f1"; textCol = "#fff"; border = "#4338ca"; }
                else if (isOutOfHeap && !isSorted) { bg = "var(--color-bg-secondary)"; textCol = "var(--color-text-muted)"; }

                return (
                  <div key={i} className="flex flex-col items-center">
                    <div className="w-11 h-11 flex items-center justify-center rounded-lg font-bold text-sm border-2 transition-all duration-300"
                      style={{ background: bg, color: textCol, borderColor: border }}>
                      {val}
                    </div>
                    <span className="text-xs mt-0.5" style={{ color: "var(--color-text-muted)" }}>[{i}]</span>
                    {i === (cur.heapSize > 0 ? cur.heapSize - 1 : -1) && (
                      <span className="text-xs" style={{ color: "#f97316" }}>←堆</span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Legend */}
            <div className="flex gap-4 mt-3 text-xs flex-wrap justify-center">
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded" style={{ background: "#6366f1" }} /><span style={{ color: "var(--color-text-muted)" }}>HEAPIFY 当前节点</span></div>
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded" style={{ background: "#f59e0b" }} /><span style={{ color: "var(--color-text-muted)" }}>待交换节点</span></div>
              <div className="flex items-center gap-1"><div className="w-3 h-3 rounded" style={{ background: "#22c55e" }} /><span style={{ color: "var(--color-text-muted)" }}>已排好（有序区）</span></div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "堆区大小", val: cur.heapSize < 0 ? "-" : cur.heapSize, color: "#6366f1" },
              { label: "已排序", val: cur.sorted.length, color: "#22c55e" },
              { label: "进度", val: `${Math.round((idx + 1) / steps.length * 100)}%`, color: "#3b82f6" },
            ].map(({ label, val, color }) => (
              <div key={label} className="rounded-lg p-2 text-center" style={{ background: "var(--color-bg-secondary)" }}>
                <div className="text-lg font-bold" style={{ color }}>{val}</div>
                <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Description */}
          <div className="rounded-lg px-4 py-3 text-sm font-medium" style={{ background: "rgba(99,102,241,0.08)", color: "var(--color-text-primary)", borderLeft: "4px solid #6366f1" }}>
            {cur.desc}
          </div>

          {/* Controls */}
          <div className="flex gap-2 justify-center">
            <button disabled={idx === 0} onClick={() => setIdx(0)} className="px-3 py-1.5 rounded text-sm border disabled:opacity-40" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>⏮</button>
            <button disabled={idx === 0} onClick={() => setIdx(i => i - 1)} className="px-3 py-1.5 rounded text-sm border disabled:opacity-40" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>◀</button>
            <button disabled={idx === steps.length - 1} onClick={() => setIdx(i => i + 1)} className="px-4 py-1.5 rounded text-sm font-bold disabled:opacity-40" style={{ background: "#6366f1", color: "#fff" }}>▶</button>
            <button disabled={idx === steps.length - 1} onClick={() => setIdx(steps.length - 1)} className="px-3 py-1.5 rounded text-sm border disabled:opacity-40" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>⏭</button>
          </div>

          <div className="w-full rounded-full h-1.5" style={{ background: "var(--color-border)" }}>
            <div className="h-1.5 rounded-full transition-all duration-300" style={{ width: `${((idx + 1) / steps.length) * 100}%`, background: cur.phase === "build" ? "#6366f1" : "#f97316" }} />
          </div>
        </>
      )}

      {/* Complexity reminder */}
      <div className="rounded-lg p-3 text-xs" style={{ background: "var(--color-bg-secondary)" }}>
        <span className="font-semibold" style={{ color: "var(--color-text-primary)" }}>复杂度：</span>
        <span style={{ color: "var(--color-text-muted)" }}>BUILD = O(n)，EXTRACT × n = O(n log n)，总计 </span>
        <strong style={{ color: "#6366f1" }}>O(n log n)，空间 O(1)</strong>
        <span style={{ color: "var(--color-text-muted)" }}>（原地排序，不稳定）</span>
      </div>
    </div>
  );
}

"use client";
import React, { useState, useCallback } from "react";

// ─────────────────────── types ───────────────────────
interface Step {
  heap: number[];
  i: number;          // 当前正在 heapify 的节点
  largest: number;    // 当前认为最大的节点
  swapWith: number;   // -1 表示无交换
  desc: string;
  phase: "compare" | "swap" | "done";
}

// ─────────────────────── helpers ─────────────────────
const PRESETS: Record<string, number[]> = {
  "基础示例": [4, 10, 3, 5, 1],
  "需要多次下沉": [1, 16, 4, 10, 14, 7, 9],
  "已满足堆性质": [16, 14, 10, 8, 7, 9, 3],
};

function buildSteps(arr: number[], rootIdx: number): Step[] {
  const steps: Step[] = [];
  const heap = [...arr];
  const n = heap.length;

  function heapifySteps(i: number) {
    let largest = i;
    const l = 2 * i + 1;
    const r = 2 * i + 2;

    steps.push({ heap: [...heap], i, largest, swapWith: -1, desc: `开始 MAX-HEAPIFY，根节点为 [${i}]=${heap[i]}。比较 left=[${l}]=${l < n ? heap[l] : 'NIL'}，right=[${r}]=${r < n ? heap[r] : 'NIL'}`, phase: "compare" });

    if (l < n && heap[l] > heap[largest]) largest = l;
    if (r < n && heap[r] > heap[largest]) largest = r;

    if (largest !== i) {
      steps.push({ heap: [...heap], i, largest, swapWith: largest, desc: `节点 [${largest}]=${heap[largest]} > 当前根 [${i}]=${heap[i]}，执行交换。`, phase: "swap" });
      [heap[i], heap[largest]] = [heap[largest], heap[i]];
      steps.push({ heap: [...heap], i: largest, largest, swapWith: -1, desc: `交换完成，继续向下递归处理位置 [${largest}]`, phase: "compare" });
      heapifySteps(largest);
    } else {
      steps.push({ heap: [...heap], i, largest, swapWith: -1, desc: `节点 [${i}]=${heap[i]} 已是最大，堆性质满足，停止。`, phase: "done" });
    }
  }

  heapifySteps(rootIdx);
  return steps;
}

// ─────────────────────── SVG Tree ────────────────────
const NODE_R = 22;
const SVG_W = 540;
const SVG_H = 200;

function nodePos(idx: number, n: number): [number, number] {
  // 按层计算位置
  let level = 0;
  let pos = idx;
  while (pos > 0) { pos = Math.floor((pos - 1) / 2); level++; }
  const h = Math.floor(Math.log2(n + 1));
  const levelCount = Math.pow(2, level);
  const levelStart = Math.pow(2, level) - 1;
  const posInLevel = idx - levelStart;
  const x = (SVG_W / (levelCount + 1)) * (posInLevel + 1);
  const y = 28 + level * 52;
  return [x, y];
}

function TreeSVG({ heap, highlight, swapWith }: { heap: number[]; highlight: number; swapWith: number }) {
  const n = heap.length;
  const positions = heap.map((_, i) => nodePos(i, n));

  return (
    <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full" style={{ height: SVG_H }}>
      {/* Edges */}
      {heap.map((_, i) => {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        const [px, py] = positions[i];
        return (
          <g key={`edge-${i}`}>
            {l < n && <line x1={px} y1={py} x2={positions[l][0]} y2={positions[l][1]} stroke="var(--color-border)" strokeWidth={1.5} />}
            {r < n && <line x1={px} y1={py} x2={positions[r][0]} y2={positions[r][1]} stroke="var(--color-border)" strokeWidth={1.5} />}
          </g>
        );
      })}
      {/* Nodes */}
      {heap.map((val, i) => {
        const [x, y] = positions[i];
        const isHighlight = i === highlight;
        const isSwap = i === swapWith;
        let fill = "var(--color-bg-card)";
        let stroke = "var(--color-border)";
        if (isHighlight) { fill = "#3b82f6"; stroke = "#1d4ed8"; }
        if (isSwap) { fill = "#f59e0b"; stroke = "#b45309"; }
        return (
          <g key={`node-${i}`}>
            <circle cx={x} cy={y} r={NODE_R} fill={fill} stroke={stroke} strokeWidth={2} />
            <text x={x} y={y} textAnchor="middle" dominantBaseline="central" fontSize={14} fontWeight="bold" fill={isHighlight || isSwap ? "#fff" : "var(--color-text-primary)"}>
              {val}
            </text>
            <text x={x} y={y + NODE_R + 11} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">[{i}]</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─────────────────────── Array View ──────────────────
function ArrayView({ heap, highlight, swapWith }: { heap: number[]; highlight: number; swapWith: number }) {
  return (
    <div className="flex gap-1 items-end flex-wrap justify-center mt-2">
      {heap.map((val, i) => {
        const isHL = i === highlight;
        const isSW = i === swapWith;
        let bg = "var(--color-bg-card)";
        let border = "var(--color-border)";
        if (isHL) { bg = "#3b82f6"; border = "#1d4ed8"; }
        if (isSW) { bg = "#f59e0b"; border = "#b45309"; }
        return (
          <div key={i} className="flex flex-col items-center">
            <div
              className="w-10 h-10 flex items-center justify-center rounded font-bold text-sm border-2 transition-colors duration-300"
              style={{ background: bg, borderColor: border, color: isHL || isSW ? "#fff" : "var(--color-text-primary)" }}
            >{val}</div>
            <span className="text-xs mt-0.5" style={{ color: "var(--color-text-muted)" }}>[{i}]</span>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────── Main Component ──────────────
export default function HeapifyAnimation() {
  const presetKeys = Object.keys(PRESETS);
  const [presetKey, setPresetKey] = useState(presetKeys[1]);
  const [initArr, setInitArr] = useState(PRESETS[presetKeys[1]]);
  const [rootIdx, setRootIdx] = useState(0);
  const [steps, setSteps] = useState<Step[]>([]);
  const [stepIdx, setStepIdx] = useState(0);
  const [generated, setGenerated] = useState(false);

  const generate = useCallback(() => {
    const s = buildSteps(initArr, rootIdx);
    setSteps(s);
    setStepIdx(0);
    setGenerated(true);
  }, [initArr, rootIdx]);

  const selectPreset = (k: string) => {
    setPresetKey(k);
    setInitArr(PRESETS[k]);
    setGenerated(false);
    setSteps([]);
  };

  const cur = steps[stepIdx];

  return (
    <div className="rounded-xl border p-4 space-y-4 select-none" style={{ borderColor: "var(--color-border)", background: "var(--color-bg-card)" }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-bold text-base" style={{ color: "var(--color-text-primary)" }}>
          🫧 MAX-HEAPIFY 下沉动画
        </h3>
        <div className="flex gap-1 flex-wrap">
          {presetKeys.map(k => (
            <button key={k} onClick={() => selectPreset(k)}
              className="px-2 py-1 rounded text-xs font-medium border transition-colors"
              style={{ background: presetKey === k ? "#3b82f6" : "var(--color-bg-secondary)", color: presetKey === k ? "#fff" : "var(--color-text-primary)", borderColor: "var(--color-border)" }}>
              {k}
            </button>
          ))}
        </div>
      </div>

      {/* Config */}
      <div className="flex gap-3 items-center flex-wrap">
        <label className="text-sm" style={{ color: "var(--color-text-muted)" }}>
          从节点索引 <span className="font-bold" style={{ color: "var(--color-text-primary)" }}>[{rootIdx}]={initArr[rootIdx] ?? "?"}</span> 开始 HEAPIFY：
        </label>
        {initArr.map((_, i) => (
          <button key={i} onClick={() => { setRootIdx(i); setGenerated(false); }}
            className="w-7 h-7 rounded text-xs font-bold border-2 transition-colors"
            style={{ background: rootIdx === i ? "#3b82f6" : "var(--color-bg-secondary)", color: rootIdx === i ? "#fff" : "var(--color-text-primary)", borderColor: "var(--color-border)" }}>
            {i}
          </button>
        ))}
        <button onClick={generate}
          className="px-3 py-1 rounded text-sm font-bold transition-all"
          style={{ background: "#3b82f6", color: "#fff" }}>
          ▶ 生成步骤
        </button>
      </div>

      {!generated && (
        <div className="rounded-lg p-4 text-center text-sm" style={{ background: "var(--color-bg-secondary)", color: "var(--color-text-muted)" }}>
          初始数组：[{initArr.join(", ")}]<br />选择起始节点后点击"生成步骤"开始动画
        </div>
      )}

      {generated && cur && (
        <>
          {/* Tree + Array side by side */}
          <div className="grid grid-cols-1 gap-3" style={{ gridTemplateColumns: "1fr" }}>
            <div className="rounded-lg p-3" style={{ background: "var(--color-bg-secondary)" }}>
              <p className="text-xs font-semibold mb-2" style={{ color: "var(--color-text-muted)" }}>🌳 树视图</p>
              <TreeSVG heap={cur.heap} highlight={cur.i} swapWith={cur.swapWith} />
            </div>
            <div className="rounded-lg p-3" style={{ background: "var(--color-bg-secondary)" }}>
              <p className="text-xs font-semibold mb-2" style={{ color: "var(--color-text-muted)" }}>📊 数组视图</p>
              <ArrayView heap={cur.heap} highlight={cur.i} swapWith={cur.swapWith} />
            </div>
          </div>

          {/* Legend */}
          <div className="flex gap-4 text-xs flex-wrap">
            <div className="flex items-center gap-1"><div className="w-4 h-4 rounded" style={{ background: "#3b82f6" }} /><span style={{ color: "var(--color-text-muted)" }}>当前节点 i</span></div>
            <div className="flex items-center gap-1"><div className="w-4 h-4 rounded" style={{ background: "#f59e0b" }} /><span style={{ color: "var(--color-text-muted)" }}>最大子节点（待交换）</span></div>
          </div>

          {/* Description */}
          <div className="rounded-lg px-4 py-3 text-sm font-medium" style={{ background: cur.phase === "done" ? "rgba(34,197,94,0.12)" : "rgba(59,130,246,0.12)", color: "var(--color-text-primary)", borderLeft: `4px solid ${cur.phase === "done" ? "#22c55e" : "#3b82f6"}` }}>
            步骤 {stepIdx + 1}/{steps.length}：{cur.desc}
          </div>

          {/* Controls */}
          <div className="flex gap-2 justify-center">
            <button disabled={stepIdx === 0} onClick={() => setStepIdx(0)}
              className="px-3 py-1.5 rounded text-sm border disabled:opacity-40"
              style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>⏮ 重置</button>
            <button disabled={stepIdx === 0} onClick={() => setStepIdx(s => s - 1)}
              className="px-3 py-1.5 rounded text-sm border disabled:opacity-40"
              style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>◀ 上一步</button>
            <button disabled={stepIdx === steps.length - 1} onClick={() => setStepIdx(s => s + 1)}
              className="px-4 py-1.5 rounded text-sm font-bold disabled:opacity-40"
              style={{ background: "#3b82f6", color: "#fff" }}>下一步 ▶</button>
            <button disabled={stepIdx === steps.length - 1} onClick={() => setStepIdx(steps.length - 1)}
              className="px-3 py-1.5 rounded text-sm border disabled:opacity-40"
              style={{ borderColor: "var(--color-border)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)" }}>⏭ 最终</button>
          </div>

          {/* Progress bar */}
          <div className="w-full rounded-full h-1.5" style={{ background: "var(--color-border)" }}>
            <div className="h-1.5 rounded-full transition-all duration-300" style={{ width: `${((stepIdx + 1) / steps.length) * 100}%`, background: "#3b82f6" }} />
          </div>
        </>
      )}
    </div>
  );
}

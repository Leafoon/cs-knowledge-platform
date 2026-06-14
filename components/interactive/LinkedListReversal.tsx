"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

type Method = "iterative" | "recursive";

interface RevStep {
  // 链表用节点数组表示：nodes[i].next = nodes[i].next index or -1
  nodes: number[];       // 当前展示的节点值序列（已展示的链表状态）
  prevIdx: number;       // 当前 prev 指向的节点位置（-1 = null）
  currIdx: number;       // 当前 curr 指向的节点位置（-1 = done）
  nxtIdx: number;        // 当前 nxt 指向的节点位置（-1 = null）
  arrows: number[];      // nodes[i].next = arrows[i] (-1 = null/reversed)
  label: string;
  phase: "init" | "save-nxt" | "reverse-ptr" | "advance" | "done";
  // 递归专用
  callDepth?: number;
  callStack?: { node: number; returning: boolean }[];
}

function buildIterSteps(vals: number[]): RevStep[] {
  const n = vals.length;
  const steps: RevStep[] = [];

  // arrows[i] = i 号节点（按 vals 顺序）的 next，初始为 i+1
  let arrows = Array.from({ length: n }, (_, i) => (i < n - 1 ? i + 1 : -1));
  let prevIdx = -1;
  let currIdx = 0;
  let nxtIdx = -1;

  steps.push({ nodes: vals.slice(), prevIdx, currIdx, nxtIdx, arrows: arrows.slice(), label: "初始化：prev=null，curr=head（节点0）", phase: "init" });

  while (currIdx !== -1) {
    // save nxt
    nxtIdx = arrows[currIdx];
    arrows = arrows.slice();
    steps.push({ nodes: vals.slice(), prevIdx, currIdx, nxtIdx, arrows: arrows.slice(), label: `① nxt = curr.next = ${nxtIdx === -1 ? "null" : `节点${nxtIdx}（值=${vals[nxtIdx]}）`}（先保存，防断链）`, phase: "save-nxt" });

    // reverse ptr
    arrows[currIdx] = prevIdx;
    steps.push({ nodes: vals.slice(), prevIdx, currIdx, nxtIdx, arrows: arrows.slice(), label: `② curr.next = prev（${prevIdx === -1 ? "null" : `节点${prevIdx}`}），指针反向 ←`, phase: "reverse-ptr" });

    // advance
    prevIdx = currIdx;
    currIdx = nxtIdx;
    steps.push({ nodes: vals.slice(), prevIdx, currIdx, nxtIdx, arrows: arrows.slice(), label: `③ prev=节点${prevIdx}，curr=${currIdx === -1 ? "null（循环结束）" : `节点${currIdx}`}`, phase: "advance" });
  }

  steps.push({ nodes: vals.slice(), prevIdx, currIdx: -1, nxtIdx: -1, arrows: arrows.slice(), label: `完成！新 head = 节点${prevIdx}（值=${vals[prevIdx]}），链表已完全反转`, phase: "done" });
  return steps;
}

function buildRecSteps(vals: number[]): RevStep[] {
  const n = vals.length;
  const steps: RevStep[] = [];
  let arrows = Array.from({ length: n }, (_, i) => (i < n - 1 ? i + 1 : -1));

  // 模拟递归过程
  const callStack: { node: number; returning: boolean }[] = [];

  function push(nodeIdx: number) {
    callStack.push({ node: nodeIdx, returning: false });
    steps.push({
      nodes: vals.slice(), prevIdx: -1, currIdx: nodeIdx, nxtIdx: -1,
      arrows: arrows.slice(),
      label: `递归调用 reverseList(节点${nodeIdx}=${vals[nodeIdx]})，压栈，深入到下一层`,
      phase: "save-nxt",
      callDepth: callStack.length,
      callStack: callStack.map((x) => ({ ...x })),
    });
  }

  // 递归下去
  for (let i = 0; i < n; i++) {
    push(i);
    if (i === n - 1) {
      steps.push({
        nodes: vals.slice(), prevIdx: -1, currIdx: i, nxtIdx: -1,
        arrows: arrows.slice(),
        label: `到达基底：节点${i}=${vals[i]} 是最后节点（next=null），直接返回它作为新头`,
        phase: "done",
        callDepth: callStack.length,
        callStack: callStack.map((x) => ({ ...x })),
      });
      break;
    }
  }

  // 递归回来，逐层反转
  let newArrows = arrows.slice();
  for (let i = n - 2; i >= 0; i--) {
    callStack.pop();
    // 反转 i → i+1 的指针
    newArrows[i + 1] = i;
    newArrows[i] = -1;
    steps.push({
      nodes: vals.slice(), prevIdx: i + 1, currIdx: i, nxtIdx: -1,
      arrows: newArrows.slice(),
      label: `从递归返回到节点${i}=${vals[i]}：令 head.next.next=head（节点${i+1}→节点${i}），head.next=null`,
      phase: "reverse-ptr",
      callDepth: callStack.length + 1,
      callStack: callStack.map((x) => ({ ...x })),
    });
  }

  steps.push({
    nodes: vals.slice(), prevIdx: n - 1, currIdx: -1, nxtIdx: -1,
    arrows: newArrows.slice(),
    label: `所有层返回完毕，新头 = 节点${n - 1}（值=${vals[n - 1]}），链表反转完成`,
    phase: "done",
    callDepth: 0,
    callStack: [],
  });

  return steps;
}

const PHASE_COLORS: Record<string, string> = {
  init:        "text-text-secondary",
  "save-nxt":  "text-blue-600 dark:text-blue-300",
  "reverse-ptr": "text-rose-600 dark:text-rose-300",
  advance:     "text-amber-600 dark:text-amber-400",
  done:        "text-emerald-600 dark:text-emerald-300",
};

export default function LinkedListReversal() {
  const [method, setMethod] = useState<Method>("iterative");
  const [inputVals, setInputVals] = useState("1 2 3 4 5");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const vals = inputVals.trim().split(/\s+/).map(Number).filter((n) => !isNaN(n)).slice(0, 8);

  const steps = React.useMemo(
    () => (method === "iterative" ? buildIterSteps(vals) : buildRecSteps(vals)),
    [method, vals.join(",")]
  );

  const cur = steps[Math.min(step, steps.length - 1)];

  const startPlay = useCallback(() => {
    if (step >= steps.length - 1) setStep(0);
    setPlaying(true);
  }, [step, steps.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => { if (s >= steps.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, 800);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, steps.length]);

  const reset = () => { setStep(0); setPlaying(false); };
  const n = vals.length;

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-rose-500/15 dark:bg-rose-500/20 flex items-center justify-center text-xl">↩️</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">反转链表：迭代 vs 递归</h3>
          <p className="text-xs text-text-secondary">逐步观察指针如何翻转，以及递归调用栈的生长与返回过程</p>
        </div>
      </div>

      {/* 方法切换 + 输入 */}
      <div className="flex flex-wrap gap-3 items-center border-t border-border-subtle pt-3">
        {(["iterative", "recursive"] as Method[]).map((m) => (
          <button key={m} onClick={() => { setMethod(m); reset(); }}
            className={`px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${method === m
              ? "bg-rose-500/20 border-rose-400/60 text-rose-700 dark:text-rose-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            {m === "iterative" ? "🔁 迭代（三指针）" : "🌀 递归"}
          </button>
        ))}
        <label className="flex items-center gap-2 text-xs text-text-secondary ml-auto">
          链表值：
          <input value={inputVals} onChange={(e) => { setInputVals(e.target.value); reset(); }}
            className="w-36 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none focus:border-rose-400/60" />
        </label>
      </div>

      {/* 控制 */}
      <div className="flex items-center gap-2">
        <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
          className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">‹</button>
        <button onClick={playing ? () => setPlaying(false) : startPlay}
          className="px-3 py-1 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-700 dark:text-rose-300 text-xs font-medium transition-colors">
          {playing ? "⏸ 暂停" : "▶ 播放"}
        </button>
        <button onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={step >= steps.length - 1}
          className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">›</button>
        <button onClick={reset} className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs">重置</button>
        <span className="ml-auto text-[10px] text-text-tertiary">{step + 1}/{steps.length}</span>
      </div>

      {/* 进度条 */}
      <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden">
        <div className="h-full bg-rose-500 rounded-full transition-all duration-300"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2 text-xs font-medium ${PHASE_COLORS[cur.phase]}`}>
        步骤 {step + 1}/{steps.length}：{cur.label}
      </div>

      {/* 链表可视化：用 SVG */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 overflow-x-auto">
        <div className="text-[10px] text-text-tertiary mb-2">
          {method === "iterative" ? "指针状态（蓝=prev，橙=nxt，高亮=curr）" : "节点状态（箭头方向=next 指针）"}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {vals.map((v, i) => {
            const isPrev = cur.prevIdx === i;
            const isCurr = cur.currIdx === i;
            const isNxt = cur.nxtIdx === i;
            const ownNext = cur.arrows[i];
            const isReversed = ownNext !== -1 && ownNext < i; // 指向左边表示已反转

            let bg = "bg-bg-secondary border-border-subtle text-text-primary";
            if (isCurr) bg = "bg-amber-500/20 border-amber-400/60 text-amber-700 dark:text-amber-400";
            else if (isPrev) bg = "bg-blue-500/20 border-blue-400/60 text-blue-700 dark:text-blue-300";
            else if (isNxt) bg = "bg-violet-500/20 border-violet-400/60 text-violet-700 dark:text-violet-300";
            else if (isReversed) bg = "bg-rose-500/10 border-rose-400/40 text-rose-700 dark:text-rose-300";

            return (
              <React.Fragment key={i}>
                <div className="flex flex-col items-center gap-0.5">
                  <div className="flex gap-0.5 text-[9px] font-medium h-4 items-end justify-center">
                    {isPrev && <span className="text-blue-600 dark:text-blue-300">prev</span>}
                    {isCurr && <span className="text-amber-600 dark:text-amber-400">curr</span>}
                    {isNxt && <span className="text-violet-600 dark:text-violet-300">nxt</span>}
                  </div>
                  <div className={`w-10 h-10 rounded-lg border flex items-center justify-center text-sm font-bold font-mono transition-all duration-300 ${bg}`}>
                    {v}
                  </div>
                  <span className="text-[8px] font-mono text-text-tertiary">[{i}]</span>
                </div>
                {i < n - 1 && (
                  <div className="flex flex-col items-center justify-center">
                    {cur.arrows[i] === i + 1 ? (
                      <span className="text-text-tertiary text-sm">→</span>
                    ) : cur.arrows[i] === i - 1 || (cur.arrows[i] !== -1 && cur.arrows[i] < i) ? (
                      <span className="text-rose-400 text-sm">←</span>
                    ) : (
                      <span className="text-text-tertiary text-sm opacity-30">✗</span>
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
          <span className="text-xs text-text-tertiary ml-1">
            → {cur.arrows[n - 1] === -1 ? "null" : `节点${cur.arrows[n - 1]}`}
          </span>
        </div>

        {/* 递归调用栈 */}
        {method === "recursive" && cur.callStack && cur.callStack.length > 0 && (
          <div className="mt-3 border-t border-border-subtle pt-2">
            <div className="text-[10px] text-text-tertiary mb-1">调用栈（底部=最早的调用）：</div>
            <div className="flex flex-wrap gap-1">
              {cur.callStack.slice().reverse().map((frame, i) => (
                <div key={i} className="bg-violet-500/10 border border-violet-400/30 rounded px-2 py-0.5 text-[10px] font-mono text-violet-700 dark:text-violet-300">
                  reverseList({vals[frame.node]})
                </div>
              ))}
            </div>
            <div className="text-[9px] text-text-tertiary mt-1">栈深度 = {cur.callStack.length}（空间复杂度 O(n)）</div>
          </div>
        )}
      </div>

      {/* 对比说明 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-text-secondary">
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2.5 space-y-1">
          <div className="font-semibold text-text-primary">🔁 迭代（推荐）</div>
          <div>时间 O(n)，空间 <span className="text-emerald-600 dark:text-emerald-300 font-medium">O(1)</span></div>
          <div>三个指针 prev/curr/nxt，一次扫描完成；需先保存 nxt 再改指针。</div>
        </div>
        <div className="rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2.5 space-y-1">
          <div className="font-semibold text-text-primary">🌀 递归</div>
          <div>时间 O(n)，空间 <span className="text-rose-600 dark:text-rose-300 font-medium">O(n)</span>（调用栈）</div>
          <div>代码简洁但栈帧深，链表很长时有栈溢出风险（Python 默认递归深度 1000）。</div>
        </div>
      </div>
    </div>
  );
}

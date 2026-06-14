"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 构建含环链表 ────────────────────────────────────────────────────────────────
// 节点用数字索引表示，实际链表结构用 next[] 数组
// 默认：0→1→2→3→4→2（环入口 = 2）

interface FloydStep {
  slow: number;
  fast: number;
  phase: "chase" | "find-entry" | "done";
  label: string;
  meetAt?: number;
}

function buildFloydSteps(listLen: number, cycleEntry: number): FloydStep[] {
  // next[i] = 下一个节点的索引，-1 表示 null
  const next: number[] = Array.from({ length: listLen }, (_, i) =>
    i === listLen - 1 ? cycleEntry : i + 1
  );
  // next[cycleEntry-1] already connects to cycleEntry naturally

  const steps: FloydStep[] = [];
  let slow = 0, fast = 0;
  steps.push({ slow, fast, phase: "chase", label: "初始化：slow=0，fast=0，均从头出发" });

  // Phase 1: detect
  let meetAt = -1;
  for (let iter = 0; iter < listLen * 3; iter++) {
    if (next[fast] === -1 || next[next[fast]] === -1) {
      steps.push({ slow, fast, phase: "done", label: "fast 到达 null，链表无环" });
      return steps;
    }
    slow = next[slow];
    fast = next[next[fast]];
    const phaseLabel = `第${iter + 1}步：slow→${slow}，fast→${fast}`;
    if (slow === fast) {
      meetAt = slow;
      steps.push({ slow, fast, phase: "chase", label: phaseLabel + `  🤝 相遇于节点 ${meetAt}！`, meetAt });
      break;
    }
    steps.push({ slow, fast, phase: "chase", label: phaseLabel });
  }

  if (meetAt === -1) {
    steps.push({ slow, fast, phase: "done", label: "未检测到环" });
    return steps;
  }

  // Phase 2: find entry
  steps.push({ slow, fast: 0, phase: "find-entry", meetAt, label: `快指针重置到头部（节点0），慢指针留在相遇点 ${slow}，两者各走1步` });
  fast = 0;
  for (let iter = 0; iter < listLen * 2; iter++) {
    slow = next[slow];
    fast = next[fast];
    const label = `第${iter + 1}步：slow→${slow}，fast→${fast}`;
    if (slow === fast) {
      steps.push({ slow, fast, phase: "done", meetAt, label: label + `  ✅ 相遇于环入口 ${slow}！（= 节点 ${cycleEntry}）` });
      break;
    }
    steps.push({ slow, fast, phase: "find-entry", meetAt, label });
  }

  return steps;
}

// 节点在圆形布局上的位置
function nodePos(idx: number, total: number, cx: number, cy: number, r: number) {
  // 前面一段（0..cycleEntry-1）在左侧直线排列，环部分在圆上
  return { x: 0, y: 0 }; // placeholder – computed inline
}

export default function FloydCycleDetection() {
  const [listLen, setListLen] = useState(8);
  const [cycleEntry, setCycleEntry] = useState(3);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const safeEntry = Math.max(0, Math.min(cycleEntry, listLen - 2));
  const steps = React.useMemo(() => buildFloydSteps(listLen, safeEntry), [listLen, safeEntry]);
  const cur = steps[Math.min(step, steps.length - 1)];

  const startPlay = useCallback(() => {
    if (step >= steps.length - 1) setStep(0);
    setPlaying(true);
  }, [step, steps.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => { if (s >= steps.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, 700);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, steps.length]);

  const reset = () => { setStep(0); setPlaying(false); };

  // ── SVG 布局计算 ─────────────────────────────────────────────────────────────
  // 链表头部（0..cycleEntry-1）横向排列，环部分（cycleEntry..listLen-1）圆形
  const W = 520, H = 200;
  const tailCount = safeEntry;                        // 不在环上的节点数
  const cycleCount = listLen - safeEntry;              // 在环上的节点数
  const nodeR = 18;
  const tailSpacingX = tailCount > 0 ? Math.min(70, (W * 0.45) / (tailCount + 1)) : 0;
  const tailStartX = 30;
  const tailY = H / 2;
  // 环圆心
  const cxCircle = tailStartX + (tailCount) * tailSpacingX + 80;
  const cyCircle = H / 2;
  const circleR = Math.min(65, Math.max(40, cycleCount * 14));

  function getTailPos(i: number) {
    return { x: tailStartX + i * tailSpacingX, y: tailY };
  }

  function getCyclePos(j: number) {
    // j = cycleCount means full circle back
    const angle = -Math.PI / 2 + (2 * Math.PI * j) / cycleCount;
    return { x: cxCircle + circleR * Math.cos(angle), y: cyCircle + circleR * Math.sin(angle) };
  }

  function getPos(idx: number) {
    if (idx < safeEntry) return getTailPos(idx);
    return getCyclePos(idx - safeEntry);
  }

  const phaseColor = cur.phase === "find-entry" ? "text-violet-600 dark:text-violet-300" :
                     cur.phase === "done" ? "text-emerald-600 dark:text-emerald-300" :
                     "text-blue-600 dark:text-blue-300";

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-violet-500/15 dark:bg-violet-500/20 flex items-center justify-center text-xl">🐢</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">Floyd 判环算法可视化</h3>
          <p className="text-xs text-text-secondary">慢指针走1步，快指针走2步，相遇后定位环入口</p>
        </div>
      </div>

      {/* 参数 */}
      <div className="flex flex-wrap gap-4 items-center border-t border-border-subtle pt-3">
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          链表总长：
          <input type="number" min={4} max={12} value={listLen}
            onChange={(e) => { setListLen(Number(e.target.value)); reset(); }}
            className="w-14 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
        </label>
        <label className="flex items-center gap-2 text-xs text-text-secondary">
          环入口索引（0-indexed）：
          <input type="number" min={0} max={listLen - 2} value={cycleEntry}
            onChange={(e) => { setCycleEntry(Number(e.target.value)); reset(); }}
            className="w-14 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1 text-xs font-mono text-text-primary outline-none" />
        </label>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">‹</button>
          <button onClick={playing ? () => setPlaying(false) : startPlay}
            className="px-3 py-1 rounded-lg bg-violet-500/15 hover:bg-violet-500/25 text-violet-700 dark:text-violet-300 text-xs font-medium transition-colors">
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={step >= steps.length - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">›</button>
          <button onClick={reset} className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors">重置</button>
        </div>
      </div>

      {/* 进度条 + 步骤说明 */}
      <div>
        <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden mb-2">
          <div className="h-full bg-violet-500 rounded-full transition-all duration-300"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
        </div>
        <div className={`rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2 text-xs font-medium ${phaseColor}`}>
          步骤 {step + 1}/{steps.length}：{cur.label}
        </div>
      </div>

      {/* SVG 链表图 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary overflow-hidden">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 220 }}>
          {/* 尾部箭头 */}
          {Array.from({ length: tailCount }).map((_, i) => {
            const from = getTailPos(i);
            const to = i === tailCount - 1 ? getCyclePos(0) : getTailPos(i + 1);
            const dx = to.x - from.x, dy = to.y - from.y;
            const len = Math.sqrt(dx * dx + dy * dy);
            const ex = from.x + (dx / len) * nodeR;
            const ey = from.y + (dy / len) * nodeR;
            const fx = to.x - (dx / len) * (nodeR + 4);
            const fy = to.y - (dy / len) * (nodeR + 4);
            return (
              <line key={`t${i}`} x1={ex} y1={ey} x2={fx} y2={fy}
                stroke="#6b7280" strokeWidth={1.5} markerEnd="url(#arrow)" />
            );
          })}

          {/* 环箭头 */}
          {Array.from({ length: cycleCount }).map((_, j) => {
            const from = getCyclePos(j);
            const to = getCyclePos((j + 1) % cycleCount);
            const dx = to.x - from.x, dy = to.y - from.y;
            const len = Math.sqrt(dx * dx + dy * dy);
            const ex = from.x + (dx / len) * nodeR;
            const ey = from.y + (dy / len) * nodeR;
            const fx = to.x - (dx / len) * (nodeR + 4);
            const fy = to.y - (dy / len) * (nodeR + 4);
            return (
              <line key={`c${j}`} x1={ex} y1={ey} x2={fx} y2={fy}
                stroke="#6b7280" strokeWidth={1.5} markerEnd="url(#arrow)" />
            );
          })}

          {/* 节点 */}
          {Array.from({ length: listLen }).map((_, i) => {
            const { x, y } = getPos(i);
            const isSlow = cur.slow === i;
            const isFast = cur.fast === i;
            const isBoth = isSlow && isFast;
            const isEntry = i === safeEntry;
            const isMeet = cur.meetAt === i;

            const fill = isBoth ? "#f59e0b" :
                         isFast ? "#8b5cf6" :
                         isSlow ? "#3b82f6" :
                         isEntry ? "#10b981" :
                         "#6b7280";
            const stroke = isEntry ? "#10b981" : isMeet ? "#f59e0b" : "#374151";
            const strokeW = isEntry || isMeet ? 3 : 1.5;

            return (
              <g key={i}>
                <circle cx={x} cy={y} r={nodeR} fill={fill} fillOpacity={0.85} stroke={stroke} strokeWidth={strokeW} />
                <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
                  fontSize={13} fontWeight="bold" fill="white">{i}</text>
                {isEntry && (
                  <text x={x} y={y + nodeR + 10} textAnchor="middle" fontSize={9} fill="#10b981">入口</text>
                )}
              </g>
            );
          })}

          {/* 箭头 marker */}
          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L0,6 L6,3 z" fill="#6b7280" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* 图例 */}
      <div className="flex flex-wrap gap-4 text-xs text-text-secondary">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-full bg-blue-500/80" />
          <span>慢指针 slow</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-full bg-violet-500/80" />
          <span>快指针 fast</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-full bg-amber-500/80" />
          <span>两指针相遇</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-4 rounded-full bg-emerald-500/80" />
          <span>环入口</span>
        </div>
      </div>

      {/* 数学公式 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2.5 text-xs text-text-secondary space-y-1">
        <div className="font-semibold text-text-primary">Floyd 判环数学原理</div>
        <div>设头到环入口距离 = a，环长 = b，相遇时慢指针在环内走了 c 步。</div>
        <div className="font-mono text-text-primary bg-bg-secondary rounded px-2 py-1 text-[11px]">
          2(a+c) = a + b·k + c  →  a = b·k - c
        </div>
        <div>相遇后将快指针重置到头，两者各走 1 步：走 a 步后同时到达<span className="text-emerald-600 dark:text-emerald-300">环入口</span>。</div>
      </div>
    </div>
  );
}

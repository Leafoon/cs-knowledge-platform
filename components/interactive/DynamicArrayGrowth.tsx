"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 类型 ──────────────────────────────────────────────────────────────────────
interface AppendEvent {
  step: number;
  value: number;
  sizeBefore: number;
  capBefore: number;
  resized: boolean;
  copiesDone: number;  // copies in THIS resize (0 if no resize)
  totalCopies: number; // cumulative total copies
  sizeAfter: number;
  capAfter: number;
}

// ── 模拟动态数组追加过程 ──────────────────────────────────────────────────────
function simulateAppends(n: number, factor: number): AppendEvent[] {
  const events: AppendEvent[] = [];
  let size = 0;
  let cap = 1;
  let totalCopies = 0;

  for (let i = 0; i < n; i++) {
    const sizeBefore = size;
    const capBefore = cap;
    let resized = false;
    let copiesDone = 0;

    if (size === cap) {
      copiesDone = size;
      totalCopies += copiesDone;
      cap = Math.ceil(cap * factor);
      resized = true;
    }
    size++;

    events.push({
      step: i,
      value: i + 1,
      sizeBefore,
      capBefore,
      resized,
      copiesDone,
      totalCopies,
      sizeAfter: size,
      capAfter: cap,
    });
  }
  return events;
}

// ── 颜色 ──────────────────────────────────────────────────────────────────────
const CELL_W = 28;
const CELL_H = 30;
const MAX_VISIBLE_CELLS = 24;

export default function DynamicArrayGrowth() {
  const [factor, setFactor] = useState<number>(2);
  const [totalN, setTotalN] = useState<number>(20);
  const [step, setStep] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 预计算所有步骤
  const events = React.useMemo(
    () => simulateAppends(totalN, factor),
    [totalN, factor]
  );

  const cur = events[Math.min(step, events.length - 1)];

  const startPlay = useCallback(() => {
    if (step >= events.length - 1) setStep(0);
    setPlaying(true);
  }, [step, events.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => {
          if (s >= events.length - 1) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, 500);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, events.length]);

  const reset = () => {
    setStep(0);
    setPlaying(false);
  };

  // 准备绘制：当前分配的槽位
  const displayCap = Math.min(cur.capAfter, MAX_VISIBLE_CELLS);
  const displaySize = Math.min(cur.sizeAfter, MAX_VISIBLE_CELLS);

  // 摊销代价历史图（从第 0 步到当前步）
  const histEvents = events.slice(0, step + 1);
  const maxTotalCopies = events[events.length - 1]?.totalCopies ?? 1;
  const maxN = totalN;

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-5">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-blue-500/15 dark:bg-blue-500/20 flex items-center justify-center text-xl">
          📦
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">动态数组扩容可视化</h3>
          <p className="text-xs text-text-secondary">观察倍增法的扩容时机、摊销代价与累计复制次数</p>
        </div>
      </div>

      {/* 控制栏 */}
      <div className="flex flex-wrap gap-3 items-center border-t border-border-subtle pt-4">
        {/* 扩容因子 */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-secondary font-medium">扩容因子：</span>
          {[1.5, 2, 3].map((f) => (
            <button
              key={f}
              onClick={() => { setFactor(f); reset(); }}
              className={`px-2.5 py-1 rounded-lg border text-xs font-mono font-medium transition-all ${factor === f
                ? "bg-blue-500/20 border-blue-400/60 text-blue-600 dark:text-blue-300"
                : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"
                }`}
            >
              ×{f}
            </button>
          ))}
        </div>

        {/* 追加次数 */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-text-secondary">追加元素：</span>
          <input
            type="range" min={8} max={32} value={totalN}
            onChange={(e) => { setTotalN(Number(e.target.value)); reset(); }}
            className="w-24 accent-blue-500"
          />
          <span className="font-mono text-blue-600 dark:text-blue-300 font-bold">{totalN}</span>
        </div>

        {/* 播放控制 */}
        <div className="flex items-center gap-2 ml-auto">
          <button
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center transition-colors"
          >
            ‹
          </button>
          <button
            onClick={playing ? () => setPlaying(false) : startPlay}
            className="px-3 py-1 rounded-lg bg-blue-500/15 dark:bg-blue-500/20 hover:bg-blue-500/25 text-blue-600 dark:text-blue-300 text-xs font-medium transition-colors"
          >
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button
            onClick={() => setStep((s) => Math.min(events.length - 1, s + 1))}
            disabled={step >= events.length - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center transition-colors"
          >
            ›
          </button>
          <button
            onClick={reset}
            className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors"
          >
            重置
          </button>
        </div>
      </div>

      {/* 进度条 */}
      <div>
        <div className="flex justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {step + 1} / {events.length}（追加第 {cur.value} 个元素）</span>
          {cur.resized && (
            <span className="text-amber-600 dark:text-amber-400 font-medium">
              ⚡ 扩容！{cur.capBefore} → {cur.capAfter}（复制 {cur.copiesDone} 个元素）
            </span>
          )}
        </div>
        <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${((step + 1) / events.length) * 100}%` }}
          />
        </div>
      </div>

      {/* 数组可视化 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
        <div className="text-xs text-text-secondary mb-3 flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-blue-500 inline-block" />
            已使用（size = {cur.sizeAfter}）
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded border border-border-strong bg-bg-secondary inline-block" />
            空槽（capacity = {cur.capAfter}）
          </span>
          <span className="text-text-tertiary ml-auto text-[10px]">
            {cur.capAfter > MAX_VISIBLE_CELLS ? `（仅显示前 ${MAX_VISIBLE_CELLS} 个槽位）` : ""}
          </span>
        </div>
        <div className="flex gap-1 flex-wrap">
          {Array.from({ length: displayCap }).map((_, i) => {
            const isFilled = i < displaySize;
            const isNew = isFilled && i === cur.sizeAfter - 1;
            const wasCopied = cur.resized && i < cur.copiesDone;
            return (
              <div
                key={i}
                className={`
                  flex flex-col items-center justify-center rounded text-xs font-mono
                  transition-all duration-300
                  ${isFilled
                    ? isNew
                      ? "bg-emerald-500/70 text-white border border-emerald-400 w-7 h-8 font-bold"
                      : wasCopied
                        ? "bg-amber-500/60 text-white border border-amber-400 w-7 h-8"
                        : "bg-blue-500 text-white border border-blue-400 w-7 h-8"
                    : "bg-bg-secondary border border-border-subtle text-text-tertiary/40 w-7 h-8"
                  }`}
              >
                {isFilled ? (i + 1) : ""}
                <span className="text-[8px] opacity-60">{i}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 统计信息 */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "当前 size", value: cur.sizeAfter, color: "text-blue-600 dark:text-blue-300" },
          { label: "当前 capacity", value: cur.capAfter, color: "text-violet-600 dark:text-violet-300" },
          { label: "累计复制次数", value: cur.totalCopies, color: "text-amber-600 dark:text-amber-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg bg-bg-tertiary border border-border-subtle p-3 text-center">
            <div className="text-xs text-text-tertiary mb-1">{label}</div>
            <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* 摊销代价折线图 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
        <div className="text-xs font-semibold text-text-primary mb-2">
          📊 摊销代价分析（每次 append 的代价 vs 平均代价）
        </div>
        <svg viewBox={`0 0 ${maxN * 14} 80`} className="w-full h-24">
          {/* 背景网格 */}
          {[0, 25, 50, 75].map((y) => (
            <line key={y} x1="0" y1={y} x2={maxN * 14} y2={y}
              stroke="currentColor" strokeOpacity="0.1" strokeWidth="1" />
          ))}
          {/* 各步骤即时代价柱 */}
          {histEvents.map((e, i) => {
            const cost = 1 + (e.resized ? e.copiesDone : 0);
            const barH = Math.min((cost / (maxN * 0.6)) * 75, 75);
            return (
              <rect
                key={i}
                x={i * 14 + 2}
                y={75 - barH}
                width={10}
                height={barH}
                fill={e.resized ? "#f59e0b" : "#3b82f6"}
                opacity={0.7}
                rx={1}
              />
            );
          })}
          {/* 平均代价折线 */}
          {histEvents.length > 1 && (
            <polyline
              points={histEvents.map((e, i) => {
                const avg = (e.totalCopies + i + 1) / (i + 1);
                const yVal = 75 - Math.min((avg / (maxN * 0.6)) * 75, 75);
                return `${i * 14 + 7},${yVal}`;
              }).join(" ")}
              fill="none"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          )}
        </svg>
        <div className="flex gap-4 text-[10px] text-text-tertiary mt-1">
          <span className="flex items-center gap-1">
            <span className="w-3 h-2 rounded bg-blue-500 inline-block" /> 普通追加（代价=1）
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-2 rounded bg-amber-500 inline-block" /> 扩容时追加
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-emerald-500 inline-block" /> 累计平均代价
          </span>
          <span className="ml-auto">
            最终摊销均值：<span className="font-mono text-text-primary">{((cur.totalCopies + step + 1) / (step + 1)).toFixed(2)}</span>
          </span>
        </div>
      </div>

      {/* 说明 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs space-y-1 text-text-secondary">
        <div>
          <span className="font-medium text-text-primary">为何选 ×{factor}？</span>
          {factor === 2 ? " 每次扩容复制次数之和 = 1+2+4+…+n/2 < n，总代价 O(n)，摊销 O(1)。" :
           factor === 1.5 ? " ×1.5 也是摊销 O(1)，但内存峰值比 ×2 低（最大浪费 33% vs 50%）。" :
           " ×3 也是摊销 O(1)，但内存峰值可能浪费 66%，一般不推荐。"}
        </div>
        <div>
          <span className="text-amber-600 dark:text-amber-400 font-medium">金色柱 = 扩容步骤</span>
          （代价 = 1 + 复制旧元素数）；绿线最终稳定在约 {factor === 2 ? "3" : factor === 1.5 ? "4" : "2"} 附近 = 摊销常数。
        </div>
      </div>
    </div>
  );
}

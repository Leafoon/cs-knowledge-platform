"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 类型定义 ─────────────────────────────────────────────────────────────────
interface StackFrame {
  id: number;
  n: number;
  acc?: number;           // 仅在尾递归模式下使用
  returnVal?: number;     // 已计算出返回值时填充
  state: "active" | "returning" | "done"; // 当前帧状态
}

type Mode = "regular" | "tail";

// ── 工具函数 ─────────────────────────────────────────────────────────────────
function buildRegularFrames(n: number): StackFrame[] {
  const frames: StackFrame[] = [];
  for (let i = n; i >= 0; i--) {
    frames.push({ id: i, n: i, state: "active" });
  }
  return frames; // 从 f(n) 到 f(0)
}

function buildTailFrames(n: number): StackFrame[] {
  const frames: StackFrame[] = [];
  for (let i = 0; i <= n; i++) {
    const acc = Array.from({ length: i }, (_, k) => n - k).reduce(
      (p, c) => p * c,
      1
    );
    frames.push({ id: i, n: n - i, acc, state: "active" });
  }
  return frames;
}

// ── 颜色工具 ─────────────────────────────────────────────────────────────────
const FRAME_COLORS = [
  "from-blue-500/20 to-blue-600/10 border-blue-400/40",
  "from-indigo-500/20 to-indigo-600/10 border-indigo-400/40",
  "from-violet-500/20 to-violet-600/10 border-violet-400/40",
  "from-purple-500/20 to-purple-600/10 border-purple-400/40",
  "from-fuchsia-500/20 to-fuchsia-600/10 border-fuchsia-400/40",
  "from-sky-500/20 to-sky-600/10 border-sky-400/40",
];

const ACTIVE_COLOR = [
  "text-blue-500 dark:text-blue-300",
  "text-indigo-500 dark:text-indigo-300",
  "text-violet-500 dark:text-violet-300",
  "text-purple-500 dark:text-purple-300",
  "text-fuchsia-500 dark:text-fuchsia-300",
  "text-sky-600 dark:text-sky-300",
];

export default function RecursionCallStack() {
  const [n, setN] = useState(4);
  const [mode, setMode] = useState<Mode>("regular");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 根据 n 和 mode 计算全部步骤
  const steps = React.useMemo(() => {
    if (mode === "regular") {
      // 步骤：依次压入 f(n), f(n-1), ..., f(0)，再依次弹出（返回值）
      const pushPhase: StackFrame[][] = [];
      for (let k = 0; k <= n; k++) {
        const frames: StackFrame[] = [];
        for (let i = k; i >= 0; i--) {
          frames.push({ id: i, n: i, state: "active" });
        }
        pushPhase.push(frames);
      }
      // 弹出阶段：f(0) 先返回 1，然后 f(1) 返回 1，f(2) 返回 2，...
      const popPhase: StackFrame[][] = [];
      const retVals: Record<number, number> = {};
      for (let k = 0; k <= n; k++) {
        retVals[k] = k === 0 ? 1 : k * retVals[k - 1];
        const frames: StackFrame[] = [];
        for (let i = n; i > k; i--) {
          frames.push({ id: i, n: i, state: "active" });
        }
        if (k < n) {
          frames.push({ id: k, n: k, returnVal: retVals[k], state: "returning" });
        }
        popPhase.push(frames);
      }
      return [...pushPhase, ...popPhase];
    } else {
      // 尾递归：每步只有一个"当前"帧（复用同一帧）
      const allSteps: StackFrame[][] = [];
      let acc = 1;
      for (let i = n; i >= 0; i--) {
        allSteps.push([{ id: n - i, n: i, acc, state: "active" }]);
        if (i > 0) acc *= i;
      }
      // 最终返回 acc
      allSteps.push([{ id: n + 1, n: 0, acc, returnVal: acc, state: "returning" }]);
      return allSteps;
    }
  }, [n, mode]);

  const totalSteps = steps.length;
  const currentFrames = steps[Math.min(step, totalSteps - 1)] ?? [];
  const maxDepth = mode === "regular" ? n + 1 : 1;

  // 自动播放
  const startPlay = useCallback(() => {
    if (step >= totalSteps - 1) setStep(0);
    setPlaying(true);
  }, [step, totalSteps]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => {
          if (s >= totalSteps - 1) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, 600);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, totalSteps]);

  const reset = () => {
    setStep(0);
    setPlaying(false);
  };

  // 判断当前阶段
  const isPushPhase =
    mode === "regular" && step <= n;
  const isPopPhase = mode === "regular" && step > n;

  // 计算当前阶段注释
  const phaseLabel = () => {
    if (mode === "tail") {
      if (step <= n) return `调用 factorial_tail(${currentFrames[0]?.n}, ${currentFrames[0]?.acc})`;
      return `返回最终结果 ${currentFrames[0]?.acc}`;
    }
    if (isPushPhase) {
      return `压栈：调用 factorial(${n - step})`;
    }
    const popN = step - n - 1;
    if (popN < 0) return `f(0) 到达递归基，返回 1`;
    return `弹栈：f(${popN}) 返回 ${(() => {
      let v = 1;
      for (let i = 1; i <= popN; i++) v *= i;
      return v;
    })()}`;
  };

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-6 my-6 shadow-sm">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-blue-500/20 flex items-center justify-center text-xl">
          📚
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">递归调用栈可视化</h3>
          <p className="text-xs text-text-secondary">观察普通递归 vs 尾递归的栈帧行为</p>
        </div>
      </div>

      {/* 控制栏 */}
      <div className="flex flex-wrap gap-3 mb-5">
        {/* 模式切换 */}
        <div className="flex rounded-lg overflow-hidden border border-border-subtle">
          <button
            onClick={() => { setMode("regular"); reset(); }}
            className={`px-3 py-1.5 text-xs font-mono font-medium transition-colors ${mode === "regular" ? "bg-blue-500/30 text-blue-700 dark:text-blue-200" : "bg-bg-tertiary text-text-secondary hover:text-text-secondary"}`}
          >
            普通递归
          </button>
          <button
            onClick={() => { setMode("tail"); reset(); }}
            className={`px-3 py-1.5 text-xs font-mono font-medium transition-colors ${mode === "tail" ? "bg-emerald-500/30 text-emerald-700 dark:text-emerald-200" : "bg-bg-tertiary text-text-secondary hover:text-text-secondary"}`}
          >
            尾递归（TCO）
          </button>
        </div>

        {/* n 控制 */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-secondary font-mono">n =</span>
          <input
            type="range" min={1} max={6} value={n}
            onChange={(e) => { setN(Number(e.target.value)); reset(); }}
            className="w-24 accent-blue-400"
          />
          <span className="text-sm font-bold text-blue-300 font-mono w-4">{n}</span>
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
            className="px-3 py-1 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-700 dark:text-blue-200 text-xs font-medium transition-colors"
          >
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button
            onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}
            disabled={step >= totalSteps - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center transition-colors"
          >
            ›
          </button>
          <button
            onClick={reset}
            className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-bg-tertiary text-text-secondary text-xs transition-colors"
          >
            重置
          </button>
        </div>
      </div>

      {/* 步骤进度条 */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {step + 1} / {totalSteps}</span>
          <span className="text-blue-500 dark:text-blue-300 font-mono">{phaseLabel()}</span>
        </div>
        <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-400 rounded-full transition-all duration-300"
            style={{ width: `${((step + 1) / totalSteps) * 100}%` }}
          />
        </div>
      </div>

      {/* 主体：调用栈 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* 左：可视化栈 */}
        <div>
          <div className="text-xs text-text-secondary mb-2 flex items-center gap-1">
            <span>调用栈（栈顶 = 当前执行）</span>
            {mode === "tail" && (
              <span className="bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded text-xs ml-auto">
                O(1) 空间
              </span>
            )}
            {mode === "regular" && (
              <span className="bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded text-xs ml-auto">
                O(n) 空间：{currentFrames.length} 帧
              </span>
            )}
          </div>

          <div className="min-h-[240px] flex flex-col-reverse gap-1.5 justify-end p-3 rounded-xl bg-bg-tertiary border border-border-subtle">
            {currentFrames.length === 0 && (
              <div className="text-text-tertiary text-xs text-center py-8">栈为空（计算完毕）</div>
            )}
            {currentFrames.map((frame, idx) => {
              const colorIdx = frame.n % FRAME_COLORS.length;
              const isTop = idx === currentFrames.length - 1;
              const isReturning = frame.state === "returning";
              return (
                <div
                  key={`${frame.id}-${mode}`}
                  className={`
                    rounded-lg border p-2.5 bg-gradient-to-r transition-all duration-300
                    ${FRAME_COLORS[colorIdx]}
                    ${isTop ? "ring-1 ring-border-strong" : ""}
                    ${isReturning ? "opacity-60 border-dashed" : ""}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-mono text-xs font-bold ${ACTIVE_COLOR[colorIdx]}`}>
                      {mode === "regular"
                        ? `factorial(${frame.n})`
                        : `factorial_tail(${frame.n}, ${frame.acc})`
                      }
                    </span>
                    <div className="flex gap-1.5 items-center">
                      {isTop && !isReturning && (
                        <span className="text-[10px] bg-border-subtle px-1.5 py-0.5 rounded text-text-primary font-mono">
                          ← 执行中
                        </span>
                      )}
                      {isReturning && (
                        <span className="text-[10px] bg-yellow-400/20 px-1.5 py-0.5 rounded text-yellow-300 font-mono">
                          返回 {frame.returnVal}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-[10px] text-text-tertiary mt-0.5 font-mono">
                    {mode === "regular"
                      ? (frame.n === 0
                          ? "触底 → return 1"
                          : `等待 factorial(${frame.n - 1}) 返回...`)
                      : (frame.n === 0
                          ? `到达递归基 → return ${frame.acc}`
                          : `计算 factorial_tail(${frame.n - 1}, ${frame.n} × ${frame.acc} = ${frame.n * (frame.acc ?? 1)})`)
                    }
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 右：代码高亮 + 说明 */}
        <div>
          <div className="text-xs text-text-secondary mb-2">
            {mode === "regular" ? "普通递归代码" : "尾递归代码（累加器版）"}
          </div>
          <div className="bg-bg-tertiary rounded-xl border border-border-subtle p-3 font-mono text-xs leading-relaxed text-text-secondary">
            {mode === "regular" ? (
              <pre className="whitespace-pre-wrap">{`def factorial(n):
    if n == 0:
        return 1          # ← 递归基
    return n * factorial(n - 1)
    # ↑ 调用后还要做乘法
    # → 必须等子调用返回
    # → 每次调用都保留栈帧`}</pre>
            ) : (
              <pre className="whitespace-pre-wrap">{`def factorial_tail(n, acc=1):
    if n == 0:
        return acc        # ← 直接返回结果
    return factorial_tail(
        n - 1, n * acc   # ← 最后一步就是调用
    )                     # → 可以复用同一栈帧
    # → O(1) 空间（需编译器支持TCO）`}</pre>
            )}
          </div>

          {/* 对比说明 */}
          <div className="mt-3 rounded-xl border border-border-subtle bg-bg-tertiary p-3">
            <div className="text-xs font-semibold text-text-primary mb-2">关键差异对比</div>
            <div className="space-y-1.5">
              <div className="flex items-start gap-2">
                <span className="text-blue-400 text-xs mt-0.5">普通</span>
                <span className="text-xs text-text-secondary">调用后还有操作（乘法），必须保留栈帧，占用 <span className="text-blue-500 dark:text-blue-300">O(n)</span> 空间</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-emerald-400 text-xs mt-0.5">尾递归</span>
                <span className="text-xs text-text-secondary">调用即最终结果，编译器（C++/Scala等）可复用栈帧，仅 <span className="text-emerald-300">O(1)</span> 空间</span>
              </div>
              <div className="mt-2 p-2 rounded bg-yellow-500/10 border border-yellow-500/20">
                <span className="text-yellow-300 text-xs">⚠️ CPython 不优化尾递归，写成尾递归形式仍占 O(n) 栈空间。</span>
              </div>
            </div>
          </div>

          {/* 栈深度指示器 */}
          <div className="mt-3">
            <div className="flex justify-between text-xs text-text-tertiary mb-1">
              <span>当前栈深度</span>
              <span className="font-mono">{currentFrames.length} / {maxDepth}</span>
            </div>
            <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${mode === "tail" ? "bg-emerald-400" : "bg-blue-400"}`}
                style={{ width: `${(currentFrames.length / Math.max(maxDepth, 1)) * 100}%` }}
              />
            </div>
            <div className="text-[10px] text-text-tertiary mt-0.5">
              {mode === "tail" ? "最大深度：1（始终复用同一帧）" : `最大深度：${n + 1} 帧`}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

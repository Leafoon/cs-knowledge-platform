"use client";

import React, { useState, useCallback } from "react";

/** 单调队列：滑动窗口最大值 O(n) 演示 */

const PRESETS: Record<string, { nums: number[]; k: number }> = {
  "经典示例": { nums: [1, 3, -1, -3, 5, 3, 6, 7], k: 3 },
  "全相同": { nums: [4, 4, 4, 4, 4], k: 2 },
  "递减序列": { nums: [8, 6, 4, 2, 1], k: 3 },
  "峰谷交替": { nums: [2, 7, 3, 8, 1, 9, 2, 5], k: 4 },
};

interface Step {
  i: number;                  // 当前遍历下标
  action: "pop_back" | "pop_front" | "push" | "record";
  deqSnap: number[];           // deque 快照（存下标）
  resultSnap: (number | null)[];
  description: string;
}

function buildSteps(nums: number[], k: number): Step[] {
  const n = nums.length;
  const result: (number | null)[] = Array(n).fill(null);
  const dq: number[] = [];
  const steps: Step[] = [];

  for (let i = 0; i < n; i++) {
    // pop_back：弹出所有不如 nums[i] 大的队尾
    while (dq.length > 0 && nums[dq[dq.length - 1]] <= nums[i]) {
      const out = dq.pop()!;
      steps.push({
        i, action: "pop_back", deqSnap: [...dq], resultSnap: [...result],
        description: `队尾 [${out}]=${nums[out]} ≤ ${nums[i]}，弹出（它不可能成为任何窗口的最大值）`,
      });
    }

    // push
    dq.push(i);
    steps.push({
      i, action: "push", deqSnap: [...dq], resultSnap: [...result],
      description: `将下标 ${i}（值 ${nums[i]}）从队尾加入`,
    });

    // pop_front：队首超出窗口
    if (dq[0] < i - k + 1) {
      dq.shift();
      steps.push({
        i, action: "pop_front", deqSnap: [...dq], resultSnap: [...result],
        description: `队首下标 ${dq.length > 0 ? dq[0] - 1 : i - k} 已超出窗口范围（< ${i - k + 1}），弹出`,
      });
    }

    // record：窗口已满
    if (i >= k - 1) {
      result[i] = nums[dq[0]];
      steps.push({
        i, action: "record", deqSnap: [...dq], resultSnap: [...result],
        description: `窗口 [${i - k + 1}..${i}] 最大值 = nums[${dq[0]}] = ${nums[dq[0]]}`,
      });
    }
  }

  return steps;
}

export default function DequeWindowMaxDemo() {
  const [preset, setPreset] = useState("经典示例");
  const [stepIdx, setStepIdx] = useState(0);

  const { nums, k } = PRESETS[preset];
  const steps = buildSteps(nums, k);
  const step = steps[stepIdx];

  const reset = () => setStepIdx(0);
  const prev = () => setStepIdx((s) => Math.max(0, s - 1));
  const next = () => setStepIdx((s) => Math.min(steps.length - 1, s + 1));

  const windowStart = step ? Math.max(0, step.i - k + 1) : 0;
  const windowEnd = step ? step.i : -1;

  const actionColor = useCallback((action: Step["action"]) => {
    if (action === "push") return "text-green-400";
    if (action === "pop_back") return "text-red-400";
    if (action === "pop_front") return "text-orange-400";
    return "text-blue-400";
  }, []);

  const actionLabel = useCallback((action: Step["action"]) => {
    if (action === "push") return "PUSH → 队尾";
    if (action === "pop_back") return "POP ← 队尾";
    if (action === "pop_front") return "POP ← 队首";
    return "记录最大值";
  }, []);

  const maxVal = Math.max(...nums, 1);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-text-primary">🪟 滑动窗口最大值（单调队列 O(n)）</h3>
          <p className="text-xs text-text-tertiary mt-0.5">窗口大小 k = {k}，数组长度 n = {nums.length}</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {Object.keys(PRESETS).map((p) => (
            <button key={p} onClick={() => { setPreset(p); reset(); }}
              className={`px-2 py-1 rounded text-xs border transition-colors ${
                preset === p
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
              }`}>
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* 数组可视化 + 窗口高亮 */}
      <div>
        <div className="text-xs text-text-tertiary mb-2">数组（窗口 [{windowStart}..{windowEnd}] 高亮）</div>
        <div className="flex items-end gap-1 h-24 px-1">
          {nums.map((v, i) => {
            const inWindow = step && i >= windowStart && i <= windowEnd;
            const isCurrent = step && i === step.i;
            const inDeque = step?.deqSnap.includes(i);
            const hPct = Math.max(8, (v / maxVal) * 90);
            return (
              <div key={i} className="flex flex-col items-center flex-1">
                <div
                  className={`w-full rounded-t transition-all duration-300 ${
                    isCurrent ? "bg-green-500"
                    : inDeque ? "bg-amber-400"
                    : inWindow ? "bg-blue-500/60"
                    : "bg-bg-tertiary border border-border-subtle"
                  }`}
                  style={{ height: `${hPct}%` }}
                />
                <span className={`text-[10px] mt-0.5 ${isCurrent ? "text-green-400 font-bold" : inDeque ? "text-amber-400 font-bold" : "text-text-secondary"}`}>{v}</span>
                <span className="text-[9px] text-text-tertiary">[{i}]</span>
              </div>
            );
          })}
        </div>
        <div className="flex gap-4 text-xs text-text-secondary mt-1">
          <span><span className="inline-block w-3 h-3 rounded bg-green-500 mr-1" />当前</span>
          <span><span className="inline-block w-3 h-3 rounded bg-amber-400 mr-1" />队列中</span>
          <span><span className="inline-block w-3 h-3 rounded bg-blue-500/60 mr-1" />窗口内</span>
        </div>
      </div>

      {/* 步骤描述 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle min-h-[60px]">
        <div className="flex justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {stepIdx + 1} / {steps.length}</span>
          {step && <span className={`font-bold ${actionColor(step.action)}`}>{actionLabel(step.action)}</span>}
        </div>
        <p className="text-text-primary text-sm">{step?.description}</p>
      </div>

      {/* 双列：Deque 状态 + 结果数组 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Deque 可视化 */}
        <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
          <div className="text-xs text-text-tertiary mb-2">单调队列（存下标，队首 → 队尾）</div>
          {step?.deqSnap.length === 0 ? (
            <div className="text-text-tertiary text-xs italic py-2">（空队列）</div>
          ) : (
            <div className="flex gap-1 items-center">
              <span className="text-xs text-orange-400 font-bold">队首↓</span>
              <div className="flex gap-1 flex-wrap flex-1">
                {step?.deqSnap.map((idx, pos) => (
                  <div key={pos}
                    className={`flex flex-col items-center px-2 py-1 rounded border text-xs ${
                      pos === 0 ? "border-orange-400 bg-orange-400/10 text-orange-300"
                      : pos === (step.deqSnap.length - 1) ? "border-purple-400 bg-purple-400/10 text-purple-300"
                      : "border-border-subtle bg-bg-secondary text-text-secondary"
                    }`}>
                    <span className="font-bold">{nums[idx]}</span>
                    <span className="text-[9px] opacity-70">[{idx}]</span>
                  </div>
                ))}
              </div>
              <span className="text-xs text-purple-400 font-bold">↑队尾</span>
            </div>
          )}
          <div className="mt-2 text-[10px] text-text-tertiary">
            队首值（窗口最大）：{step && step.deqSnap.length > 0 ? (
              <span className="text-orange-300 font-bold">{nums[step.deqSnap[0]]}</span>
            ) : "—"}
          </div>
        </div>

        {/* 结果数组 */}
        <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
          <div className="text-xs text-text-tertiary mb-2">输出结果（每窗口最大值）</div>
          <div className="flex gap-1 flex-wrap">
            {step?.resultSnap.map((v, i) => (
              v !== null ? (
                <div key={i} className="flex flex-col items-center px-2 py-1 rounded border border-blue-500 bg-blue-600/15 text-xs">
                  <span className="text-blue-300 font-bold">{v}</span>
                  <span className="text-[9px] text-text-tertiary">i={i}</span>
                </div>
              ) : null
            ))}
            {step?.resultSnap.every((v) => v === null) && (
              <span className="text-text-tertiary text-xs italic">（窗口尚未满足大小 k）</span>
            )}
          </div>
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors">
          ↩ 重置
        </button>
        <button onClick={prev} disabled={stepIdx === 0}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors disabled:opacity-40">
          ← 上一步
        </button>
        <button onClick={next} disabled={stepIdx === steps.length - 1}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs hover:bg-blue-700 transition-colors disabled:opacity-40">
          下一步 →
        </button>
      </div>

      {/* 最终结果 */}
      {stepIdx === steps.length - 1 && (
        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-3">
          <div className="text-green-400 text-xs font-semibold mb-2">✅ 完成！滑动窗口最大值结果：</div>
          <div className="flex gap-2 flex-wrap">
            {step?.resultSnap.map((v, i) =>
              v !== null ? (
                <div key={i} className="text-xs bg-bg-tertiary rounded px-2 py-1 border border-green-500/40">
                  <span className="text-text-secondary">窗口[{Math.max(0,i-k+1)}..{i}]：</span>
                  <span className="text-green-400 font-bold">{v}</span>
                </div>
              ) : null
            )}
          </div>
          <div className="text-xs text-text-tertiary mt-2">
            输出数组：[{step?.resultSnap.filter((v) => v !== null).join(", ")}]
          </div>
        </div>
      )}
    </div>
  );
}

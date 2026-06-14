"use client";

import React, { useState, useCallback } from "react";

/** 单调栈追踪器：支持「下一个更大元素」(NGE) 和「下一个更小元素」(NSE) */

type Mode = "NGE" | "NSE";

interface Step {
  index: number;          // 当前遍历的数组下标
  action: "push" | "pop" | "scan"; // 当前动作
  poppedIdx?: number;     // 被弹出的元素下标（pop 步骤用）
  poppedAnswer?: number;  // 被弹出元素的答案
  stackSnapshot: number[]; // 操作后栈的快照（存下标）
  resultSnapshot: number[]; // 操作后 result 数组快照
  description: string;
}

// 预设数组
const PRESETS: Record<string, number[]> = {
  "示例 1": [2, 1, 5, 6, 2, 3],
  "示例 2": [3, 1, 4, 1, 5, 9],
  "全递增": [1, 2, 3, 4, 5],
  "全递减": [5, 4, 3, 2, 1],
};

function buildSteps(nums: number[], mode: Mode): Step[] {
  const n = nums.length;
  const result = Array(n).fill(-1);
  const stack: number[] = [];
  const steps: Step[] = [];

  const cmp = (top: number, cur: number) =>
    mode === "NGE" ? nums[top] < cur : nums[top] > cur;

  for (let i = 0; i < n; i++) {
    // scan step
    steps.push({
      index: i,
      action: "scan",
      stackSnapshot: [...stack],
      resultSnapshot: [...result],
      description: `扫描 nums[${i}] = ${nums[i]}`,
    });

    // pop steps
    while (stack.length > 0 && cmp(stack[stack.length - 1], nums[i])) {
      const poppedIdx = stack.pop()!;
      result[poppedIdx] = nums[i];
      steps.push({
        index: i,
        action: "pop",
        poppedIdx,
        poppedAnswer: nums[i],
        stackSnapshot: [...stack],
        resultSnapshot: [...result],
        description:
          mode === "NGE"
            ? `弹出 nums[${poppedIdx}]=${nums[poppedIdx]}，其下一个更大元素是 ${nums[i]}`
            : `弹出 nums[${poppedIdx}]=${nums[poppedIdx]}，其下一个更小元素是 ${nums[i]}`,
      });
    }

    // push step
    stack.push(i);
    steps.push({
      index: i,
      action: "push",
      stackSnapshot: [...stack],
      resultSnapshot: [...result],
      description: `将 nums[${i}]=${nums[i]} 压入单调栈`,
    });
  }

  return steps;
}

export default function MonotonicStackTrace() {
  const [preset, setPreset] = useState("示例 1");
  const [customInput, setCustomInput] = useState("");
  const [mode, setMode] = useState<Mode>("NGE");
  const [stepIdx, setStepIdx] = useState(0);

  const nums = customInput.trim()
    ? customInput.split(/[\s,，]+/).map(Number).filter((n) => !isNaN(n)).slice(0, 10)
    : PRESETS[preset];

  const steps = buildSteps(nums, mode);
  const currentStep = steps[stepIdx];
  const finalResult = buildSteps(nums, mode).at(-1)?.resultSnapshot ?? [];

  const reset = () => setStepIdx(0);
  const prev = () => setStepIdx((s) => Math.max(0, s - 1));
  const next = () => setStepIdx((s) => Math.min(steps.length - 1, s + 1));

  const barColor = useCallback(
    (idx: number): string => {
      if (!currentStep) return "bg-blue-500";
      if (currentStep.action === "pop" && currentStep.poppedIdx === idx)
        return "bg-red-500";
      if (currentStep.index === idx && currentStep.action !== "pop")
        return "bg-green-500";
      if (currentStep.stackSnapshot.includes(idx))
        return "bg-amber-400";
      return "bg-bg-tertiary border border-border-subtle";
    },
    [currentStep]
  );

  const maxHeight = Math.max(...nums, 1);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-5 font-mono text-sm">
      {/* 标题 */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h3 className="text-base font-bold text-text-primary">
          🗂️ 单调栈追踪器
        </h3>
        {/* 模式切换 */}
        <div className="flex gap-2">
          {(["NGE", "NSE"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); reset(); }}
              className={`px-3 py-1 rounded-lg text-xs font-semibold border transition-colors ${
                mode === m
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
              }`}
            >
              {m === "NGE" ? "下一个更大 (NGE)" : "下一个更小 (NSE)"}
            </button>
          ))}
        </div>
      </div>

      {/* 数据选择区 */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-text-tertiary text-xs">预设：</span>
        {Object.keys(PRESETS).map((k) => (
          <button
            key={k}
            onClick={() => { setPreset(k); setCustomInput(""); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${
              preset === k && !customInput
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
            }`}
          >
            {k}
          </button>
        ))}
        <input
          type="text"
          placeholder="自定义（逗号/空格分隔）"
          value={customInput}
          onChange={(e) => { setCustomInput(e.target.value); reset(); }}
          className="ml-2 px-2 py-1 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-xs focus:outline-none focus:border-blue-400 w-44"
        />
      </div>

      {/* 数组 + 柱状图 */}
      <div>
        <div className="text-xs text-text-tertiary mb-2">
          输入：[{nums.join(", ")}]
        </div>
        <div className="flex items-end gap-2 h-28 px-2">
          {nums.map((v, i) => {
            const heightPct = Math.max(10, (v / maxHeight) * 100);
            const color = barColor(i);
            const isInStack = currentStep?.stackSnapshot.includes(i);
            return (
              <div key={i} className="flex flex-col items-center gap-1 flex-1">
                {/* 柱子 */}
                <div
                  className={`w-full rounded-t transition-all duration-300 ${color}`}
                  style={{ height: `${heightPct}%` }}
                />
                {/* 值标签 */}
                <span className={`text-xs ${isInStack ? "text-amber-500 font-bold" : "text-text-secondary"}`}>
                  {v}
                </span>
                {/* 下标 */}
                <span className="text-[10px] text-text-tertiary">[{i}]</span>
              </div>
            );
          })}
        </div>
        {/* 图例 */}
        <div className="flex gap-4 mt-2 text-xs text-text-secondary">
          <span><span className="inline-block w-3 h-3 rounded bg-green-500 mr-1" />当前扫描</span>
          <span><span className="inline-block w-3 h-3 rounded bg-red-500 mr-1" />弹出</span>
          <span><span className="inline-block w-3 h-3 rounded bg-amber-400 mr-1" />栈内</span>
        </div>
      </div>

      {/* 步骤信息 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle min-h-[60px]">
        <div className="flex justify-between text-xs text-text-tertiary mb-1">
          <span>步骤 {stepIdx + 1} / {steps.length}</span>
          <span className={`font-semibold ${
            currentStep?.action === "push" ? "text-green-500"
            : currentStep?.action === "pop" ? "text-red-500"
            : "text-blue-400"
          }`}>
            {currentStep?.action.toUpperCase()}
          </span>
        </div>
        <p className="text-text-primary text-sm">{currentStep?.description}</p>
      </div>

      {/* 栈状态 */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
          <div className="text-xs text-text-tertiary mb-2">栈（下标 → 值）</div>
          <div className="flex flex-col-reverse gap-1">
            {currentStep?.stackSnapshot.length === 0 ? (
              <span className="text-text-tertiary text-xs italic">（空）</span>
            ) : (
              currentStep?.stackSnapshot.map((idx, pos) => (
                <div
                  key={pos}
                  className={`flex justify-between px-2 py-1 rounded text-xs ${
                    pos === (currentStep.stackSnapshot.length - 1)
                      ? "bg-amber-400/20 border border-amber-400 text-amber-400"
                      : "bg-bg-secondary text-text-secondary"
                  }`}
                >
                  <span>[{idx}]</span>
                  <span className="font-semibold">{nums[idx]}</span>
                </div>
              ))
            )}
          </div>
          {currentStep?.stackSnapshot && currentStep.stackSnapshot.length > 0 && (
            <div className="text-[10px] text-text-tertiary mt-1 text-center">↑ 栈顶</div>
          )}
        </div>

        <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
          <div className="text-xs text-text-tertiary mb-2">
            结果数组（-1 = 未知）
          </div>
          <div className="flex flex-col gap-1">
            {currentStep?.resultSnapshot.map((val, i) => (
              <div key={i} className="flex justify-between px-2 py-1 rounded text-xs bg-bg-secondary">
                <span className="text-text-secondary">nums[{i}]={nums[i]}</span>
                <span
                  className={`font-semibold ${
                    val === -1 ? "text-text-tertiary" : "text-blue-400"
                  }`}
                >
                  → {val === -1 ? "?" : val}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex gap-2 justify-center">
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors"
        >
          ↩ 重置
        </button>
        <button
          onClick={prev}
          disabled={stepIdx === 0}
          className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors disabled:opacity-40"
        >
          ← 上一步
        </button>
        <button
          onClick={next}
          disabled={stepIdx === steps.length - 1}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs hover:bg-blue-700 transition-colors disabled:opacity-40"
        >
          下一步 →
        </button>
      </div>

      {/* 最终结果（算法完成后显示） */}
      {stepIdx === steps.length - 1 && (
        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-3">
          <div className="text-green-400 text-xs font-semibold mb-2">
            ✅ 算法完成！{mode === "NGE" ? "下一个更大元素" : "下一个更小元素"} 结果：
          </div>
          <div className="flex gap-2 flex-wrap">
            {nums.map((v, i) => (
              <div key={i} className="text-xs bg-bg-tertiary rounded px-2 py-1 border border-border-subtle">
                <span className="text-text-secondary">{v} →</span>{" "}
                <span className={`font-bold ${finalResult[i] === -1 ? "text-text-tertiary" : "text-green-400"}`}>
                  {finalResult[i] === -1 ? "无" : finalResult[i]}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

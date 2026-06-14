"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 数据与算法 ────────────────────────────────────────────────────────────────
type Problem = "twosum" | "container" | "threesum" | "movezeroes";

interface Step {
  l: number;
  r: number;
  label: string;
  action: "init" | "move-l" | "move-r" | "found" | "skip";
  pairSum?: number;
  result?: number | number[];
  highlight?: number[]; // extra highlights
}

function genTwoSumSteps(nums: number[], target: number): Step[] {
  const steps: Step[] = [];
  let l = 0, r = nums.length - 1;
  steps.push({ l, r, label: `初始化：l=0，r=${r}，目标和=${target}`, action: "init" });
  while (l < r) {
    const s = nums[l] + nums[r];
    if (s === target) {
      steps.push({ l, r, label: `✅ nums[${l}]+nums[${r}]=${s}=目标！找到配对`, action: "found", pairSum: s, result: [l, r] });
      break;
    } else if (s < target) {
      steps.push({ l, r, label: `nums[${l}]+nums[${r}]=${s}<${target}，需更大 → l右移`, action: "move-l", pairSum: s });
      l++;
    } else {
      steps.push({ l, r, label: `nums[${l}]+nums[${r}]=${s}>${target}，需更小 → r左移`, action: "move-r", pairSum: s });
      r--;
    }
  }
  return steps;
}

function genContainerSteps(height: number[]): Step[] {
  const steps: Step[] = [];
  let l = 0, r = height.length - 1, best = 0;
  steps.push({ l, r, label: "初始化：l=0，r=最右端", action: "init" });
  while (l < r) {
    const water = Math.min(height[l], height[r]) * (r - l);
    if (water > best) best = water;
    if (height[l] < height[r]) {
      steps.push({ l, r, label: `容量=${water}（min(${height[l]},${height[r]})×${r-l}），最大更新为${best}，height[${l}]更矮→l右移`, action: "move-l", result: best });
      l++;
    } else {
      steps.push({ l, r, label: `容量=${water}（min(${height[l]},${height[r]})×${r-l}），最大更新为${best}，height[${r}]更矮→r左移`, action: "move-r", result: best });
      r--;
    }
  }
  steps.push({ l, r: l, label: `指针相遇，结束。最大容水量=${best}`, action: "found", result: best });
  return steps;
}

function genMoveZeroesSteps(nums: number[]): Step[] {
  const steps: Step[] = [];
  const arr = nums.slice();
  let slow = 0;
  steps.push({ l: slow, r: 0, label: "初始化：slow=0（慢指针），fast=0", action: "init" });
  for (let fast = 0; fast < arr.length; fast++) {
    if (arr[fast] !== 0) {
      if (slow !== fast) {
        [arr[slow], arr[fast]] = [arr[fast], arr[slow]];
        steps.push({ l: slow, r: fast, label: `nums[${fast}]=${arr[slow]}≠0，交换arr[${slow}]↔arr[${fast}]，slow右移`, action: "move-l", highlight: [slow] });
      } else {
        steps.push({ l: slow, r: fast, label: `nums[${fast}]=${arr[fast]}≠0（已在正确位置），slow右移`, action: "skip" });
      }
      slow++;
    } else {
      steps.push({ l: slow, r: fast, label: `nums[${fast}]=0，跳过，fast继续右移`, action: "move-r" });
    }
  }
  steps.push({ l: slow, r: arr.length - 1, label: `完成！数组变为 [${arr.join(",")}]，zero 全移到末尾`, action: "found" });
  return steps;
}

const PROBLEM_DEFS: Record<Problem, { label: string; desc: string; defaultArr: number[]; target?: number }> = {
  twosum:    { label: "两数之和（对撞）", desc: "有序数组找 target=13 的配对", defaultArr: [1, 3, 4, 6, 7, 9, 12, 13, 15], target: 13 },
  container: { label: "盛最多水（对撞）", desc: "找面积最大的容器（两板之积）", defaultArr: [1, 8, 6, 2, 5, 4, 8, 3, 7] },
  threesum:  { label: "移动零（快慢）", desc: "快慢指针将非零元素前移", defaultArr: [0, 1, 0, 3, 12] },
  movezeroes:{ label: "移动零（快慢）", desc: "快慢指针将非零元素前移", defaultArr: [0, 1, 0, 3, 12] },
};

const ACTION_COLORS: Record<Step["action"], string> = {
  init:    "text-text-secondary",
  "move-l":"text-blue-600 dark:text-blue-300",
  "move-r":"text-violet-600 dark:text-violet-300",
  found:   "text-emerald-600 dark:text-emerald-300",
  skip:    "text-amber-600 dark:text-amber-400",
};

export default function TwoPointerRace() {
  const [problem, setProblem] = useState<Problem>("twosum");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const def = PROBLEM_DEFS[problem === "threesum" ? "movezeroes" : problem];
  const nums = def.defaultArr;

  const steps = React.useMemo(() => {
    if (problem === "twosum")
      return genTwoSumSteps(nums, def.target!);
    if (problem === "container")
      return genContainerSteps(nums);
    return genMoveZeroesSteps(nums);
  }, [problem]);

  const cur = steps[Math.min(step, steps.length - 1)];
  const { l, r } = cur;

  const startPlay = useCallback(() => {
    if (step >= steps.length - 1) setStep(0);
    setPlaying(true);
  }, [step, steps.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => {
          if (s >= steps.length - 1) { setPlaying(false); return s; }
          return s + 1;
        });
      }, 900);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, steps.length]);

  const reset = () => { setStep(0); setPlaying(false); };

  const maxH = problem === "container" ? Math.max(...nums) : 0;

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-rose-500/15 dark:bg-rose-500/20 flex items-center justify-center text-xl">
          ↔️
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">双指针追及可视化</h3>
          <p className="text-xs text-text-secondary">对撞指针 & 快慢指针的步进动画，观察指针移动决策</p>
        </div>
      </div>

      {/* 问题选择 */}
      <div className="flex flex-wrap gap-2 border-t border-border-subtle pt-3">
        {(["twosum", "container", "movezeroes"] as Problem[]).map((p) => (
          <button
            key={p}
            onClick={() => { setProblem(p); reset(); }}
            className={`px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${problem === p
              ? "bg-rose-500/20 border-rose-400/60 text-rose-700 dark:text-rose-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}
          >
            {PROBLEM_DEFS[p === "threesum" ? "movezeroes" : p].label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center transition-colors">
            ‹
          </button>
          <button onClick={playing ? () => setPlaying(false) : startPlay}
            className="px-3 py-1 rounded-lg bg-rose-500/15 dark:bg-rose-500/20 hover:bg-rose-500/25 text-rose-700 dark:text-rose-300 text-xs font-medium transition-colors">
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={step >= steps.length - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center transition-colors">
            ›
          </button>
          <button onClick={reset}
            className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors">
            重置
          </button>
        </div>
      </div>

      {/* 进度 */}
      <div>
        <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden mb-3">
          <div className="h-full bg-rose-500 rounded-full transition-all duration-300"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
        </div>

        {/* 当前操作标签 */}
        <div className={`rounded-lg border border-border-subtle bg-bg-tertiary px-3 py-2 text-xs font-medium ${ACTION_COLORS[cur.action]}`}>
          步骤 {step + 1}/{steps.length}：{cur.label}
          {cur.result !== undefined && (
            <span className="ml-2 text-text-tertiary">
              {typeof cur.result === "number" ? `当前最优: ${cur.result}` : `结果索引: [${(cur.result as number[]).join(",")}]`}
            </span>
          )}
        </div>
      </div>

      {/* 数组可视化 */}
      {problem === "container" ? (
        /* 容器问题：展示高度柱 */
        <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
          <div className="text-xs text-text-secondary mb-3">height 数组（容器高度）</div>
          <div className="flex items-end gap-1" style={{ height: 100 }}>
            {nums.map((h, i) => {
              const isL = i === l, isR = i === r;
              const inRange = i >= l && i <= r;
              const barH = Math.max((h / maxH) * 90, 4);
              return (
                <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5">
                  {isL && <span className="text-[10px] text-blue-600 dark:text-blue-300 font-bold">L</span>}
                  {isR && <span className="text-[10px] text-violet-600 dark:text-violet-300 font-bold">R</span>}
                  {!isL && !isR && <span className="text-[10px] invisible">·</span>}
                  <div className={`w-full rounded-t transition-all duration-300 ${isL ? "bg-blue-500" : isR ? "bg-violet-500" : inRange ? "bg-rose-400/50" : "bg-bg-secondary border border-border-subtle"}`}
                    style={{ height: `${barH}px` }} />
                  <span className="text-[8px] font-mono text-text-tertiary">{h}</span>
                </div>
              );
            })}
          </div>
          {cur.result !== undefined && (
            <div className="mt-2 text-center text-sm font-bold text-emerald-600 dark:text-emerald-300">
              当前最大容量：{cur.result}
            </div>
          )}
        </div>
      ) : (
        /* 通用数组视图 */
        <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
          <div className="text-xs text-text-secondary mb-3">
            {problem === "twosum" ? `有序数组（target = ${def.target}）` : "数组（移动零）"}
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {nums.map((v, i) => {
              const isL = i === l;
              const isR = i === r && problem === "twosum";
              const isSlow = i === l && problem === "movezeroes";
              const isFast = i === r && problem === "movezeroes";
              const isHl = cur.highlight?.includes(i);
              return (
                <div key={i} className="flex flex-col items-center gap-0.5">
                  <span className={`text-[10px] font-bold ${isL && problem === "twosum" ? "text-blue-600 dark:text-blue-300" :
                    isR && problem === "twosum" ? "text-violet-600 dark:text-violet-300" :
                    isSlow ? "text-blue-600 dark:text-blue-300" : isFast ? "text-amber-600 dark:text-amber-400" : "invisible"}`}>
                    {isL && problem === "twosum" ? "L" : isR && problem === "twosum" ? "R" : isSlow ? "慢" : isFast ? "快" : "·"}
                  </span>
                  <div className={`w-9 h-10 rounded-lg border flex items-center justify-center text-sm font-bold font-mono transition-all duration-300
                    ${isL && problem === "twosum" ? "bg-blue-500 border-blue-400 text-white" :
                      isR && problem === "twosum" ? "bg-violet-500 border-violet-400 text-white" :
                      isSlow ? "bg-blue-500 border-blue-400 text-white" :
                      isFast ? "bg-amber-500 border-amber-400 text-white" :
                      isHl ? "bg-emerald-500/20 border-emerald-400/60 text-emerald-700 dark:text-emerald-300" :
                      v === 0 ? "bg-bg-secondary border-border-subtle text-text-tertiary" :
                      "bg-bg-secondary border-border-subtle text-text-primary"}`}>
                    {v}
                  </div>
                  <span className="text-[8px] font-mono text-text-tertiary">[{i}]</span>
                </div>
              );
            })}
          </div>

          {problem === "twosum" && cur.pairSum !== undefined && (
            <div className="mt-3 text-xs text-text-secondary font-mono">
              nums[{l}] + nums[r] = {nums[l]} + {nums[r]} = {cur.pairSum}
              {" "}
              {cur.pairSum < def.target! ? "< target → l右移" :
               cur.pairSum > def.target! ? "> target → r左移" : "= target ✅"}
            </div>
          )}
        </div>
      )}

      {/* 说明框 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-semibold text-text-primary">
          {problem === "twosum" ? "对撞指针（Two Sum）" :
           problem === "container" ? "对撞指针（盛水容器）" :
           "快慢指针（移动零）"}
        </div>
        <div>
          {problem === "twosum" ? "左右指针各最多移动 n 步，总共 O(n) 步，利用数组有序性保证不错过答案。" :
           problem === "container" ? "每次移动较矮的板：若移动较高的，宽度减少但高度不增，容量只减不增；移较矮则有可能增大。" :
           "慢指针指向下一个放置位，快指针扫描非零元素并放到慢指针位置，每个元素最多被访问两次 → O(n)。"}
        </div>
        <div className="text-text-tertiary">时间复杂度 O(n)，空间复杂度 O(1)（原地操作）。</div>
      </div>
    </div>
  );
}

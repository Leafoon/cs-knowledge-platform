"use client";
import React, { useState } from "react";

/* ─── Problem definitions ────────────────────────────────────────────────── */

interface BSearchStep {
  lo: number;
  hi: number;
  mid: number;
  midVal: number;       // check(mid) result value (hours / days)
  feasible: boolean;    // check(mid) <= threshold?
  action: "go_left" | "go_right" | "done";
  decision: string;
  newLo: number;
  newHi: number;
}

interface Problem {
  id: string;
  title: string;
  leetcode: string;
  description: string;
  searchLabel: string;   // x-axis label
  checkLabel: string;    // y-axis label (what check() measures)
  threshold: number;     // check() <= threshold means feasible
  params: string;        // param display
  lo: number;
  hi: number;
  answer: number;
  // precomputed: for each x in [lo..hi], the check value
  checkValues: { x: number; val: number }[];
  // precomputed binary search steps
  steps: BSearchStep[];
  // detail render for each mid
  computeDetail: (mid: number) => { parts: string[]; total: number };
}

/* ─── Koko eating bananas (#875) ─────────────────────────────────────────── */

const KOKO_PILES = [3, 6, 7, 11];
const KOKO_H = 8;

function kokoHours(k: number): number {
  return KOKO_PILES.reduce((s, p) => s + Math.ceil(p / k), 0);
}

const kokoCheckValues = Array.from({ length: 11 }, (_, i) => ({ x: i + 1, val: kokoHours(i + 1) }));

const kokoSteps: BSearchStep[] = [
  { lo: 1, hi: 11, mid: 6, midVal: 6,  feasible: true,  action: "go_left",  decision: "6 ≤ 8 → 可行，右边界收缩 hi=6", newLo: 1, newHi: 6 },
  { lo: 1, hi: 6,  mid: 3, midVal: 10, feasible: false, action: "go_right", decision: "10 > 8 → 不可行，左边界右移 lo=4", newLo: 4, newHi: 6 },
  { lo: 4, hi: 6,  mid: 5, midVal: 8,  feasible: true,  action: "go_left",  decision: "8 ≤ 8 → 可行，右边界收缩 hi=5", newLo: 4, newHi: 5 },
  { lo: 4, hi: 5,  mid: 4, midVal: 8,  feasible: true,  action: "go_left",  decision: "8 ≤ 8 → 可行，右边界收缩 hi=4", newLo: 4, newHi: 4 },
  { lo: 4, hi: 4,  mid: 4, midVal: 8,  feasible: true,  action: "done",     decision: "lo == hi == 4 → 最小可行速度 k = 4 ✓", newLo: 4, newHi: 4 },
];

const kokoComputeDetail = (mid: number) => ({
  parts: KOKO_PILES.map(p => `⌈${p}/${mid}⌉=${Math.ceil(p / mid)}`),
  total: kokoHours(mid),
});

/* ─── Ship packages (#1011) ──────────────────────────────────────────────── */

const SHIP_WEIGHTS = [3, 2, 2, 4, 1, 4];
const SHIP_DAYS = 3;

function shipDays(cap: number): number {
  let days = 1, cur = 0;
  for (const w of SHIP_WEIGHTS) {
    if (cur + w > cap) { days++; cur = 0; }
    cur += w;
  }
  return days;
}

// search space lo=max(weights)=4, hi=sum=16, but show 4..14 for visual clarity
const shipCheckValues = Array.from({ length: 13 }, (_, i) => ({ x: i + 4, val: shipDays(i + 4) }));

const shipSteps: BSearchStep[] = [
  { lo: 4, hi: 16, mid: 10, midVal: 2, feasible: true,  action: "go_left",  decision: "2 ≤ 3 → 可行，hi=10", newLo: 4, newHi: 10 },
  { lo: 4, hi: 10, mid: 7,  midVal: 3, feasible: true,  action: "go_left",  decision: "3 ≤ 3 → 可行，hi=7",  newLo: 4, newHi: 7 },
  { lo: 4, hi: 7,  mid: 5,  midVal: 4, feasible: false, action: "go_right", decision: "4 > 3 → 不可行，lo=6", newLo: 6, newHi: 7 },
  { lo: 6, hi: 7,  mid: 6,  midVal: 3, feasible: true,  action: "go_left",  decision: "3 ≤ 3 → 可行，hi=6",  newLo: 6, newHi: 6 },
  { lo: 6, hi: 6,  mid: 6,  midVal: 3, feasible: true,  action: "done",     decision: "lo == hi == 6 → 最小载重 = 6 ✓", newLo: 6, newHi: 6 },
];

const shipComputeDetail = (mid: number) => {
  const parts: string[] = [];
  let days = 1, cur = 0;
  let segStart = 0;
  const segments: number[][] = [[]];
  for (const w of SHIP_WEIGHTS) {
    if (cur + w > mid) { days++; cur = 0; segments.push([]); }
    cur += w;
    segments[segments.length - 1].push(w);
  }
  segments.forEach((seg, i) => parts.push(`第${i + 1}天:[${seg.join("+")}]=${seg.reduce((a, b) => a + b, 0)}`));
  return { parts, total: days };
};

/* ─── Problem registry ───────────────────────────────────────────────────── */

const PROBLEMS: Problem[] = [
  {
    id: "koko",
    title: "🍌 Koko 吃香蕉",
    leetcode: "#875",
    description: "piles=[3,6,7,11]，h=8 小时内吃完，求最小速度 k（根/小时）",
    searchLabel: "速度 k（根/小时）",
    checkLabel: "总耗时（小时）",
    threshold: KOKO_H,
    params: `piles=[3,6,7,11]，h=${KOKO_H}`,
    lo: 1, hi: 11, answer: 4,
    checkValues: kokoCheckValues,
    steps: kokoSteps,
    computeDetail: kokoComputeDetail,
  },
  {
    id: "ship",
    title: "🚢 运货上船",
    leetcode: "#1011",
    description: "weights=[3,2,2,4,1,4]，3 天内运完，求最小载重",
    searchLabel: "载重 m",
    checkLabel: "所需天数",
    threshold: SHIP_DAYS,
    params: `weights=[3,2,2,4,1,4]，days=${SHIP_DAYS}`,
    lo: 4, hi: 16, answer: 6,
    checkValues: shipCheckValues,
    steps: shipSteps,
    computeDetail: shipComputeDetail,
  },
];

/* ─── Bar Chart ──────────────────────────────────────────────────────────── */

function BarChart({
  problem,
  step,
}: {
  problem: Problem;
  step: BSearchStep;
}) {
  const maxVal = Math.max(...problem.checkValues.map(c => c.val));
  const CHART_HEIGHT = 140; // px

  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-4">
      <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-3 flex items-center justify-between">
        <span>check({problem.searchLabel}) 的值 vs 阈值 {problem.threshold}</span>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-emerald-400 dark:bg-emerald-500" />
            <span>可行（≤{problem.threshold}）</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-rose-400 dark:bg-rose-500" />
            <span>不可行</span>
          </div>
        </div>
      </div>

      <div className="relative" style={{ height: CHART_HEIGHT + 40 }}>
        {/* Threshold line */}
        <div
          className="absolute left-0 right-0 border-t-2 border-dashed border-amber-500 dark:border-amber-400 z-10"
          style={{ bottom: (problem.threshold / maxVal) * CHART_HEIGHT + 24 }}>
          <span className="absolute -top-4 right-0 text-[10px] bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 px-1.5 py-0.5 rounded font-bold border border-amber-200 dark:border-amber-700">
            阈值 = {problem.threshold}
          </span>
        </div>

        {/* Bars */}
        <div className="absolute bottom-6 left-0 right-0 flex items-end gap-0.5">
          {problem.checkValues.map(({ x, val }) => {
            const isInRange = x >= step.lo && x <= step.hi;
            const isMid = x === step.mid;
            const isAnswer = step.action === "done" && x === step.mid;
            const feasible = val <= problem.threshold;
            const barH = Math.round((val / maxVal) * CHART_HEIGHT);

            let barColor = feasible
              ? (isInRange ? "bg-emerald-400 dark:bg-emerald-500" : "bg-emerald-200 dark:bg-emerald-800/50")
              : (isInRange ? "bg-rose-400 dark:bg-rose-500" : "bg-rose-200 dark:bg-rose-800/50");

            if (isMid) barColor = isAnswer
              ? "bg-emerald-500 dark:bg-emerald-400 ring-2 ring-emerald-300 dark:ring-emerald-500"
              : "bg-amber-400 dark:bg-amber-500 ring-2 ring-amber-300 dark:ring-amber-600";

            return (
              <div key={x} className="flex-1 flex flex-col items-center">
                {/* Value label on top when mid or answer */}
                <div className="h-5 flex items-end justify-center">
                  {(isMid || !isInRange) && (
                    <span className={`text-[9px] font-bold ${
                      isMid ? "text-amber-700 dark:text-amber-300" : "text-slate-400 dark:text-slate-600"
                    }`}>
                      {val}
                    </span>
                  )}
                </div>
                {/* Bar */}
                <div
                  className={`w-full rounded-t transition-all duration-400 ${barColor} ${
                    !isInRange ? "opacity-35" : ""
                  }`}
                  style={{ height: Math.max(barH, 3) }}
                />
                {/* mid indicator */}
                <div className="h-4 flex items-center justify-center">
                  {isMid && (
                    <span className="text-[9px] font-bold text-amber-600 dark:text-amber-400">↑</span>
                  )}
                </div>
                {/* x label */}
                <span className={`text-[9px] font-mono ${
                  isMid ? "font-bold text-amber-700 dark:text-amber-300"
                  : isInRange ? "text-slate-600 dark:text-slate-300"
                  : "text-slate-300 dark:text-slate-600"
                }`}>
                  {x}
                </span>
              </div>
            );
          })}
        </div>

        {/* Range bracket overlay label */}
        <div className="absolute bottom-0 left-0 right-0 text-center">
          <span className="text-[10px] text-indigo-500 dark:text-indigo-400 font-semibold">
            {step.action === "done"
              ? `答案：${problem.answer}`
              : `当前搜索范围 [${step.lo}, ${step.hi}]，mid = ${step.mid}`}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function AnswerBinarySearchDemo() {
  const [probIdx, setProbIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);

  const prob = PROBLEMS[probIdx];
  const step = prob.steps[Math.min(stepIdx, prob.steps.length - 1)];
  const totalSteps = prob.steps.length;
  const isDone = step.action === "done";

  const handleProblem = (i: number) => {
    setProbIdx(i);
    setStepIdx(0);
  };

  const detail = prob.computeDetail(step.mid);

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-fuchsia-500 via-purple-500 to-indigo-500 p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-bold text-lg tracking-tight">对答案二分（Answer Binary Search）</h3>
            <p className="text-purple-100 text-sm mt-0.5">
              将「最优化问题」转化为「可行性判断」，再对答案空间 lower_bound
            </p>
          </div>
          <div className="text-white/70 text-right text-xs leading-relaxed">
            <div className="font-bold text-white text-base">步骤 {stepIdx + 1} / {totalSteps}</div>
            <div>搜索空间 [{prob.lo}, {prob.hi}]</div>
          </div>
        </div>

        {/* progress */}
        <div className="flex gap-1 mt-4">
          {prob.steps.map((_, i) => (
            <button key={i} onClick={() => setStepIdx(i)}
              className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                i <= stepIdx ? "bg-white" : "bg-white/25"
              }`}
            />
          ))}
        </div>
      </div>

      <div className="p-5 space-y-4">

        {/* ── Problem selector ── */}
        <div className="flex gap-2">
          {PROBLEMS.map((p, i) => (
            <button key={p.id} onClick={() => handleProblem(i)}
              className={`flex-1 rounded-xl px-3 py-2.5 text-sm font-semibold border transition-all ${
                i === probIdx
                  ? "bg-purple-50 dark:bg-purple-900/30 border-purple-300 dark:border-purple-600 text-purple-700 dark:text-purple-300"
                  : "border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}>
              {p.title}
              <span className="ml-1.5 text-[10px] font-normal opacity-70">LeetCode {p.leetcode}</span>
            </button>
          ))}
        </div>

        {/* Problem description */}
        <div className="rounded-xl bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 px-4 py-3 text-sm">
          <div className="font-semibold text-purple-800 dark:text-purple-200">{prob.description}</div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-purple-600 dark:text-purple-400">
            <span>搜索空间：<strong>[{prob.lo}, {prob.hi}]</strong></span>
            <span>check 条件：<strong>{prob.checkLabel} ≤ {prob.threshold}</strong></span>
            <span>答案：<strong>{prob.answer}</strong></span>
          </div>
        </div>

        {/* ── Bar Chart ── */}
        <BarChart problem={prob} step={step} />

        {/* ── Check computation ── */}
        <div className={`rounded-xl border-2 p-4 space-y-3 transition-all ${
          isDone
            ? "border-emerald-300 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-900/20"
            : step.feasible
            ? "border-sky-200 dark:border-sky-700 bg-sky-50 dark:bg-sky-900/20"
            : "border-rose-200 dark:border-rose-700 bg-rose-50 dark:bg-rose-900/20"
        }`}>

          <div className="flex items-center gap-3">
            <div className={`flex-none px-3 py-1.5 rounded-lg text-white text-xs font-bold ${
              isDone ? "bg-emerald-500 dark:bg-emerald-600"
              : step.feasible ? "bg-sky-500 dark:bg-sky-600"
              : "bg-rose-500 dark:bg-rose-600"
            }`}>
              {isDone ? "完成 ✓" : `第 ${stepIdx + 1} 步`}
            </div>
            <div className={`font-semibold text-sm ${
              isDone ? "text-emerald-700 dark:text-emerald-300"
              : step.feasible ? "text-sky-700 dark:text-sky-300"
              : "text-rose-700 dark:text-rose-300"
            }`}>
              {step.decision}
            </div>
          </div>

          {/* Binary search state */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "lo", val: step.lo, sub: `${prob.searchLabel}下界`, color: "text-indigo-600 dark:text-indigo-400" },
              { label: "mid", val: step.mid, sub: `check(${step.mid})=${step.midVal}`, color: "text-amber-600 dark:text-amber-400" },
              { label: "hi", val: step.hi, sub: `${prob.searchLabel}上界`, color: "text-indigo-600 dark:text-indigo-400" },
            ].map(({ label, val, sub, color }) => (
              <div key={label} className="rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3 text-center">
                <div className="text-[10px] text-slate-400 dark:text-slate-500 font-medium uppercase">{label}</div>
                <div className={`text-2xl font-bold font-mono ${color}`}>{val}</div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">{sub}</div>
              </div>
            ))}
          </div>

          {/* check() computation breakdown */}
          {!isDone && (
            <div className="rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">
                check({step.mid}) 计算过程：
              </div>
              <div className="flex flex-wrap gap-1.5">
                {detail.parts.map((part, i) => (
                  <span key={i}
                    className={`text-xs font-mono px-2.5 py-1 rounded-lg border ${
                      step.feasible
                        ? "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300"
                        : "bg-rose-50 dark:bg-rose-900/30 border-rose-200 dark:border-rose-700 text-rose-700 dark:text-rose-300"
                    }`}>
                    {part}
                  </span>
                ))}
                <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded-lg border ${
                  step.feasible
                    ? "bg-emerald-100 dark:bg-emerald-900/50 border-emerald-300 dark:border-emerald-600 text-emerald-800 dark:text-emerald-200"
                    : "bg-rose-100 dark:bg-rose-900/50 border-rose-300 dark:border-rose-600 text-rose-800 dark:text-rose-200"
                }`}>
                  = {detail.total} {step.feasible ? `≤ ${prob.threshold} ✓` : `> ${prob.threshold} ✗`}
                </span>
              </div>
            </div>
          )}

          {/* Done banner */}
          {isDone && (
            <div className="rounded-xl bg-emerald-100 dark:bg-emerald-900/40 border border-emerald-300 dark:border-emerald-600 p-3 text-center">
              <div className="text-emerald-700 dark:text-emerald-300 font-bold text-sm">
                🎯 最终答案：{prob.id === "koko" ? "最小速度" : "最小载重"} = <span className="font-mono text-lg">{prob.answer}</span>
              </div>
              <div className="text-emerald-600 dark:text-emerald-400 text-xs mt-1">
                lo = hi = {prob.answer}，lower_bound 收敛，搜索完成
              </div>
            </div>
          )}
        </div>

        {/* ── Strategic insight ── */}
        <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-4">
          <div className="text-xs font-semibold text-slate-600 dark:text-slate-300 mb-2">对答案二分的通用框架</div>
          <div className="grid grid-cols-3 gap-3 text-center">
            {[
              { step: "①", title: "确定搜索空间", desc: `[lo=${prob.lo}, hi=${prob.hi}]`, color: "text-purple-600 dark:text-purple-400" },
              { step: "②", title: "设计 check()", desc: `${prob.checkLabel} ≤ ${prob.threshold}？`, color: "text-indigo-600 dark:text-indigo-400" },
              { step: "③", title: "找左边界", desc: "最小可行的答案", color: "text-emerald-600 dark:text-emerald-400" },
            ].map(({ step: s, title, desc, color }) => (
              <div key={s} className="rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-3">
                <div className={`text-lg font-bold ${color}`}>{s}</div>
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-200 mt-1">{title}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Controls ── */}
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => setStepIdx(s => Math.max(0, s - 1))}
            disabled={stepIdx === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ← 上一步
          </button>

          <div className="flex gap-1.5">
            {prob.steps.map((_, i) => (
              <button key={i} onClick={() => setStepIdx(i)}
                className={`h-2 rounded-full transition-all ${
                  i === stepIdx ? "bg-purple-500 w-6" : "bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500 w-2"
                }`}
              />
            ))}
          </div>

          <button
            onClick={() => setStepIdx(s => Math.min(totalSteps - 1, s + 1))}
            disabled={stepIdx >= totalSteps - 1}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-purple-500 hover:bg-purple-600 text-white shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            下一步 →
          </button>
        </div>
      </div>
    </div>
  );
}

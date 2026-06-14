"use client";
import React, { useState } from "react";

/* ─── Data ───────────────────────────────────────────────────────────────── */

// 演示数组：15 个元素，分为 3 组，每组 5 个
const ORIGINAL = [29, 3, 17, 81, 45, 52, 8, 36, 64, 12, 71, 23, 58, 9, 40];
const GROUPS_ORIGINAL = [
  [29, 3, 17, 81, 45],
  [52, 8, 36, 64, 12],
  [71, 23, 58, 9, 40],
];
const GROUPS_SORTED = [
  [3, 17, 29, 45, 81],   // 中位数 = 29（第3小）
  [8, 12, 36, 52, 64],   // 中位数 = 36
  [9, 23, 40, 58, 71],   // 中位数 = 40
];
const MEDIANS = [29, 36, 40];  // 中位数数组 M
const MOM = 36;                // 中位数的中位数 m*（M 的中位数）
const MOM_IDX = 1;             // m* 在 MEDIANS 中的下标

// 划分结果：以 m* = 36 为 pivot
// 小于 36：3, 17, 29, 8, 12, 9, 23 (7个)
// 等于 36：36 (1个)
// 大于 36：45, 81, 52, 64, 40, 58, 71 (7个)
const LESS_THAN_MOM   = [3, 17, 29, 8, 12, 9, 23];
const EQUAL_MOM       = [36];
const GREATER_THAN_MOM= [45, 81, 52, 64, 40, 58, 71];

type Phase = 0 | 1 | 2 | 3 | 4;

interface PhaseConfig {
  title: string;
  subtitle: string;
  icon: string;
}

const PHASE_CONFIGS: PhaseConfig[] = [
  { title: "原始数组",        subtitle: "n = 15 个元素，准备分组",        icon: "①" },
  { title: "⌈n/5⌉ = 3 组",   subtitle: "每组 5 个元素各自排序",          icon: "②" },
  { title: "提取各组中位数",  subtitle: "每组第 3 小的元素即为中位数",    icon: "③" },
  { title: "递归求 m*",       subtitle: "对中位数数组 M 再次调用 BFPRT",  icon: "④" },
  { title: "划分保证",        subtitle: "m* 保证两侧各至少 30% 的元素",   icon: "⑤" },
];

/* ─── Sub-components ─────────────────────────────────────────────────────── */

function Cell({
  value,
  highlight,
  dimmed,
  label,
  size = "md",
}: {
  value: number;
  highlight?: "pivot" | "median" | "mom" | "less" | "greater" | "equal";
  dimmed?: boolean;
  label?: string;
  size?: "sm" | "md";
}) {
  const sizeClass = size === "sm" ? "w-9 h-9 text-sm" : "w-11 h-11 text-base";
  const styleMap = {
    pivot:   "bg-amber-100 dark:bg-amber-900/50 border-amber-400 dark:border-amber-500 text-amber-800 dark:text-amber-200 ring-2 ring-amber-300 dark:ring-amber-600",
    median:  "bg-sky-100 dark:bg-sky-900/50 border-sky-400 dark:border-sky-500 text-sky-800 dark:text-sky-200 ring-2 ring-sky-300 dark:ring-sky-600",
    mom:     "bg-emerald-100 dark:bg-emerald-900/50 border-emerald-500 dark:border-emerald-400 text-emerald-800 dark:text-emerald-200 ring-2 ring-emerald-400 dark:ring-emerald-500 scale-110 shadow-lg",
    less:    "bg-blue-100 dark:bg-blue-900/40 border-blue-400 dark:border-blue-500 text-blue-800 dark:text-blue-200",
    greater: "bg-rose-100 dark:bg-rose-900/40 border-rose-400 dark:border-rose-500 text-rose-800 dark:text-rose-200",
    equal:   "bg-emerald-100 dark:bg-emerald-900/40 border-emerald-500 dark:border-emerald-400 text-emerald-800 dark:text-emerald-200",
  };
  const base = "bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300";

  return (
    <div className="flex flex-col items-center gap-1">
      <div className={`
        ${sizeClass} rounded-xl border-2 flex items-center justify-center
        font-bold font-mono transition-all duration-300
        ${dimmed ? "opacity-30 scale-90" : ""}
        ${highlight ? styleMap[highlight] : base}
      `}>
        {value}
      </div>
      {label && <span className="text-[10px] text-slate-400 dark:text-slate-500 font-medium">{label}</span>}
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function BFPRTGrouping() {
  const [phase, setPhase] = useState<Phase>(0);

  const cfg = PHASE_CONFIGS[phase];

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-bold text-lg tracking-tight">BFPRT：五元组分组可视化</h3>
            <p className="text-emerald-100 text-sm mt-0.5">逐步看懂「中位数的中位数」如何选出好 Pivot</p>
          </div>
          <div className="text-white/80 text-right">
            <div className="text-2xl font-bold">{cfg.icon}</div>
            <div className="text-xs mt-0.5">共 {PHASE_CONFIGS.length} 步</div>
          </div>
        </div>

        {/* Phase stepper */}
        <div className="flex gap-1 mt-4">
          {PHASE_CONFIGS.map((_, i) => (
            <button
              key={i}
              onClick={() => setPhase(i as Phase)}
              className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                i <= phase ? "bg-white" : "bg-white/30"
              }`}
            />
          ))}
        </div>
      </div>

      <div className="p-5 space-y-5">

        {/* Step title */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-teal-500 text-white flex items-center justify-center font-bold text-sm flex-none">
            {phase + 1}
          </div>
          <div>
            <div className="font-bold text-slate-800 dark:text-slate-100">{cfg.title}</div>
            <div className="text-sm text-slate-500 dark:text-slate-400">{cfg.subtitle}</div>
          </div>
        </div>

        {/* ─── Phase 0: Original array ─── */}
        {phase === 0 && (
          <div className="space-y-3">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              原始数组 A（n = 15）
            </div>
            <div className="flex flex-wrap gap-2">
              {ORIGINAL.map((v, i) => (
                <Cell key={i} value={v} />
              ))}
            </div>
            <div className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 p-4">
              <p className="text-sm text-slate-600 dark:text-slate-300">
                任务：在 <strong>n = 15</strong> 个元素中，找第 <strong>k</strong> 小的元素。
                BFPRT 首先把这 15 个元素每 5 个分成一组，得到 <strong>⌈15/5⌉ = 3</strong> 组。
                接下来对每组排序，提取各组中位数。
              </p>
            </div>
          </div>
        )}

        {/* ─── Phase 1: Groups sorted ─── */}
        {phase === 1 && (
          <div className="space-y-4">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              三组各自排序（每组至多 5 个，插入排序 O(1) 次比较）
            </div>
            <div className="grid grid-cols-3 gap-3">
              {GROUPS_SORTED.map((grp, gi) => (
                <div key={gi} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-3">
                  <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-3 text-center">
                    第 {gi + 1} 组（已排序）
                  </div>
                  <div className="flex flex-col items-center gap-1.5">
                    {grp.map((v, vi) => (
                      <Cell
                        key={vi}
                        value={v}
                        highlight={vi === 2 ? "median" : undefined}
                        label={vi === 2 ? "中位数" : undefined}
                        size="sm"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-sky-700 dark:text-sky-300 bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-800 rounded-lg px-3 py-2">
              <span className="text-sky-500">◈</span>
              <span>每组第 3 小的元素（下标 2，0-indexed）即为该组中位数，蓝色高亮标注</span>
            </div>
          </div>
        )}

        {/* ─── Phase 2: Extract medians ─── */}
        {phase === 2 && (
          <div className="space-y-4">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              提取各组中位数 → 新数组 M
            </div>
            {/* 三组，只高亮中位数 */}
            <div className="grid grid-cols-3 gap-3">
              {GROUPS_SORTED.map((grp, gi) => (
                <div key={gi} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-3">
                  <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2 text-center">组 {gi + 1}</div>
                  <div className="flex justify-center gap-1">
                    {grp.map((v, vi) => (
                      <Cell key={vi} value={v}
                        highlight={vi === 2 ? "median" : undefined}
                        dimmed={vi !== 2}
                        size="sm"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Arrow */}
            <div className="flex justify-center">
              <div className="flex flex-col items-center gap-1 text-teal-500 dark:text-teal-400">
                <div className="text-xs font-semibold">↓ 提取中位数</div>
                <div className="w-0.5 h-6 bg-teal-400 dark:bg-teal-500" />
              </div>
            </div>

            {/* Medians array M */}
            <div className="rounded-xl border-2 border-sky-300 dark:border-sky-600 bg-sky-50 dark:bg-sky-900/20 p-4">
              <div className="text-xs font-semibold text-sky-600 dark:text-sky-300 mb-3 text-center uppercase tracking-wider">
                中位数数组 M（共 ⌈n/5⌉ = 3 个）
              </div>
              <div className="flex gap-3 justify-center">
                {MEDIANS.map((v, i) => (
                  <Cell key={i} value={v} highlight="median" label={`M[${i}]`} />
                ))}
              </div>
              <p className="text-xs text-sky-700 dark:text-sky-300 text-center mt-3">
                下一步：对 M 再次递归调用 BFPRT，找 M 的中位数
              </p>
            </div>
          </div>
        )}

        {/* ─── Phase 3: Median of medians ─── */}
        {phase === 3 && (
          <div className="space-y-4">
            <div className="rounded-xl border-2 border-sky-300 dark:border-sky-700 bg-sky-50 dark:bg-sky-900/20 p-4">
              <div className="text-xs font-semibold text-sky-600 dark:text-sky-300 mb-3 text-center">
                中位数数组 M（递归输入）
              </div>
              <div className="flex gap-3 justify-center">
                {MEDIANS.map((v, i) => (
                  <Cell key={i} value={v}
                    highlight={i === MOM_IDX ? "mom" : "median"}
                    label={i === MOM_IDX ? "m* = 中位数" : `M[${i}]`}
                  />
                ))}
              </div>
            </div>

            <div className="flex justify-center">
              <div className="px-4 py-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/40 border border-emerald-300 dark:border-emerald-600 text-emerald-700 dark:text-emerald-300 text-sm font-bold">
                m* = BFPRT(M, ⌊(|M|+1)/2⌋) = <span className="text-xl ml-1">{MOM}</span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-3">
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                为什么 m* = {MOM} 是一个「好的」pivot？
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 p-3">
                  <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">≤ m* 的元素</div>
                  <div className="text-blue-600 dark:text-blue-400 font-mono text-xs">
                    {LESS_THAN_MOM.join(", ")}{EQUAL_MOM.length > 0 ? ", " + EQUAL_MOM[0] : ""}
                  </div>
                  <div className="text-blue-500 dark:text-blue-400 mt-1 font-bold">
                    {LESS_THAN_MOM.length + 1} 个 ≥ 3n/10 = {(0.3 * 15).toFixed(0)} ✓
                  </div>
                </div>
                <div className="rounded-lg bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-700 p-3">
                  <div className="font-semibold text-rose-700 dark:text-rose-300 mb-1">&gt; m* 的元素</div>
                  <div className="text-rose-600 dark:text-rose-400 font-mono text-xs">
                    {GREATER_THAN_MOM.join(", ")}
                  </div>
                  <div className="text-rose-500 dark:text-rose-400 mt-1 font-bold">
                    {GREATER_THAN_MOM.length} 个 ≥ 3n/10 = {(0.3 * 15).toFixed(0)} ✓
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── Phase 4: Partition guarantee ─── */}
        {phase === 4 && (
          <div className="space-y-4">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              以 m* = {MOM} 为 pivot，PARTITION 后的三段结果
            </div>

            {/* Partition result */}
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-xl border-2 border-blue-300 dark:border-blue-600 bg-blue-50 dark:bg-blue-900/20 p-3">
                <div className="text-xs font-semibold text-blue-600 dark:text-blue-300 text-center mb-2">
                  小于 m* 的元素
                </div>
                <div className="flex flex-wrap gap-1 justify-center">
                  {LESS_THAN_MOM.map((v, i) => (
                    <Cell key={i} value={v} highlight="less" size="sm" />
                  ))}
                </div>
                <div className="text-center text-blue-700 dark:text-blue-300 font-bold text-sm mt-2">
                  {LESS_THAN_MOM.length} 个
                </div>
              </div>

              <div className="rounded-xl border-2 border-emerald-400 dark:border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20 p-3 flex flex-col items-center justify-center">
                <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-300 text-center mb-2">
                  等于 m*
                </div>
                {EQUAL_MOM.map((v, i) => (
                  <Cell key={i} value={v} highlight="mom" />
                ))}
                <div className="text-center text-emerald-700 dark:text-emerald-300 font-bold text-sm mt-2">
                  m* = {MOM}
                </div>
              </div>

              <div className="rounded-xl border-2 border-rose-300 dark:border-rose-600 bg-rose-50 dark:bg-rose-900/20 p-3">
                <div className="text-xs font-semibold text-rose-600 dark:text-rose-300 text-center mb-2">
                  大于 m* 的元素
                </div>
                <div className="flex flex-wrap gap-1 justify-center">
                  {GREATER_THAN_MOM.map((v, i) => (
                    <Cell key={i} value={v} highlight="greater" size="sm" />
                  ))}
                </div>
                <div className="text-center text-rose-700 dark:text-rose-300 font-bold text-sm mt-2">
                  {GREATER_THAN_MOM.length} 个
                </div>
              </div>
            </div>

            {/* Bar chart */}
            <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-4 space-y-3">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">划分比例（共 15 个元素）</div>
              <div className="space-y-2">
                {[
                  { label: "左侧（< m*）", count: LESS_THAN_MOM.length, color: "bg-blue-400 dark:bg-blue-500" },
                  { label: "右侧（> m*）", count: GREATER_THAN_MOM.length, color: "bg-rose-400 dark:bg-rose-500" },
                ].map(({ label, count, color }) => (
                  <div key={label} className="flex items-center gap-3">
                    <div className="w-28 text-xs text-slate-600 dark:text-slate-400">{label}</div>
                    <div className="flex-1 h-5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${color} rounded-full transition-all duration-500 flex items-center justify-end pr-2`}
                        style={{ width: `${(count / 15) * 100}%` }}>
                        <span className="text-white text-xs font-bold">{count}</span>
                      </div>
                    </div>
                    <div className="w-16 text-xs text-slate-500 dark:text-slate-400">
                      {((count / 15) * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
                {/* 3n/10 reference line explanation */}
                <div className="text-xs text-slate-500 dark:text-slate-400 pl-31 pt-1">
                  ≥ 30%（= 3n/10）保证每次递归子问题 ≤ 7n/10 个元素
                </div>
              </div>
            </div>

            {/* Recurrence */}
            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 p-4">
              <div className="text-xs font-semibold text-emerald-700 dark:text-emerald-300 mb-2">递推关系</div>
              <code className="text-sm font-mono text-emerald-800 dark:text-emerald-200">
                T(n) ≤ T(n/5) + T(7n/10 + 6) + O(n)
              </code>
              <p className="text-xs text-emerald-700 dark:text-emerald-400 mt-2">
                1/5 + 7/10 = 9/10 &lt; 1，因此递推收敛：<strong>T(n) = O(n)</strong>
              </p>
            </div>
          </div>
        )}

        {/* ── Navigation ── */}
        <div className="flex items-center justify-between pt-1">
          <button
            onClick={() => setPhase(p => Math.max(0, p - 1) as Phase)}
            disabled={phase === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ← 上一步
          </button>

          <div className="flex gap-1.5">
            {PHASE_CONFIGS.map((_, i) => (
              <button key={i} onClick={() => setPhase(i as Phase)}
                className={`w-2 h-2 rounded-full transition-all ${
                  i === phase ? "bg-teal-500 w-6" : "bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500"
                }`}
              />
            ))}
          </div>

          <button
            onClick={() => setPhase(p => Math.min(4, p + 1) as Phase)}
            disabled={phase === 4}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-teal-500 hover:bg-teal-600 text-white shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            下一步 →
          </button>
        </div>
      </div>
    </div>
  );
}

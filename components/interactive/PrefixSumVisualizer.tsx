"use client";

import React, { useState, useCallback } from "react";

// ── 工具 ──────────────────────────────────────────────────────────────────────
function buildPrefix(arr: number[]) {
  const pre = [0];
  for (const v of arr) pre.push(pre[pre.length - 1] + v);
  return pre;
}

function buildDiff(arr: number[]): number[] {
  return arr.slice(); // 差分初始化为原数组
}

function applyRangeAdd(diff: number[], l: number, r: number, delta: number) {
  const d = diff.slice();
  d[l] += delta;
  if (r + 1 < d.length) d[r + 1] -= delta;
  return d;
}

function restoreFromDiff(diff: number[]): number[] {
  const arr = diff.slice();
  for (let i = 1; i < arr.length; i++) arr[i] += arr[i - 1];
  return arr;
}

type Mode = "prefix" | "diff";

const PRESETS_ARR = [3, 1, 4, 1, 5, 9, 2, 6];
const PASTEL_COLORS = [
  "bg-blue-500",
  "bg-violet-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-sky-500",
  "bg-orange-500",
  "bg-teal-500",
];

export default function PrefixSumVisualizer() {
  const [mode, setMode] = useState<Mode>("prefix");
  const [arr, setArr] = useState<number[]>(PRESETS_ARR);
  const [inputStr, setInputStr] = useState(PRESETS_ARR.join(", "));
  const [queryL, setQueryL] = useState(2);
  const [queryR, setQueryR] = useState(5);
  const [diffL, setDiffL] = useState(1);
  const [diffR, setDiffR] = useState(4);
  const [diffDelta, setDiffDelta] = useState(3);
  const [diffApplied, setDiffApplied] = useState(false);
  const [showSteps, setShowSteps] = useState(true);

  const n = arr.length;
  const pre = buildPrefix(arr);
  const queryResult = pre[queryR + 1] - pre[queryL];

  // 差分模式
  const [diffArr, setDiffArr] = useState<number[]>(PRESETS_ARR.slice());
  const [diffHistory, setDiffHistory] = useState<{ l: number; r: number; delta: number }[]>([]);

  const handleParseArray = useCallback((s: string) => {
    const parsed = s.split(/[\s,]+/).map(Number).filter((v) => !isNaN(v)).slice(0, 12);
    if (parsed.length >= 2) {
      setArr(parsed);
      setDiffArr(parsed.slice());
      setDiffHistory([]);
      setDiffApplied(false);
      setQueryL(0);
      setQueryR(Math.min(3, parsed.length - 1));
      setDiffL(0);
      setDiffR(Math.min(3, parsed.length - 1));
    }
  }, []);

  const handleApplyDiff = () => {
    const newDiff = applyRangeAdd(diffArr, diffL, diffR, diffDelta);
    setDiffArr(newDiff);
    setDiffHistory((h) => [...h, { l: diffL, r: diffR, delta: diffDelta }]);
    setDiffApplied(true);
  };

  const handleResetDiff = () => {
    setDiffArr(arr.slice());
    setDiffHistory([]);
    setDiffApplied(false);
  };

  const restoredArr = restoreFromDiff(diffArr);
  const maxVal = Math.max(...arr, 1);
  const maxPre = Math.max(...pre, 1);

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-emerald-500/15 dark:bg-emerald-500/20 flex items-center justify-center text-xl">
          ∑
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">前缀和 & 差分数组可视化</h3>
          <p className="text-xs text-text-secondary">前缀和：O(1) 区间查询 ｜ 差分数组：O(1) 区间更新</p>
        </div>
        {/* 模式切换 */}
        <div className="ml-auto flex rounded-lg overflow-hidden border border-border-subtle text-xs">
          <button
            onClick={() => setMode("prefix")}
            className={`px-3 py-1.5 font-medium transition-colors ${mode === "prefix"
              ? "bg-emerald-500/25 text-emerald-700 dark:text-emerald-300"
              : "bg-bg-tertiary text-text-secondary hover:text-text-primary"}`}
          >
            前缀和
          </button>
          <button
            onClick={() => setMode("diff")}
            className={`px-3 py-1.5 font-medium transition-colors ${mode === "diff"
              ? "bg-violet-500/25 text-violet-700 dark:text-violet-300"
              : "bg-bg-tertiary text-text-secondary hover:text-text-primary"}`}
          >
            差分数组
          </button>
        </div>
      </div>

      {/* 输入数组 */}
      <div className="flex gap-2 items-center border-t border-border-subtle pt-3">
        <span className="text-xs text-text-secondary shrink-0">输入数组：</span>
        <input
          type="text"
          value={inputStr}
          onChange={(e) => setInputStr(e.target.value)}
          onBlur={() => handleParseArray(inputStr)}
          onKeyDown={(e) => e.key === "Enter" && handleParseArray(inputStr)}
          className="flex-1 bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-1.5 text-sm font-mono text-text-primary focus:outline-none focus:border-emerald-500 transition-colors"
          placeholder="输入数字，逗号或空格分隔"
        />
        <button
          onClick={() => { const s = PRESETS_ARR.join(", "); setInputStr(s); handleParseArray(s); }}
          className="px-2.5 py-1.5 rounded-lg bg-bg-tertiary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors"
        >
          重置示例
        </button>
      </div>

      {/* ──── 前缀和模式 ──── */}
      {mode === "prefix" && (
        <div className="space-y-4">
          {/* 原数组柱状图 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
            <div className="text-xs text-text-secondary mb-3">原数组 A（蓝色为查询区间 [{queryL}, {queryR}]）</div>
            <div className="flex items-end gap-1" style={{ height: 80 }}>
              {arr.map((v, i) => {
                const inRange = i >= queryL && i <= queryR;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5">
                    <span className="text-[9px] font-mono text-text-tertiary">{v}</span>
                    <div
                      className={`w-full rounded-t transition-all duration-300 ${inRange
                        ? "bg-emerald-500"
                        : PASTEL_COLORS[i % PASTEL_COLORS.length] + " opacity-40"
                        }`}
                      style={{ height: `${Math.max((v / maxVal) * 65, 4)}px` }}
                    />
                    <span className="text-[8px] font-mono text-text-tertiary">{i}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 查询区间控制 */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-text-secondary block mb-1">左边界 l = {queryL}</label>
              <input type="range" min={0} max={queryR} value={queryL}
                onChange={(e) => setQueryL(Number(e.target.value))}
                className="w-full accent-emerald-500" />
            </div>
            <div>
              <label className="text-xs text-text-secondary block mb-1">右边界 r = {queryR}</label>
              <input type="range" min={queryL} max={n - 1} value={queryR}
                onChange={(e) => setQueryR(Number(e.target.value))}
                className="w-full accent-emerald-500" />
            </div>
          </div>

          {/* 前缀和数组 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
            <div className="text-xs text-text-secondary mb-2 flex justify-between">
              <span>前缀和数组 pre（长度 = {n + 1}，pre[0]=0 为哨兵）</span>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={showSteps} onChange={(e) => setShowSteps(e.target.checked)}
                  className="accent-emerald-500 w-3 h-3" />
                <span>显示公式</span>
              </label>
            </div>
            <div className="flex gap-1 flex-wrap">
              {pre.map((v, i) => {
                // pre[queryL] 和 pre[queryR+1] 是参与计算的两个位置
                const isL = i === queryL;
                const isR1 = i === queryR + 1;
                return (
                  <div key={i} className="flex flex-col items-center gap-0.5 min-w-[36px]">
                    <span className={`text-xs font-mono font-bold px-2 py-1 rounded transition-colors
                      ${isR1 ? "bg-emerald-500 text-white" :
                        isL ? "bg-rose-500 text-white" :
                        "bg-bg-secondary text-text-primary border border-border-subtle"}`}>
                      {v}
                    </span>
                    <span className="text-[8px] font-mono text-text-tertiary">pre[{i}]</span>
                  </div>
                );
              })}
            </div>
            {showSteps && (
              <div className="mt-3 px-3 py-2 rounded-lg bg-bg-secondary border border-border-subtle text-xs font-mono space-y-1">
                <div className="text-text-secondary">
                  sum({queryL}, {queryR}) = pre[{queryR + 1}] − pre[{queryL}]
                </div>
                <div className="text-text-secondary">
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span className="text-emerald-600 dark:text-emerald-400">{pre[queryR + 1]}</span> − <span className="text-rose-600 dark:text-rose-400">{pre[queryL]}</span>
                </div>
                <div className="text-text-primary font-semibold">
                  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 🎯 {queryResult}
                </div>
              </div>
            )}
          </div>

          {/* 前缀和柱状图 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
            <div className="text-xs text-text-secondary mb-2">前缀和数组可视化（pre[i] = A[0]+…+A[i-1]）</div>
            <div className="flex items-end gap-1" style={{ height: 60 }}>
              {pre.map((v, i) => {
                const isL = i === queryL;
                const isR1 = i === queryR + 1;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5">
                    <div
                      className={`w-full rounded-t transition-all duration-300
                        ${isR1 ? "bg-emerald-500" : isL ? "bg-rose-500" : "bg-blue-500/40"}`}
                      style={{ height: `${Math.max((v / maxPre) * 55, 2)}px` }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ──── 差分数组模式 ──── */}
      {mode === "diff" && (
        <div className="space-y-4">
          {/* 操作控制 */}
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4 space-y-3">
            <div className="text-xs font-semibold text-text-primary">区间加法操作：A[{diffL}..{diffR}] += {diffDelta}</div>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="text-xs text-text-secondary block mb-1">左边界 l = {diffL}</label>
                <input type="range" min={0} max={diffR} value={diffL}
                  onChange={(e) => setDiffL(Number(e.target.value))}
                  className="w-full accent-violet-500" />
              </div>
              <div>
                <label className="text-xs text-text-secondary block mb-1">右边界 r = {diffR}</label>
                <input type="range" min={diffL} max={n - 1} value={diffR}
                  onChange={(e) => setDiffR(Number(e.target.value))}
                  className="w-full accent-violet-500" />
              </div>
              <div>
                <label className="text-xs text-text-secondary block mb-1">增量 δ = {diffDelta}</label>
                <input type="range" min={-5} max={10} value={diffDelta}
                  onChange={(e) => setDiffDelta(Number(e.target.value))}
                  className="w-full accent-violet-500" />
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleApplyDiff}
                className="px-4 py-1.5 rounded-lg bg-violet-500/20 hover:bg-violet-500/30 border border-violet-400/50 text-violet-700 dark:text-violet-300 text-xs font-medium transition-colors"
              >
                应用区间更新 O(1)
              </button>
              {diffHistory.length > 0 && (
                <button
                  onClick={handleResetDiff}
                  className="px-3 py-1.5 rounded-lg bg-bg-secondary border border-border-subtle text-xs text-text-secondary hover:text-text-primary transition-colors"
                >
                  重置
                </button>
              )}
            </div>
          </div>

          {/* 差分数组与还原结果并排 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 差分数组 */}
            <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
              <div className="text-xs text-text-secondary mb-2">差分数组 diff（更新后）</div>
              <div className="flex gap-1 flex-wrap">
                {diffArr.map((v, i) => {
                  const inRange = i >= diffL && i <= diffR;
                  const isL = diffHistory.length > 0 && i === diffL;
                  const isR1 = diffHistory.length > 0 && i === diffR + 1;
                  return (
                    <div key={i} className="flex flex-col items-center gap-0.5 min-w-[32px]">
                      <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border transition-colors
                        ${isL || isR1 ? "bg-violet-500/25 border-violet-400/60 text-violet-700 dark:text-violet-300" :
                          "bg-bg-secondary border-border-subtle text-text-primary"}`}>
                        {v > 0 ? `+${v}` : v}
                      </span>
                      <span className="text-[8px] text-text-tertiary font-mono">d[{i}]</span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-2 text-[10px] text-text-tertiary">
                区间 [{diffL},{diffR}] 更新：diff[{diffL}]+={diffDelta}，diff[{diffR + 1}]-={diffDelta}
              </div>
            </div>

            {/* 还原数组 */}
            <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-3">
              <div className="text-xs text-text-secondary mb-2">还原数组 = prefix_sum(diff)（O(n)）</div>
              <div className="flex gap-1 flex-wrap">
                {restoredArr.map((v, i) => {
                  const changed = v !== arr[i];
                  return (
                    <div key={i} className="flex flex-col items-center gap-0.5 min-w-[32px]">
                      <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border transition-colors
                        ${changed ? "bg-emerald-500/20 border-emerald-400/50 text-emerald-700 dark:text-emerald-300" :
                          "bg-bg-secondary border-border-subtle text-text-primary"}`}>
                        {v}
                      </span>
                      <span className={`text-[8px] font-mono ${changed ? "text-emerald-600 dark:text-emerald-400" : "text-text-tertiary"}`}>
                        {changed ? `+${v - arr[i]}` : arr[i]}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 操作历史 */}
          {diffHistory.length > 0 && (
            <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs">
              <div className="text-text-secondary font-medium mb-2">操作历史（全部 O(1) 完成）：</div>
              {diffHistory.map((op, i) => (
                <div key={i} className="font-mono text-text-secondary">
                  {i + 1}. A[{op.l}..{op.r}] += {op.delta}
                  <span className="text-text-tertiary">
                    {" "}→ diff[{op.l}]+={op.delta}, diff[{op.r + 1}]-={op.delta}
                  </span>
                </div>
              ))}
              <div className="mt-2 text-violet-700 dark:text-violet-300 font-medium">
                → 最终调用 1 次 O(n) 还原，得到更新后的完整数组
              </div>
            </div>
          )}

          {/* 互逆关系说明 */}
          <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
            <div className="font-semibold text-text-primary">📐 差分 ↔ 前缀和 互逆关系</div>
            <div>diff(A) = A[i] − A[i−1]（差分运算）</div>
            <div>prefix_sum(diff(A)) = A（还原 = 求差分的前缀和）</div>
            <div className="text-text-tertiary">
              两者结合可以实现：O(1) 区间更新 + O(n) 批量还原，
              适合"先批量修改，最后一次性输出"的场景。
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

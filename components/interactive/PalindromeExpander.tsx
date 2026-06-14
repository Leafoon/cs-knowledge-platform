"use client";

import React, { useState, useMemo } from "react";

/** 中心扩展法回文检测可视化：奇/偶中心扩展步进演示，高亮当前窗口与最长回文 */

const PRESETS = ["racecar", "babad", "cbbd", "abcba", "aaaa", "amanaplanacanalpanama"];

interface PalindromeStep {
  centerIdx: number;      // 中心位置（字符位，0-based）
  isOdd: boolean;         // true=奇数长度，false=偶数长度
  radius: number;         // 当前扩展半径（0=仅中心/间隙）
  left: number;           // 当前窗口左边界（字符位）
  right: number;          // 当前窗口右边界（字符位)
  isMatching: boolean;    // 当前两端字符是否匹配
  done: boolean;          // 本中心扩展结束
  palindrome: string;     // 当前回文串
  bestPalindrome: string; // 当前已知最长回文
  bestStart: number;
  bestEnd: number;        // 最长回文区间[bestStart, bestEnd]
  description: string;
}

function buildPalindromeSteps(s: string): PalindromeStep[] {
  const steps: PalindromeStep[] = [];
  const n = s.length;
  let bestPalindrome = s.length > 0 ? s[0] : "";
  let bestStart = 0, bestEnd = 0;

  for (let c = 0; c < 2 * n - 1; c++) {
    const isOdd = c % 2 === 0;
    const centerIdx = Math.floor(c / 2);
    let left = centerIdx, right = isOdd ? centerIdx : centerIdx + 1;

    // 扩展
    steps.push({
      centerIdx, isOdd, radius: 0,
      left, right, isMatching: true, done: false,
      palindrome: isOdd ? s[centerIdx] : s[centerIdx] + s[right] || "",
      bestPalindrome, bestStart, bestEnd,
      description: isOdd
        ? `新中心 c=${c}：字符 '${s[centerIdx]}'（位置 ${centerIdx}），奇数长度中心`
        : `新中心 c=${c}：字符间隙 ${centerIdx}↔${right}，偶数长度中心`,
    });

    let r = 1;
    while (true) {
      const l = centerIdx - r + (isOdd ? 0 : -1 + 1);
      const lo = isOdd ? centerIdx - r : centerIdx - r + 1 - 1;
      const hi = isOdd ? centerIdx + r : centerIdx + r;
      if (lo < 0 || hi >= n) {
        steps.push({
          centerIdx, isOdd, radius: r,
          left: lo + 1, right: hi - 1, isMatching: false, done: true,
          palindrome: s.slice(lo + 1, hi),
          bestPalindrome, bestStart, bestEnd,
          description: `越界：lo=${lo} 或 hi=${hi} 超出字符串边界，本中心扩展停止`,
        });
        break;
      }
      const lchar = s[lo], rchar = s[hi];
      const match = lchar === rchar;
      if (match) {
        const pal = s.slice(lo, hi + 1);
        if (pal.length > bestPalindrome.length) {
          bestPalindrome = pal;
          bestStart = lo;
          bestEnd = hi;
        }
        steps.push({
          centerIdx, isOdd, radius: r,
          left: lo, right: hi, isMatching: true, done: false,
          palindrome: pal, bestPalindrome, bestStart, bestEnd,
          description: `s[${lo}]='${lchar}' == s[${hi}]='${rchar}'，扩展成功！当前回文："${pal}"${pal.length > (steps[steps.length-1]?.bestPalindrome?.length ?? 0) ? " ← 新最长！" : ""}`,
        });
        r++;
      } else {
        steps.push({
          centerIdx, isOdd, radius: r,
          left: lo, right: hi, isMatching: false, done: true,
          palindrome: s.slice(lo + 1, hi),
          bestPalindrome, bestStart, bestEnd,
          description: `s[${lo}]='${lchar}' ≠ s[${hi}]='${rchar}'，扩展终止，本轮最长回文："${s.slice(lo + 1, hi)}"`,
        });
        break;
      }
    }
  }

  steps.push({
    centerIdx: n, isOdd: true, radius: 0,
    left: bestStart, right: bestEnd, isMatching: true, done: true,
    palindrome: bestPalindrome, bestPalindrome, bestStart, bestEnd,
    description: `✅ 完成！最长回文子串："${bestPalindrome}"（长度 ${bestPalindrome.length}，位置 [${bestStart}, ${bestEnd}]）`,
  });

  return steps;
}

export default function PalindromeExpander() {
  const [input, setInput] = useState("racecar");
  const [stepIdx, setStepIdx] = useState(0);
  const [autoMode, setAutoMode] = useState(false);

  const s = input.slice(0, 18) || "a";
  const steps = useMemo(() => buildPalindromeSteps(s), [s]);
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  React.useEffect(() => {
    if (!autoMode) return;
    if (stepIdx >= steps.length - 1) { setAutoMode(false); return; }
    const t = setTimeout(() => setStepIdx((i) => i + 1), 600);
    return () => clearTimeout(t);
  }, [autoMode, stepIdx, steps.length]);

  const reset = () => { setStepIdx(0); setAutoMode(false); };
  const prev = () => setStepIdx((i) => Math.max(0, i - 1));
  const next = () => setStepIdx((i) => Math.min(steps.length - 1, i + 1));

  const getCharStyle = (idx: number): string => {
    if (!step || step.centerIdx >= s.length) {
      // final state
      if (idx >= step.bestStart && idx <= step.bestEnd) return "bg-amber-500/50 text-amber-200 font-bold border-amber-500/70";
      return "bg-bg-tertiary text-text-secondary border-border-subtle";
    }
    const { left, right, isMatching, done, bestStart, bestEnd } = step;

    if (idx >= bestStart && idx <= bestEnd && done)
      return "bg-amber-500/30 text-amber-200 border-amber-500/40";

    if (idx === left && idx === right && step.isOdd)
      return "bg-purple-500 text-white font-bold border-purple-400 ring-2 ring-purple-300/50";
    if ((idx === left || idx === right) && !done)
      return isMatching
        ? "bg-green-500 text-white font-bold border-green-400 ring-2 ring-green-300/50"
        : "bg-red-500 text-white font-bold border-red-400 ring-2 ring-red-300/50";
    if (idx > left && idx < right && !done)
      return "bg-blue-500/30 text-blue-200 border-blue-500/40";
    if (idx >= bestStart && idx <= bestEnd)
      return "bg-amber-500/20 text-amber-300/80 border-amber-500/30";
    return "bg-bg-tertiary text-text-secondary border-border-subtle";
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-bold text-text-primary">🔭 回文中心扩展可视化</h3>
          <p className="text-xs text-text-tertiary mt-0.5">2n-1 个中心逐一展开，O(n²) 时间，O(1) 空间</p>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/40 rounded px-3 py-1 text-xs text-amber-300">
          最长回文："{step?.bestPalindrome}"（{step?.bestPalindrome?.length}）
        </div>
      </div>

      {/* 预设 */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-text-tertiary">预设：</span>
        {PRESETS.map((p) => (
          <button key={p} onClick={() => { setInput(p); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${
              input === p
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
            }`}>
            {p.length > 10 ? p.slice(0, 9) + "…" : p}
          </button>
        ))}
      </div>

      {/* 自定义输入 */}
      <div>
        <label className="text-xs text-text-tertiary mb-1 block">自定义字符串（≤18字符）</label>
        <input type="text" maxLength={18} value={input}
          onChange={(e) => { setInput(e.target.value.toLowerCase().replace(/[^a-z]/g, "")); reset(); }}
          className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
          placeholder="输入纯字母..."
        />
      </div>

      {/* 字符串可视化 */}
      <div>
        <div className="text-xs text-text-tertiary mb-2">字符串（n={s.length}，共 {2 * s.length - 1} 个中心位）</div>
        <div className="flex gap-1 items-end flex-wrap">
          {s.split("").map((ch, idx) => (
            <React.Fragment key={idx}>
              <div className="flex flex-col items-center">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center text-base border-2 transition-all duration-200 ${getCharStyle(idx)}`}>
                  {ch}
                </div>
                <span className="text-[9px] text-text-tertiary mt-0.5">{idx}</span>
                {/* 奇数中心 marker */}
                {step && step.isOdd && step.centerIdx === idx && step.centerIdx < s.length && (
                  <span className="text-[9px] text-purple-400 mt-0.5">▲奇</span>
                )}
              </div>
              {/* 偶数中心 gap marker */}
              {idx < s.length - 1 && (
                <div className="flex flex-col items-center self-center pb-4">
                  <div className={`w-1.5 h-5 rounded-full transition-colors ${
                    step && !step.isOdd && step.centerIdx === idx ? "bg-purple-400" : "bg-border-subtle/40"
                  }`} />
                  {step && !step.isOdd && step.centerIdx === idx && (
                    <span className="text-[8px] text-purple-400 mt-0.5">偶</span>
                  )}
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg p-3 border min-h-[64px] transition-all ${
        stepIdx >= steps.length - 1 ? "bg-amber-500/10 border-amber-500/50"
        : step?.isMatching && !step?.done ? "bg-green-500/10 border-green-500/40"
        : step?.done && !step?.isMatching ? "bg-bg-tertiary border-border-subtle"
        : "bg-bg-tertiary border-border-subtle"
      }`}>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-tertiary">步骤 {stepIdx + 1} / {steps.length}</span>
          <span className="text-text-tertiary">
            {step?.centerIdx < s.length
              ? (step?.isOdd ? `奇心 c=${step?.centerIdx}` : `偶隙 ${step?.centerIdx}|${step?.centerIdx! + 1}`)
              : "完成"}
          </span>
        </div>
        <p className={`text-sm ${
          stepIdx >= steps.length - 1 ? "text-amber-300"
          : step?.isMatching && !step?.done ? "text-green-300"
          : "text-text-primary"
        }`}>
          {step?.description}
        </p>
      </div>

      {/* 当前扩展状态 */}
      {step?.palindrome && step.centerIdx < s.length && (
        <div className="bg-bg-tertiary rounded p-2 flex gap-4 text-xs text-text-secondary border border-border-subtle">
          <span>当前回文：<span className="font-bold text-blue-300">"{step.palindrome}"</span></span>
          <span>半径：{step.radius}</span>
          <span>窗口：[{step.left}, {step.right}]</span>
        </div>
      )}

      {/* 图例 */}
      <div className="flex gap-3 flex-wrap text-xs text-text-secondary">
        <span><span className="inline-block w-3 h-3 rounded bg-purple-500 mr-1" />当前中心</span>
        <span><span className="inline-block w-3 h-3 rounded bg-green-500 mr-1" />扩展匹配</span>
        <span><span className="inline-block w-3 h-3 rounded bg-red-500 mr-1" />扩展失配</span>
        <span><span className="inline-block w-3 h-3 rounded bg-blue-500/40 mr-1" />窗口内</span>
        <span><span className="inline-block w-3 h-3 rounded bg-amber-500/50 mr-1" />最长回文</span>
      </div>

      {/* 控制 */}
      <div className="flex gap-2 justify-center flex-wrap">
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs">↩ 重置</button>
        <button onClick={prev} disabled={stepIdx === 0} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs disabled:opacity-40">← 上一步</button>
        <button onClick={next} disabled={stepIdx >= steps.length - 1} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs hover:bg-blue-700 disabled:opacity-40">下一步 →</button>
        <button onClick={() => setAutoMode((b) => !b)} className={`px-4 py-2 rounded-lg text-xs border transition-colors ${autoMode ? "bg-amber-600 text-white border-amber-600" : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-amber-400"}`}>
          {autoMode ? "⏸ 暂停" : "▶ 自动播放"}
        </button>
      </div>

      {/* 算法对比 */}
      <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
        <div className="text-xs text-text-tertiary mb-2 font-bold">算法对比：中心扩展 vs Manacher</div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-blue-500/10 rounded p-2 border border-blue-500/30">
            <div className="font-bold text-blue-300 mb-1">中心扩展（本演示）</div>
            <div className="text-text-secondary">时间：O(n²)</div>
            <div className="text-text-secondary">空间：O(1)</div>
            <div className="text-text-secondary">简单易实现</div>
          </div>
          <div className="bg-purple-500/10 rounded p-2 border border-purple-500/30">
            <div className="font-bold text-purple-300 mb-1">Manacher 算法</div>
            <div className="text-text-secondary">时间：O(n)</div>
            <div className="text-text-secondary">空间：O(n)</div>
            <div className="text-text-secondary">利用对称复用，面试进阶</div>
          </div>
        </div>
      </div>
    </div>
  );
}

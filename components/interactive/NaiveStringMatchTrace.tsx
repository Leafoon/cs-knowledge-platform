"use client";

import React, { useState, useCallback } from "react";

/** 朴素字符串匹配步进动画：模式串逐位滑动，失配回退，高亮显示 */

const PRESETS: Record<string, { text: string; pattern: string }> = {
  "最坏情况": { text: "aaaaaab", pattern: "aaab" },
  "正常匹配": { text: "abababab", pattern: "abab" },
  "多次匹配": { text: "aaaaaa", pattern: "aa" },
  "无匹配": { text: "abcdefgh", pattern: "xyz" },
  "末位失配": { text: "aaabaab", pattern: "aaaa" },
};

interface MatchStep {
  i: number;           // 文本起始偏移（外循环）
  j: number;           // 模式当前比较位（内循环）
  match: boolean;      // 当前位是否匹配
  completed: boolean;  // 本轮完整匹配成功
  failed: boolean;     // 本轮已失败（内循环退出）
  comparisons: number; // 累计比较次数
  description: string;
}

function buildSteps(text: string, pattern: string): MatchStep[] {
  const steps: MatchStep[] = [];
  const n = text.length, m = pattern.length;
  let totalCmp = 0;

  for (let i = 0; i <= n - m; i++) {
    let j = 0;
    // scan step at start of each outer iteration
    steps.push({
      i, j, match: text[i] === pattern[0], completed: false, failed: false,
      comparisons: totalCmp,
      description: `外循环 i=${i}：从文本位置 ${i} 开始尝试匹配`,
    });

    while (j < m) {
      const isMatch = text[i + j] === pattern[j];
      totalCmp++;
      if (isMatch) {
        steps.push({
          i, j, match: true, completed: j === m - 1, failed: false,
          comparisons: totalCmp,
          description: j === m - 1
            ? `✅ text[${i + j}]='${text[i + j]}' == pattern[${j}]='${pattern[j]}'，全部匹配！在位置 ${i} 找到匹配`
            : `text[${i + j}]='${text[i + j]}' == pattern[${j}]='${pattern[j]}'，继续匹配`,
        });
        j++;
      } else {
        steps.push({
          i, j, match: false, completed: false, failed: true,
          comparisons: totalCmp,
          description: `❌ text[${i + j}]='${text[i + j]}' ≠ pattern[${j}]='${pattern[j]}'，失配，i 后移到 ${i + 1}`,
        });
        break;
      }
    }
  }

  steps.push({
    i: n, j: 0, match: false, completed: false, failed: false,
    comparisons: totalCmp,
    description: `算法结束，共比较 ${totalCmp} 次（最坏 O(nm)=${n * m}）`,
  });

  return steps;
}

export default function NaiveStringMatchTrace() {
  const [preset, setPreset] = useState("最坏情况");
  const [customText, setCustomText] = useState("");
  const [customPattern, setCustomPattern] = useState("");
  const [stepIdx, setStepIdx] = useState(0);

  const text = customText || PRESETS[preset].text;
  const pattern = customPattern || PRESETS[preset].pattern;
  const steps = buildSteps(text.slice(0, 20), pattern.slice(0, 8));
  const step = steps[Math.min(stepIdx, steps.length - 1)];

  const reset = () => setStepIdx(0);
  const prev = () => setStepIdx((s) => Math.max(0, s - 1));
  const next = () => setStepIdx((s) => Math.min(steps.length - 1, s + 1));

  // 已成功匹配的位置集合
  const matchedPositions = new Set<number>();
  for (let k = 0; k <= stepIdx; k++) {
    const s = steps[k];
    if (s.completed) matchedPositions.add(s.i);
  }

  const getCellStyle = useCallback(
    (charIdx: number, isText: boolean): string => {
      if (!step) return "";
      if (isText) {
        const { i, j, match, completed, failed } = step;
        const isActive = charIdx === i + j && !completed;
        const inWindow = charIdx >= i && charIdx < i + pattern.length && !completed && !failed;
        const isMatched = matchedPositions.has(i) && charIdx >= i && charIdx < i + pattern.length && completed;

        if (completed && charIdx >= step.i && charIdx < step.i + pattern.length)
          return "bg-green-500 text-white font-bold";
        if (matchedPositions.has(charIdx - /* any matching start */ 0)) {
          // already-matched highlight handled below
        }
        if (isActive && match) return "bg-green-400 text-white font-bold ring-2 ring-green-200";
        if (isActive && !match) return "bg-red-500 text-white font-bold ring-2 ring-red-300";
        if (inWindow) return "bg-blue-500/30 text-blue-200";
        return "bg-bg-tertiary text-text-secondary";
      } else {
        // pattern char
        const { j, match, completed, failed } = step;
        if (completed) return "bg-green-500 text-white font-bold";
        if (charIdx === j && !failed && !completed) {
          return match ? "bg-green-400 text-white font-bold" : "bg-red-500 text-white font-bold";
        }
        if (charIdx < j) return "bg-blue-500/40 text-blue-200";
        return "bg-bg-tertiary text-text-secondary";
      }
    },
    [step, matchedPositions, pattern.length]
  );

  // Recompute matched positions cleanly
  const cleanMatchedStarts = new Set<number>();
  for (let k = 0; k <= stepIdx; k++) {
    if (steps[k]?.completed) cleanMatchedStarts.add(steps[k].i);
  }

  const getTextCellStyle = (charIdx: number): string => {
    if (!step) return "bg-bg-tertiary text-text-secondary";
    const { i, j, match, completed, failed } = step;

    // Already-found matches
    for (const matchStart of cleanMatchedStarts) {
      if (charIdx >= matchStart && charIdx < matchStart + pattern.length)
        return "bg-green-600/40 text-green-200 font-bold";
    }

    // Current window highlight
    const windowEnd = i + pattern.length - 1;
    const activeChar = i + j;

    if (charIdx === activeChar) {
      if (match) return "bg-green-400 text-white font-bold ring-2 ring-green-200";
      return "bg-red-500 text-white font-bold ring-2 ring-red-300";
    }
    if (charIdx > i && charIdx < activeChar) return "bg-blue-500/40 text-blue-200";
    if (charIdx === i && j > 0) return "bg-blue-500/40 text-blue-200";
    if (charIdx >= i && charIdx <= windowEnd && !failed)
      return "bg-bg-tertiary border border-blue-500/30 text-text-primary";
    return "bg-bg-tertiary text-text-secondary";
  };

  const getCmpColor = () => {
    const ratio = step.comparisons / (text.length * pattern.length);
    if (ratio > 0.7) return "text-red-400";
    if (ratio > 0.4) return "text-amber-400";
    return "text-green-400";
  };

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-base font-bold text-text-primary">🔍 朴素字符串匹配追踪</h3>
          <p className="text-xs text-text-tertiary mt-0.5">O(nm) 最坏情况—每次失配从头重试</p>
        </div>
        <div className={`text-sm font-bold px-3 py-1 rounded border border-border-subtle bg-bg-tertiary ${getCmpColor()}`}>
          比较次数：{step?.comparisons ?? 0} / {text.length * pattern.length}（上界）
        </div>
      </div>

      {/* 预设选择 */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-text-tertiary">预设：</span>
        {Object.keys(PRESETS).map((p) => (
          <button key={p} onClick={() => { setPreset(p); setCustomText(""); setCustomPattern(""); reset(); }}
            className={`px-2 py-1 rounded text-xs border transition-colors ${
              preset === p && !customText
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
            }`}>
            {p}
          </button>
        ))}
      </div>

      {/* 自定义输入 */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-tertiary mb-1 block">文本串（≤20字符）</label>
          <input type="text" maxLength={20} value={customText}
            onChange={(e) => { setCustomText(e.target.value); reset(); }}
            placeholder={PRESETS[preset].text}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
          />
        </div>
        <div>
          <label className="text-xs text-text-tertiary mb-1 block">模式串（≤8字符）</label>
          <input type="text" maxLength={8} value={customPattern}
            onChange={(e) => { setCustomPattern(e.target.value); reset(); }}
            placeholder={PRESETS[preset].pattern}
            className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
          />
        </div>
      </div>

      {/* 文本串可视化 */}
      <div>
        <div className="text-xs text-text-tertiary mb-1">文本串（T，长度 n={text.length}）</div>
        <div className="flex gap-1 flex-wrap">
          {text.split("").map((ch, idx) => (
            <div key={idx} className="flex flex-col items-center">
              <div className={`w-8 h-8 rounded flex items-center justify-center text-sm border transition-all duration-200 ${getTextCellStyle(idx)}`}>
                {ch}
              </div>
              <span className="text-[9px] text-text-tertiary mt-0.5">{idx}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 偏移指示器 */}
      {step && step.i < text.length && (
        <div className="flex gap-1">
          {Array.from({ length: text.length }, (_, idx) => (
            <div key={idx} className="w-8 flex justify-center">
              {idx === step.i ? (
                <span className="text-amber-400 text-xs font-bold">↑i</span>
              ) : null}
            </div>
          ))}
        </div>
      )}

      {/* 模式串可视化 */}
      <div>
        <div className="text-xs text-text-tertiary mb-1">模式串（P，长度 m={pattern.length}）</div>
        <div className="flex gap-1 flex-wrap">
          {pattern.split("").map((ch, idx) => (
            <div key={idx} className="flex flex-col items-center">
              <div className={`w-8 h-8 rounded flex items-center justify-center text-sm border transition-all duration-200 ${getCellStyle(idx, false)}`}>
                {ch}
              </div>
              <span className="text-[9px] text-text-tertiary mt-0.5">{idx}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 步骤说明 */}
      <div className={`rounded-lg p-3 border min-h-[56px] transition-colors ${
        step?.completed ? "bg-green-500/10 border-green-500/50"
        : step?.failed ? "bg-red-500/10 border-red-500/50"
        : "bg-bg-tertiary border-border-subtle"
      }`}>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-text-tertiary">步骤 {stepIdx + 1} / {steps.length}</span>
          <span className="text-text-tertiary">i={step?.i ?? 0}，j={step?.j ?? 0}</span>
        </div>
        <p className={`text-sm ${step?.completed ? "text-green-300" : step?.failed ? "text-red-300" : "text-text-primary"}`}>
          {step?.description}
        </p>
      </div>

      {/* 图例 */}
      <div className="flex gap-3 flex-wrap text-xs text-text-secondary">
        <span><span className="inline-block w-3 h-3 rounded bg-green-400 mr-1" />匹配</span>
        <span><span className="inline-block w-3 h-3 rounded bg-red-500 mr-1" />失配</span>
        <span><span className="inline-block w-3 h-3 rounded bg-blue-500/40 mr-1" />窗口内已匹配</span>
        <span><span className="inline-block w-3 h-3 rounded bg-green-600/40 mr-1" />找到的匹配</span>
      </div>

      {/* 控制 */}
      <div className="flex gap-2 justify-center">
        <button onClick={reset} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors">↩ 重置</button>
        <button onClick={prev} disabled={stepIdx === 0} className="px-4 py-2 rounded-lg bg-bg-tertiary text-text-secondary border border-border-subtle hover:border-blue-400 text-xs transition-colors disabled:opacity-40">← 上一步</button>
        <button onClick={next} disabled={stepIdx >= steps.length - 1} className="px-4 py-2 rounded-lg bg-blue-600 text-white text-xs hover:bg-blue-700 transition-colors disabled:opacity-40">下一步 →</button>
      </div>

      {/* 匹配结果汇总 */}
      {stepIdx >= steps.length - 1 && (
        <div className="bg-bg-tertiary border border-border-subtle rounded-lg p-3">
          <div className="text-xs text-text-tertiary mb-2">匹配结果汇总</div>
          {cleanMatchedStarts.size === 0 ? (
            <div className="text-text-secondary text-sm">未找到匹配</div>
          ) : (
            <div className="flex gap-2 flex-wrap">
              {[...cleanMatchedStarts].sort((a, b) => a - b).map((pos) => (
                <div key={pos} className="bg-green-600/20 border border-green-500/50 rounded px-2 py-1 text-xs text-green-300">
                  位置 {pos}："{text.slice(pos, pos + pattern.length)}"
                </div>
              ))}
            </div>
          )}
          <div className="text-xs text-text-tertiary mt-2">
            总比较次数：<span className={getCmpColor()}>{step.comparisons}</span>
            <span className="ml-3">理论最坏：n×m = {text.length}×{pattern.length} = {text.length * pattern.length}</span>
          </div>
        </div>
      )}
    </div>
  );
}

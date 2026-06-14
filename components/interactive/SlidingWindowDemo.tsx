"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── 类型 ───────────────────────────────────────────────────────────────────────
type Mode = "norepeat" | "fixed" | "minlen";

interface SWStep {
  l: number;
  r: number;
  label: string;
  action: "expand" | "shrink" | "update" | "done";
  windowContent: string | number[];
  windowMax?: number;
  charMap?: Record<string, number>;
  windowSum?: number;
  k?: number;
}

// ── 算法：无重复字符最长子串 ─────────────────────────────────────────────────────
function noRepeatSteps(s: string): SWStep[] {
  const steps: SWStep[] = [];
  const charIdx: Record<string, number> = {};
  let l = 0, maxLen = 0;
  for (let r = 0; r < s.length; r++) {
    const c = s[r];
    if (charIdx[c] !== undefined && charIdx[c] >= l) {
      const prevL = l;
      l = charIdx[c] + 1;
      steps.push({
        l, r,
        label: `s[${r}]='${c}'已在窗口（位置${charIdx[c]}），l跳至${l}（跳过重复）`,
        action: "shrink",
        windowContent: s.slice(l, r + 1),
        charMap: { ...charIdx },
        windowMax: maxLen,
      });
    }
    charIdx[c] = r;
    const len = r - l + 1;
    if (len > maxLen) {
      maxLen = len;
      steps.push({
        l, r,
        label: `s[${r}]='${c}'加入，窗口"${s.slice(l, r + 1)}"长度=${len}→更新最大=${maxLen}`,
        action: "update",
        windowContent: s.slice(l, r + 1),
        charMap: { ...charIdx },
        windowMax: maxLen,
      });
    } else {
      steps.push({
        l, r,
        label: `s[${r}]='${c}'加入，窗口"${s.slice(l, r + 1)}"长度=${len}，最大=${maxLen}`,
        action: "expand",
        windowContent: s.slice(l, r + 1),
        charMap: { ...charIdx },
        windowMax: maxLen,
      });
    }
  }
  steps.push({ l, r: s.length - 1, label: `扫描完毕！最长无重复子串长度 = ${maxLen}`, action: "done", windowContent: s.slice(l), windowMax: maxLen });
  return steps;
}

// ── 算法：固定窗口最大子数组和 ───────────────────────────────────────────────────
function fixedWindowSteps(nums: number[], k: number): SWStep[] {
  const steps: SWStep[] = [];
  let sum = nums.slice(0, k).reduce((a, b) => a + b, 0);
  let best = sum;
  steps.push({ l: 0, r: k - 1, label: `初始窗口[0..${k - 1}]，和=${sum}，最大=${best}`, action: "expand", windowContent: nums.slice(0, k), windowSum: sum, k, windowMax: best });
  for (let r = k; r < nums.length; r++) {
    const l = r - k + 1;
    sum = sum - nums[r - k] + nums[r];
    if (sum > best) {
      best = sum;
      steps.push({ l, r, label: `窗口右移→[${l}..${r}]，和=${sum}，更新最大=${best}`, action: "update", windowContent: nums.slice(l, r + 1), windowSum: sum, k, windowMax: best });
    } else {
      steps.push({ l, r, label: `窗口右移→[${l}..${r}]，和=${sum}，最大=${best}`, action: "expand", windowContent: nums.slice(l, r + 1), windowSum: sum, k, windowMax: best });
    }
  }
  steps.push({ l: nums.length - k, r: nums.length - 1, label: `完成！最大子数组和 = ${best}（固定窗口 k=${k}）`, action: "done", windowContent: nums.slice(nums.length - k), windowSum: sum, windowMax: best, k });
  return steps;
}

// ── 算法：最小覆盖（和 ≥ target） ────────────────────────────────────────────────
function minLenSteps(nums: number[], target: number): SWStep[] {
  const steps: SWStep[] = [];
  let l = 0, sum = 0, minLen = Infinity;
  for (let r = 0; r < nums.length; r++) {
    sum += nums[r];
    steps.push({ l, r, label: `r=${r}加入，和=${sum}`, action: "expand", windowContent: nums.slice(l, r + 1), windowSum: sum, windowMax: minLen === Infinity ? -1 : minLen });
    while (sum >= target) {
      const len = r - l + 1;
      if (len < minLen) minLen = len;
      steps.push({ l, r, label: `和=${sum}≥${target}，窗口[${l}..${r}]长度=${len}，最小=${minLen}，收缩l`, action: sum >= target ? "update" : "shrink", windowContent: nums.slice(l, r + 1), windowSum: sum, windowMax: minLen });
      sum -= nums[l];
      l++;
    }
  }
  steps.push({ l, r: nums.length - 1, label: minLen === Infinity ? "无解（无法达到目标和）" : `完成！最小长度 = ${minLen}`, action: "done", windowContent: [], windowMax: minLen });
  return steps;
}

// ── 常量 ───────────────────────────────────────────────────────────────────────
const MODE_DEFS: Record<Mode, { label: string; desc: string }> = {
  norepeat: { label: "最长无重复子串", desc: "可变窗口·字符串" },
  fixed:    { label: "固定窗口最大和", desc: "固定窗口·数组" },
  minlen:   { label: "最小满足子数组", desc: "可变窗口·数组" },
};

const ACTION_BADGE: Record<SWStep["action"], { text: string; cls: string }> = {
  expand: { text: "扩张 r→",  cls: "bg-blue-500/15 text-blue-700 dark:text-blue-300" },
  shrink: { text: "← 收缩 l", cls: "bg-rose-500/15 text-rose-700 dark:text-rose-300" },
  update: { text: "★ 最优",   cls: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300" },
  done:   { text: "完成",     cls: "bg-violet-500/15 text-violet-700 dark:text-violet-300" },
};

export default function SlidingWindowDemo() {
  const [mode, setMode] = useState<Mode>("norepeat");
  const [strInput, setStrInput] = useState("abcabcbb");
  const [numInput, setNumInput] = useState("2 3 1 2 4 3");
  const [k, setK] = useState(3);
  const [target, setTarget] = useState(7);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const nums = numInput.trim().split(/\s+/).map(Number).filter((n) => !isNaN(n));
  const str = strInput.trim() || "abcabcbb";

  const steps = React.useMemo(() => {
    if (mode === "norepeat") return noRepeatSteps(str);
    if (mode === "fixed") return fixedWindowSteps(nums, Math.max(1, Math.min(k, nums.length)));
    return minLenSteps(nums, target);
  }, [mode, str, nums.join(","), k, target]);

  const cur = steps[Math.min(step, steps.length - 1)];

  const startPlay = useCallback(() => {
    if (step >= steps.length - 1) setStep(0);
    setPlaying(true);
  }, [step, steps.length]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setStep((s) => { if (s >= steps.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, 850);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, steps.length]);

  const reset = () => { setStep(0); setPlaying(false); };

  const isString = mode === "norepeat";
  const items = isString ? str.split("") : nums;
  const { l, r, action } = cur;
  const badge = ACTION_BADGE[action];

  const maxNum = nums.length ? Math.max(...nums) : 1;

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-5 my-6 shadow-sm space-y-4">
      {/* 标题 */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-sky-500/15 dark:bg-sky-500/20 flex items-center justify-center text-xl">🪟</div>
        <div>
          <h3 className="font-bold text-text-primary text-base">滑动窗口可视化</h3>
          <p className="text-xs text-text-secondary">可变/固定窗口在字符串与数组上的步进动画</p>
        </div>
      </div>

      {/* 模式切换 */}
      <div className="flex flex-wrap gap-2 border-t border-border-subtle pt-3">
        {(Object.keys(MODE_DEFS) as Mode[]).map((m) => (
          <button key={m} onClick={() => { setMode(m); reset(); }}
            className={`px-3 py-1.5 rounded-xl border text-xs font-medium transition-all ${mode === m
              ? "bg-sky-500/20 border-sky-400/60 text-sky-700 dark:text-sky-300"
              : "bg-bg-tertiary border-border-subtle text-text-secondary hover:text-text-primary"}`}>
            {MODE_DEFS[m].label}
            <span className="ml-1 text-[9px] opacity-60">{MODE_DEFS[m].desc}</span>
          </button>
        ))}
      </div>

      {/* 输入控制 */}
      {isString ? (
        <div className="flex items-center gap-3 flex-wrap">
          <label className="text-xs text-text-secondary">字符串：</label>
          <input value={strInput} onChange={(e) => { setStrInput(e.target.value.replace(/\s/g, "")); reset(); }}
            className="flex-1 min-w-0 bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-1.5 text-xs font-mono text-text-primary outline-none focus:border-sky-400/60" />
        </div>
      ) : (
        <div className="flex flex-wrap gap-2 items-center">
          <label className="text-xs text-text-secondary">数组（空格分隔）：</label>
          <input value={numInput} onChange={(e) => { setNumInput(e.target.value); reset(); }}
            className="flex-1 min-w-0 bg-bg-tertiary border border-border-subtle rounded-lg px-3 py-1.5 text-xs font-mono text-text-primary outline-none focus:border-sky-400/60" />
          {mode === "fixed" && (
            <>
              <label className="text-xs text-text-secondary">k=</label>
              <input type="number" min={1} max={nums.length} value={k}
                onChange={(e) => { setK(Number(e.target.value)); reset(); }}
                className="w-14 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1.5 text-xs font-mono text-text-primary outline-none" />
            </>
          )}
          {mode === "minlen" && (
            <>
              <label className="text-xs text-text-secondary">目标和=</label>
              <input type="number" min={1} value={target}
                onChange={(e) => { setTarget(Number(e.target.value)); reset(); }}
                className="w-16 bg-bg-tertiary border border-border-subtle rounded-lg px-2 py-1.5 text-xs font-mono text-text-primary outline-none" />
            </>
          )}
        </div>
      )}

      {/* 控制按钮 + 进度条 */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <button onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">‹</button>
          <button onClick={playing ? () => setPlaying(false) : startPlay}
            className="px-3 py-1 rounded-lg bg-sky-500/15 dark:bg-sky-500/20 hover:bg-sky-500/25 text-sky-700 dark:text-sky-300 text-xs font-medium transition-colors">
            {playing ? "⏸ 暂停" : "▶ 播放"}
          </button>
          <button onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={step >= steps.length - 1}
            className="w-7 h-7 rounded-lg bg-bg-tertiary hover:bg-border-subtle disabled:opacity-30 text-text-primary text-xs flex items-center justify-center">›</button>
          <button onClick={reset}
            className="px-2 py-1 rounded-lg bg-bg-tertiary hover:bg-border-subtle text-text-secondary text-xs transition-colors">重置</button>
          <span className="ml-auto text-[10px] text-text-tertiary">{step + 1}/{steps.length}</span>
        </div>
        <div className="h-1 bg-bg-tertiary rounded-full overflow-hidden">
          <div className="h-full bg-sky-500 rounded-full transition-all duration-300"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
        </div>
      </div>

      {/* 当前步骤说明 */}
      <div className={`rounded-lg px-3 py-2 text-xs font-medium border border-border-subtle ${badge.cls}`}>
        <span className="mr-2 font-semibold">[{badge.text}]</span>{cur.label}
      </div>

      {/* 主视图：字符/数字可视化 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-4">
        {isString ? (
          /* 字符串视图 */
          <div className="space-y-3">
            <div className="text-xs text-text-secondary mb-1">
              字符串：<span className="font-mono text-text-primary">{str}</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {items.map((ch, i) => {
                const inWindow = i >= l && i <= r;
                const isL = i === l;
                const isR = i === r;
                return (
                  <div key={i} className="flex flex-col items-center gap-0.5">
                    <div className={`w-9 h-10 rounded-lg flex items-center justify-center text-sm font-bold font-mono border transition-all duration-300
                      ${isL && isR ? "bg-sky-500 border-sky-400 text-white ring-2 ring-sky-300" :
                        isL ? "bg-blue-500 border-blue-400 text-white" :
                        isR ? "bg-sky-400 border-sky-300 text-white" :
                        inWindow ? "bg-sky-500/20 border-sky-400/40 text-sky-700 dark:text-sky-300" :
                        "bg-bg-secondary border-border-subtle text-text-primary"}`}>
                      {ch}
                    </div>
                    <span className="text-[8px] font-mono text-text-tertiary">{i}</span>
                  </div>
                );
              })}
            </div>
            <div className="flex items-center gap-4 text-xs text-text-secondary flex-wrap">
              {cur.charMap && (
                <div>
                  <span className="text-text-tertiary">字符位置表：</span>
                  {Object.entries(cur.charMap).map(([c, idx]) => (
                    <span key={c} className="font-mono ml-1">{`'${c}':${idx}`}</span>
                  ))}
                </div>
              )}
              {cur.windowMax !== undefined && cur.windowMax >= 0 && (
                <div className="text-emerald-600 dark:text-emerald-300 font-medium">
                  当前最长 = {cur.windowMax}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* 数字数组视图（带柱状高度） */
          <div className="space-y-2">
            <div className="text-xs text-text-secondary mb-1">
              数组 nums，窗口 [{l}..{r}]
              {cur.k && <span className="ml-2 text-text-tertiary">固定 k={cur.k}</span>}
              {mode === "minlen" && <span className="ml-2 text-text-tertiary">目标和={target}</span>}
            </div>
            <div className="flex items-end gap-1.5" style={{ height: 80 }}>
              {(items as number[]).map((v, i) => {
                const inWindow = i >= l && i <= r;
                const isL = i === l;
                const isR = i === r;
                const barH = Math.max(4, (v / maxNum) * 72);
                return (
                  <div key={i} className="flex-1 flex flex-col items-center justify-end gap-0.5">
                    <div className={`w-full rounded-t transition-all duration-300
                      ${isL && isR ? "bg-sky-500" :
                        isL ? "bg-blue-500" :
                        isR ? "bg-sky-400" :
                        inWindow ? "bg-sky-400/50" :
                        "bg-bg-secondary border border-border-subtle"}`}
                      style={{ height: `${barH}px` }} />
                    <span className="text-[8px] font-mono text-text-tertiary">{v}</span>
                  </div>
                );
              })}
            </div>
            <div className="flex flex-wrap gap-4 text-xs text-text-secondary mt-1">
              {cur.windowSum !== undefined && (
                <span>当前窗口和 = <span className="text-text-primary font-medium">{cur.windowSum}</span></span>
              )}
              {cur.windowMax !== undefined && cur.windowMax > 0 && (
                <span className="text-emerald-600 dark:text-emerald-300 font-medium">
                  {mode === "fixed" ? `最大和 = ${cur.windowMax}` : `最小长度 = ${cur.windowMax}`}
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 窗口内容预览 */}
      <div className="rounded-xl border border-border-subtle bg-bg-tertiary px-4 py-3">
        <div className="text-[10px] text-text-tertiary mb-1">当前窗口内容</div>
        <div className="text-sm font-mono text-sky-700 dark:text-sky-300">
          [{isString ? `"${cur.windowContent}"` : (cur.windowContent as number[]).join(", ")}]
          <span className="ml-2 text-xs text-text-tertiary">长度 = {Array.isArray(cur.windowContent) ? cur.windowContent.length : (cur.windowContent as string).length}</span>
        </div>
      </div>

      {/* 复杂度说明 */}
      <div className="rounded-lg border border-border-subtle bg-bg-tertiary p-3 text-xs text-text-secondary space-y-1">
        <div className="font-semibold text-text-primary">
          {mode === "norepeat" ? "可变窗口·最长无重复子串 (LeetCode 3)" :
           mode === "fixed"    ? "固定窗口·最大子数组和 (k 固定)" :
           "可变窗口·最小满足子数组 (LeetCode 209)"}
        </div>
        <div className="text-text-secondary">
          {mode === "norepeat" ? "r 不断向右，遇重复字符时 l 跳至 charIdx[c]+1；charIdx 维护每个字符最近出现位置，每字符最多入/出窗口各一次 → O(n)。" :
           mode === "fixed"    ? "先计算初始窗口和，每次新增右端元素、减去左端元素，O(1) 维护窗口和 → 整体 O(n)，避免重复求和 O(nk)。" :
           "r 扩张加元素、满足条件时 l 收缩并记录长度；两指针各最多移动 n 步 → O(n)，类似两端摊销分析。"}
        </div>
        <div className="text-text-tertiary">时间 O(n)，空间 O(|Σ|) 或 O(1)（仅数字情形）</div>
      </div>
    </div>
  );
}

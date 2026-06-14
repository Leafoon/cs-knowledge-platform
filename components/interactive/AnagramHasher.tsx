"use client";

import React, { useState, useMemo } from "react";

/** 异位词哈希可视化：字符频次直方图对比，支持单词分组演示 */

const LETTER_PAIRS = [
  { s: "anagram", t: "nagaram", label: "经典异位词" },
  { s: "rat", t: "car", label: "非异位词" },
  { s: "listen", t: "silent", label: "Listen/Silent" },
  { s: "hello", t: "world", label: "无关两词" },
  { s: "abcabc", t: "abcabc", label: "完全相同" },
];

const GROUP_PRESETS: Record<string, string[]> = {
  "经典三组": ["eat", "tea", "tan", "ate", "nat", "bat"],
  "字母顺序": ["abc", "bca", "cab", "def", "fed"],
  "混合大小写": ["listen", "silent", "enlist", "inlets"],
};

const ALPHABET = "abcdefghijklmnopqrstuvwxyz".split("");

function getFreq(str: string): Record<string, number> {
  const freq: Record<string, number> = {};
  for (const ch of str.toLowerCase()) {
    if (/[a-z]/.test(ch)) freq[ch] = (freq[ch] || 0) + 1;
  }
  return freq;
}

function groupByAnagram(words: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const w of words) {
    const key = w.split("").sort().join("");
    if (!groups[key]) groups[key] = [];
    groups[key].push(w);
  }
  return groups;
}

type Tab = "compare" | "group";

export default function AnagramHasher() {
  const [tab, setTab] = useState<Tab>("compare");
  const [pairIdx, setPairIdx] = useState(0);
  const [customS, setCustomS] = useState("");
  const [customT, setCustomT] = useState("");
  const [groupPreset, setGroupPreset] = useState("经典三组");
  const [animStep, setAnimStep] = useState(0);

  // Compare tab
  const s = (customS || LETTER_PAIRS[pairIdx].s).toLowerCase();
  const t = (customT || LETTER_PAIRS[pairIdx].t).toLowerCase();
  const freqS = useMemo(() => getFreq(s), [s]);
  const freqT = useMemo(() => getFreq(t), [t]);
  const maxFreq = Math.max(...Object.values({ ...freqS, ...freqT }), 1);
  const isAnagram = useMemo(() => {
    const diff: Record<string, number> = {};
    for (const ch of s) { if (/[a-z]/.test(ch)) diff[ch] = (diff[ch] || 0) + 1; }
    for (const ch of t) { if (/[a-z]/.test(ch)) diff[ch] = (diff[ch] || 0) - 1; }
    return Object.values(diff).every((v) => v === 0);
  }, [s, t]);

  // Group tab
  const groupWords = GROUP_PRESETS[groupPreset];
  const groups = useMemo(() => groupByAnagram(groupWords), [groupWords]);

  // Animate step
  const totalChars = s.length + t.length;
  React.useEffect(() => {
    if (animStep < totalChars) {
      const t = setTimeout(() => setAnimStep((i) => i + 1), 200);
      return () => clearTimeout(t);
    }
  }, [animStep, totalChars]);

  const usedLetters = ALPHABET.filter((ch) => freqS[ch] || freqT[ch]);

  return (
    <div className="rounded-xl border border-border-subtle bg-bg-secondary p-5 space-y-4 font-mono text-sm">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-bold text-text-primary">🔡 异位词哈希分析器</h3>
          <p className="text-xs text-text-tertiary mt-0.5">字符频次直方图对比 + 分组可视化（LeetCode #242 & #49）</p>
        </div>
        <div className="flex gap-1 rounded-lg overflow-hidden border border-border-subtle">
          {(["compare", "group"] as Tab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1.5 text-xs transition-colors ${tab === t ? "bg-blue-600 text-white" : "bg-bg-tertiary text-text-secondary hover:bg-bg-secondary"}`}>
              {t === "compare" ? "🔍 两字符串对比" : "📦 分组归类"}
            </button>
          ))}
        </div>
      </div>

      {tab === "compare" ? (
        <>
          {/* 预设选择 */}
          <div className="flex gap-2 flex-wrap items-center">
            <span className="text-xs text-text-tertiary">预设：</span>
            {LETTER_PAIRS.map((p, i) => (
              <button key={i} onClick={() => { setPairIdx(i); setCustomS(""); setCustomT(""); setAnimStep(0); }}
                className={`px-2 py-1 rounded text-xs border transition-colors ${
                  pairIdx === i && !customS
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-blue-400"
                }`}>
                {p.label}
              </button>
            ))}
          </div>

          {/* 自定义输入 */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-text-tertiary mb-1 block">字符串 s（≤10字符）</label>
              <input type="text" maxLength={10} value={customS}
                onChange={(e) => { setCustomS(e.target.value.toLowerCase()); setAnimStep(0); }}
                placeholder={LETTER_PAIRS[pairIdx].s}
                className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
              />
            </div>
            <div>
              <label className="text-xs text-text-tertiary mb-1 block">字符串 t（≤10字符）</label>
              <input type="text" maxLength={10} value={customT}
                onChange={(e) => { setCustomT(e.target.value.toLowerCase()); setAnimStep(0); }}
                placeholder={LETTER_PAIRS[pairIdx].t}
                className="w-full px-2 py-1.5 rounded border border-border-subtle bg-bg-tertiary text-text-primary text-sm focus:outline-none focus:border-blue-400"
              />
            </div>
          </div>

          {/* 字符串显示 */}
          <div className="grid grid-cols-2 gap-3">
            {[{ str: s, label: "s", color: "blue" }, { str: t, label: "t", color: "orange" }].map(({ str, label, color }, strIdx) => (
              <div key={strIdx}>
                <div className={`text-xs mb-1 ${color === "blue" ? "text-blue-400" : "text-amber-400"}`}>字符串 {label}："{str}"</div>
                <div className="flex gap-1 flex-wrap">
                  {str.split("").map((ch, i) => (
                    <div key={i} className={`w-7 h-7 rounded flex items-center justify-center text-xs border font-bold transition-all ${
                      color === "blue"
                        ? "bg-blue-500/20 border-blue-500/50 text-blue-300"
                        : "bg-amber-500/20 border-amber-500/50 text-amber-300"
                    }`}>
                      {ch}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* 频次直方图 */}
          <div>
            <div className="text-xs text-text-tertiary mb-2">字符频次直方图（仅显示出现的字母）</div>
            {usedLetters.length === 0 ? (
              <div className="text-text-tertiary text-xs">请输入包含字母的字符串</div>
            ) : (
              <div className="flex gap-1 items-end" style={{ height: "100px" }}>
                {usedLetters.map((ch) => {
                  const fs = freqS[ch] || 0, ft = freqT[ch] || 0;
                  const diffCount = fs - ft;
                  return (
                    <div key={ch} className="flex flex-col items-center gap-0.5" style={{ minWidth: "28px" }}>
                      <div className="flex items-end gap-0.5" style={{ height: "80px" }}>
                        {/* s bar */}
                        <div
                          className={`w-3 rounded-t transition-all duration-300 ${fs > 0 ? "bg-blue-500" : "bg-transparent"}`}
                          style={{ height: `${(fs / maxFreq) * 76}px` }}
                          title={`s["${ch}"]=${fs}`}
                        />
                        {/* t bar */}
                        <div
                          className={`w-3 rounded-t transition-all duration-300 ${ft > 0 ? "bg-amber-500" : "bg-transparent"}`}
                          style={{ height: `${(ft / maxFreq) * 76}px` }}
                          title={`t["${ch}"]=${ft}`}
                        />
                      </div>
                      <span className={`text-[10px] font-bold ${diffCount !== 0 ? "text-red-400" : "text-text-tertiary"}`}>{ch}</span>
                      {diffCount !== 0 && (
                        <span className="text-[9px] text-red-400">{diffCount > 0 ? `+${diffCount}` : diffCount}</span>
                      )}
                    </div>
                  );
                })}
                {/* 图例 */}
                <div className="ml-3 flex flex-col justify-end gap-1 text-[10px]">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block" />s</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-500 inline-block" />t</span>
                </div>
              </div>
            )}
          </div>

          {/* 差分数组示意 */}
          <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle">
            <div className="text-xs text-text-tertiary mb-2 font-bold">差分计数法（O(n) 时间，O(1) 空间 for 26字母）</div>
            <div className="font-mono text-xs text-text-secondary space-y-1">
              <div>count = [0] × 26</div>
              <div>for ch in <span className="text-blue-400">s</span>: count[ord(ch)-97] <span className="text-green-400">+= 1</span></div>
              <div>for ch in <span className="text-amber-400">t</span>: count[ord(ch)-97] <span className="text-red-400">-= 1</span></div>
              <div>return all(c == 0 for c in count)</div>
            </div>
          </div>

          {/* 结果 */}
          <div className={`rounded-xl p-4 border-2 text-center font-bold text-lg transition-colors ${
            isAnagram ? "bg-green-500/10 border-green-500/50 text-green-300" : "bg-red-500/10 border-red-500/50 text-red-300"
          }`}>
            {isAnagram ? `✅ "${s}" 和 "${t}" 是异位词` : `❌ "${s}" 和 "${t}" 不是异位词`}
          </div>
        </>
      ) : (
        <>
          {/* Group 分组模式 */}
          <div className="flex gap-2 flex-wrap items-center">
            <span className="text-xs text-text-tertiary">预设：</span>
            {Object.keys(GROUP_PRESETS).map((p) => (
              <button key={p} onClick={() => setGroupPreset(p)}
                className={`px-2 py-1 rounded text-xs border transition-colors ${
                  groupPreset === p
                    ? "bg-purple-600 text-white border-purple-600"
                    : "bg-bg-tertiary text-text-secondary border-border-subtle hover:border-purple-400"
                }`}>
                {p}
              </button>
            ))}
          </div>

          {/* 单词列表 */}
          <div>
            <div className="text-xs text-text-tertiary mb-2">输入单词（{groupWords.length} 个）</div>
            <div className="flex gap-2 flex-wrap">
              {groupWords.map((w, i) => (
                <div key={i} className="px-2 py-1 bg-bg-tertiary rounded border border-border-subtle text-text-primary text-xs">
                  {w}
                  <span className="ml-1 text-text-tertiary">→ key: <span className="text-purple-300">{w.split("").sort().join("")}</span></span>
                </div>
              ))}
            </div>
          </div>

          {/* 分组结果可视化 */}
          <div>
            <div className="text-xs text-text-tertiary mb-2">分组结果（排序键 HashMap）</div>
            <div className="space-y-2">
              {Object.entries(groups).map(([key, words], gi) => {
                const colors = ["blue", "amber", "green", "purple", "rose"];
                const color = colors[gi % colors.length];
                const colorMap: Record<string, string> = {
                  blue: "bg-blue-500/10 border-blue-500/40 text-blue-300",
                  amber: "bg-amber-500/10 border-amber-500/40 text-amber-300",
                  green: "bg-green-500/10 border-green-500/40 text-green-300",
                  purple: "bg-purple-500/10 border-purple-500/40 text-purple-300",
                  rose: "bg-rose-500/10 border-rose-500/40 text-rose-300",
                };
                return (
                  <div key={key} className={`rounded-lg border p-3 ${colorMap[color]}`}>
                    <div className="text-xs mb-2">
                      <span className="font-bold">排序键："{key}"</span>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {words.map((w, i) => (
                        <div key={i} className="bg-bg-secondary rounded px-2 py-1 text-xs text-text-primary border border-border-subtle">
                          {w}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 算法说明 */}
          <div className="bg-bg-tertiary rounded-lg p-3 border border-border-subtle text-xs text-text-secondary space-y-1">
            <div className="font-bold text-text-primary mb-1">LeetCode #49 算法复杂度</div>
            <div>时间：O(N × K log K)，N = 单词数，K = 最长单词长度</div>
            <div>空间：O(N × K)，存储分组结果</div>
            <div className="text-text-tertiary mt-1">优化：可用 26 维频次向量作为 key，避免排序 → O(N × K)</div>
          </div>
        </>
      )}
    </div>
  );
}

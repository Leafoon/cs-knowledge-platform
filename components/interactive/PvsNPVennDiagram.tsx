"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Mode = "pneqnp" | "peqnp";
type Region = "P" | "NP" | "NPC" | "NPhard" | "UNDEC" | null;

const EXAMPLES: Record<string, { label: string; region: Region; desc: string }[]> = {
  P: [
    { label: "排序", region: "P", desc: "O(n log n) 归并排序，多项式时间确定可解" },
    { label: "最短路径", region: "P", desc: "Dijkstra / Bellman-Ford，多项式时间" },
    { label: "最大二分匹配", region: "P", desc: "Hopcroft-Karp O(E√V)" },
    { label: "2-SAT", region: "P", desc: "强连通分量 O(V+E)" },
    { label: "最小生成树", region: "P", desc: "Prim / Kruskal，多项式时间" },
  ],
  NP: [
    { label: "图同构（GI）", region: "NP", desc: "在 NP 中，但未证明是 NPC，也未在 P 中" },
    { label: "因子分解", region: "NP", desc: "RSA 密码学基础，NP 但未知是否为 NPC" },
    { label: "离散对数", region: "NP", desc: "ECC 基础，类似因子分解地位" },
  ],
  NPC: [
    { label: "SAT / 3-SAT", region: "NPC", desc: "Cook-Levin 定理证明的第一个 NPC 问题" },
    { label: "独立集（IS）", region: "NPC", desc: "3-SAT ≤p IS" },
    { label: "顶点覆盖", region: "NPC", desc: "IS ≤p VC（互补关系）" },
    { label: "旅行商（TSP 判定版）", region: "NPC", desc: "HAM-CYCLE ≤p TSP" },
    { label: "子集和问题", region: "NPC", desc: "3-SAT ≤p SUBSET-SUM" },
    { label: "哈密顿回路", region: "NPC", desc: "3-SAT ≤p HAM-CYCLE" },
    { label: "图3-着色", region: "NPC", desc: "3-SAT ≤p 3-COLORING" },
    { label: "背包问题", region: "NPC", desc: "SUBSET-SUM ≤p KNAPSACK" },
  ],
  NPhard: [
    { label: "TSP 最优化版", region: "NPhard", desc: "找最短哈密顿回路（不只是判定），NP-hard" },
    { label: "最大团优化版", region: "NPhard", desc: "找最大团的大小，NP-hard" },
    { label: "停机问题（变体）", region: "NPhard", desc: "NP-hard 且不可判定" },
  ],
  UNDEC: [
    { label: "停机问题", region: "UNDEC", desc: "不可判定，图灵（1936）证明" },
    { label: "Post 对应问题", region: "UNDEC", desc: "不可判定" },
    { label: "希尔伯特第十问题", region: "UNDEC", desc: "丢番图方程，不可判定" },
  ],
};

const REGION_STYLE: Record<string, string> = {
  P:       "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 border-indigo-300 dark:border-indigo-700",
  NP:      "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700",
  NPC:     "bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-300 border-rose-300 dark:border-rose-700",
  NPhard:  "bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300 border-orange-300 dark:border-orange-700",
  UNDEC:   "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-300 dark:border-slate-600",
};

const REGION_NAME: Record<string, string> = {
  P: "P 类", NP: "NP（疑似不在 NPC）", NPC: "NP 完全（NPC）",
  NPhard: "NP-hard（不在 NP）", UNDEC: "不可判定",
};

export function PvsNPVennDiagram() {
  const [mode, setMode] = useState<Mode>("pneqnp");
  const [selected, setSelected] = useState<Region>("NPC");

  const handleRegionClick = (r: Region) => setSelected(r);

  const isPneqNP = mode === "pneqnp";

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-blue-200 dark:border-blue-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 px-6 py-4 flex items-center justify-between">
        <div className="text-white">
          <h3 className="text-lg font-bold">P vs NP 复杂度类韦恩图</h3>
          <p className="text-sm text-indigo-100 mt-0.5">点击区域查看代表性问题</p>
        </div>
        <div className="flex bg-white/10 rounded-xl p-1 gap-1">
          {([["pneqnp", "P ≠ NP（主流假设）"], ["peqnp", "P = NP（若成立）"]] as const).map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                mode === m ? "bg-white text-indigo-700 shadow" : "text-white hover:bg-white/20"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Venn Diagram SVG */}
          <div className="flex-shrink-0 flex items-center justify-center">
            <AnimatePresence mode="wait">
              {isPneqNP ? (
                <motion.svg
                  key="pneqnp"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  viewBox="0 0 420 320"
                  className="w-full max-w-sm"
                >
                  {/* Background - all decision problems */}
                  <rect x="4" y="4" width="412" height="312" rx="16" fill="none"
                    stroke="#94a3b8" strokeDasharray="6 4" strokeWidth="1.5" />
                  <text x="14" y="21" fontSize="10" fill="#94a3b8" fontFamily="sans-serif">所有决策问题</text>

                  {/* Undecidable zone (top right corner) */}
                  <rect x="320" y="14" width="88" height="60" rx="8"
                    fill="rgba(148,163,184,0.15)" stroke="#94a3b8" strokeDasharray="4 3" strokeWidth="1" />
                  <text x="326" y="36" fontSize="8.5" fill="#94a3b8" fontFamily="sans-serif" fontWeight="600">不可判定</text>
                  <text x="326" y="50" fontSize="8" fill="#94a3b8" fontFamily="sans-serif">停机问题…</text>

                  {/* NP-hard region (large dashed, extends beyond NP) */}
                  <ellipse cx="270" cy="185" rx="140" ry="115"
                    fill="rgba(251,146,60,0.08)" stroke="#f97316" strokeDasharray="5 3" strokeWidth="2" />
                  <text x="350" y="130" fontSize="9.5" fill="#ea580c" fontFamily="sans-serif" fontWeight="700"
                    className="dark:fill-orange-400" textAnchor="middle">NP-hard</text>
                  <text x="360" y="275" fontSize="8" fill="#f97316" fontFamily="sans-serif" textAnchor="middle">(含 NPC)</text>

                  {/* NP region */}
                  <ellipse cx="195" cy="185" rx="175" ry="115"
                    fill="rgba(99,102,241,0.08)" stroke="#6366f1" strokeWidth="2.5" />
                  <text x="120" y="100" fontSize="11" fill="#4f46e5" fontFamily="sans-serif" fontWeight="700">NP</text>

                  {/* P region */}
                  <ellipse cx="120" cy="195" rx="90" ry="70"
                    fill="rgba(99,102,241,0.18)" stroke="#4f46e5" strokeWidth="2"
                    style={{ cursor: "pointer" }}
                    onClick={() => handleRegionClick("P")}
                  />
                  <text x="90" y="190" fontSize="11" fill="#3730a3" fontFamily="sans-serif" fontWeight="800"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("P")}>P</text>
                  <text x="75" y="207" fontSize="8.5" fill="#4f46e5" fontFamily="sans-serif"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("P")}>多项式时间</text>
                  <text x="88" y="220" fontSize="8.5" fill="#4f46e5" fontFamily="sans-serif"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("P")}>可解</text>

                  {/* NP but not P label (middle of NP) */}
                  <text x="218" y="163" fontSize="8.5" fill="#6366f1" fontFamily="sans-serif" textAnchor="middle"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NP")}>NP-only</text>
                  <text x="218" y="175" fontSize="8" fill="#818cf8" fontFamily="sans-serif" textAnchor="middle"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NP")}>(GI, 因子分解?)</text>

                  {/* NPC zone - intersection of NP and NP-hard */}
                  <ellipse cx="310" cy="195" rx="52" ry="50"
                    fill="rgba(244,63,94,0.20)" stroke="#f43f5e" strokeWidth="2"
                    style={{ cursor: "pointer" }}
                    onClick={() => handleRegionClick("NPC")}
                  />
                  <text x="310" y="189" fontSize="9.5" fill="#be123c" fontFamily="sans-serif" fontWeight="800"
                    textAnchor="middle" style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NPC")}>NPC</text>
                  <text x="310" y="202" fontSize="8" fill="#f43f5e" fontFamily="sans-serif" textAnchor="middle"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NPC")}>3-SAT, IS…</text>

                  {/* NP-hard outside NP label */}
                  <text x="382" y="200" fontSize="8.5" fill="#ea580c" fontFamily="sans-serif" textAnchor="middle"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NPhard")}>NP-hard</text>
                  <text x="382" y="212" fontSize="8" fill="#f97316" fontFamily="sans-serif" textAnchor="middle"
                    style={{ cursor: "pointer" }} onClick={() => handleRegionClick("NPhard")}>(not in NP)</text>

                  {/* P⊂NP annotation */}
                  <line x1="208" y1="185" x2="240" y2="185" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 2"/>
                  <text x="248" y="213" fontSize="8" fill="#94a3b8" fontFamily="sans-serif">P ⊆ NP</text>
                  <text x="248" y="224" fontSize="8" fill="#94a3b8" fontFamily="sans-serif">P≠NP 假设下</text>
                </motion.svg>
              ) : (
                <motion.svg
                  key="peqnp"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  viewBox="0 0 420 320"
                  className="w-full max-w-sm"
                >
                  {/* Background */}
                  <rect x="4" y="4" width="412" height="312" rx="16" fill="none"
                    stroke="#94a3b8" strokeDasharray="6 4" strokeWidth="1.5" />
                  <text x="14" y="21" fontSize="10" fill="#94a3b8" fontFamily="sans-serif">所有决策问题</text>

                  {/* NP-hard extends outside */}
                  <ellipse cx="210" cy="175" rx="195" ry="130"
                    fill="rgba(251,146,60,0.08)" stroke="#f97316" strokeDasharray="5 3" strokeWidth="2" />

                  {/* P = NP = NPC - one big bold circle */}
                  <ellipse cx="210" cy="175" rx="140" ry="100"
                    fill="rgba(99,102,241,0.18)" stroke="#6366f1" strokeWidth="3" />

                  {/* Labels */}
                  <text x="210" y="148" fontSize="16" fill="#3730a3" fontFamily="sans-serif" fontWeight="900"
                    textAnchor="middle">P = NP = NPC</text>
                  <text x="210" y="168" fontSize="11" fill="#6366f1" fontFamily="sans-serif" textAnchor="middle">
                    所有 NP 问题均有多项式解
                  </text>
                  <text x="210" y="185" fontSize="9" fill="#818cf8" fontFamily="sans-serif" textAnchor="middle">
                    排序 = SAT = TSP = 图同构 = …
                  </text>
                  <text x="210" y="210" fontSize="9" fill="#94a3b8" fontFamily="sans-serif" textAnchor="middle">
                    加密算法将全部失效
                  </text>
                  <text x="210" y="225" fontSize="9" fill="#94a3b8" fontFamily="sans-serif" textAnchor="middle">
                    蛋白质折叠、优化可秒算
                  </text>

                  {/* NP-hard outer label */}
                  <text x="360" y="120" fontSize="9" fill="#ea580c" fontFamily="sans-serif" textAnchor="middle">NP-hard</text>
                  <text x="360" y="133" fontSize="8.5" fill="#f97316" fontFamily="sans-serif" textAnchor="middle">(若 P=NP 则</text>
                  <text x="360" y="145" fontSize="8.5" fill="#f97316" fontFamily="sans-serif" textAnchor="middle">NPC=P=NP)</text>

                  {/* Shock badge */}
                  <rect x="300" y="230" width="108" height="54" rx="10"
                    fill="rgba(253,224,71,0.2)" stroke="#eab308" strokeWidth="1.5" />
                  <text x="354" y="252" fontSize="9" fill="#a16207" fontFamily="sans-serif" textAnchor="middle" fontWeight="700">千禧年难题（$100万）</text>
                  <text x="354" y="266" fontSize="8.5" fill="#ca8a04" fontFamily="sans-serif" textAnchor="middle">至今悬而未决</text>
                  <text x="354" y="279" fontSize="8.5" fill="#ca8a04" fontFamily="sans-serif" textAnchor="middle">主流认为 P ≠ NP</text>
                </motion.svg>
              )}
            </AnimatePresence>
          </div>

          {/* Right panel: problem examples */}
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
              {isPneqNP ? "点击韦恩图区域 · 代表性问题" : "P=NP 的惊人推论"}
            </p>

            {isPneqNP ? (
              <div className="space-y-2">
                {(["P", "NP", "NPC", "NPhard", "UNDEC"] as Region[]).filter(Boolean).map(region => (
                  <div
                    key={region!}
                    onClick={() => setSelected(region)}
                    className={`border rounded-xl cursor-pointer transition-all overflow-hidden ${
                      REGION_STYLE[region!]
                    } ${selected === region ? "shadow-md ring-2 ring-offset-1" : "opacity-80 hover:opacity-100"}`}
                  >
                    <div className="px-4 py-2.5">
                      <p className="text-xs font-bold uppercase tracking-wider mb-1">{REGION_NAME[region!]}</p>
                      <AnimatePresence>
                        {selected === region && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="space-y-1.5 overflow-hidden"
                          >
                            {EXAMPLES[region!]?.map((ex, i) => (
                              <div key={i} className="text-xs opacity-90">
                                <span className="font-semibold">{ex.label}</span>
                                <span className="opacity-70"> — {ex.desc}</span>
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {[
                  { icon: "💣", title: "密码学全面崩溃", desc: "RSA、ECC 依赖因子分解难题。若 P=NP，这些可在多项式时间内破解，互联网安全体系瓦解。" },
                  { icon: "🧬", title: "蛋白质折叠秒算", desc: "蛋白质结构预测（涉及 NPC 问题）可多项式时间求解，药物开发革命性提速。" },
                  { icon: "🚀", title: "AI 与优化飞跃", desc: "所有 NPC 优化问题（调度、路由、背包）均有精确多项式算法，运筹学彻底变革。" },
                  { icon: "🔬", title: "数学证明自动化", desc: "定理证明验证在 NP 中，若 P=NP 则可高效自动寻找数学证明。" },
                  { icon: "❓", title: "为何主流认为 P≠NP？", desc: "数十年来对于 3-SAT 等问题，无人找到多项式算法；大量密码系统的安全性依赖 P≠NP；Razborov-Rudich 等定理表明存在「自然证明障碍」。" },
                ].map((item, i) => (
                  <div key={i} className="flex gap-3 p-3 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
                    <span className="text-xl flex-shrink-0">{item.icon}</span>
                    <div>
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{item.title}</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

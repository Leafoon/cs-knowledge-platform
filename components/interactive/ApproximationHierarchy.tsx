"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Layer = "fptas" | "ptas" | "const" | "apx" | null;

interface LayerInfo {
  id: Layer & string;
  label: string;
  sublabel: string;
  definition: string;
  time: string;
  examples: { name: string; ratio: string; note: string }[];
  color: { bg: string; border: string; text: string; badge: string; dot: string };
}

const LAYERS: LayerInfo[] = [
  {
    id: "fptas",
    label: "FPTAS",
    sublabel: "全多项式时间近似方案",
    definition: "对任意 ε > 0，给出 (1+ε)-近似解，且运行时间是 n 和 1/ε 的多项式。",
    time: "poly(n, 1/ε)，如 O(n²/ε)",
    examples: [
      { name: "子集和最大化", ratio: "(1+ε)", note: "数值缩放 + DP，O(n²/ε)" },
      { name: "背包问题", ratio: "(1+ε)", note: "价值缩放，O(n²/ε)" },
      { name: "调度最大化 C_max", ratio: "(1+ε)", note: "Graham 的 FPTAS" },
    ],
    color: {
      bg: "bg-emerald-50 dark:bg-emerald-950/60",
      border: "border-emerald-400 dark:border-emerald-500",
      text: "text-emerald-800 dark:text-emerald-200",
      badge: "bg-emerald-500 text-white",
      dot: "bg-emerald-500",
    },
  },
  {
    id: "ptas",
    label: "PTAS",
    sublabel: "多项式时间近似方案",
    definition: "对任意 ε > 0，给出 (1+ε)-近似，但时间可以是 1/ε 的指数（如 O(n^{1/ε}) 或 O(2^{1/ε}⋅n)）。",
    time: "poly(n) 关于 n，可以是 1/ε 的指数",
    examples: [
      { name: "欧氏 TSP", ratio: "(1+ε)", note: "Arora PTAS，O(n log^{O(1/ε)} n)" },
      { name: "平面图最大独立集", ratio: "(1+ε)", note: "Baker 技术，O(2^{O(1/ε)} n)" },
      { name: "最大 3-SAT", ratio: "(1+ε)", note: "随机化 PTAS" },
    ],
    color: {
      bg: "bg-blue-50 dark:bg-blue-950/60",
      border: "border-blue-400 dark:border-blue-500",
      text: "text-blue-800 dark:text-blue-200",
      badge: "bg-blue-500 text-white",
      dot: "bg-blue-500",
    },
  },
  {
    id: "const",
    label: "固定常数近似",
    sublabel: "Constant-factor Approximation",
    definition: "存在固定常数 α 使得算法在多项式时间内给出 α-近似（与 ε 无关）。",
    time: "多项式时间，常数 α 固定",
    examples: [
      { name: "顶点覆盖", ratio: "2-近似", note: "最大匹配贪心，O(V+E)" },
      { name: "度量 TSP（Christofides）", ratio: "1.5-近似", note: "MST + 奇度完美匹配" },
      { name: "集合覆盖（特殊）", ratio: "2-近似", note: "LP 松弛取整" },
      { name: "MAX-CUT（随机）", ratio: "2-近似", note: "随机二着色，期望 ≥ |E|/2" },
      { name: "MAX-CUT（GW）", ratio: "0.878-近似", note: "半定规划，1995" },
    ],
    color: {
      bg: "bg-amber-50 dark:bg-amber-950/60",
      border: "border-amber-400 dark:border-amber-500",
      text: "text-amber-800 dark:text-amber-200",
      badge: "bg-amber-500 text-white",
      dot: "bg-amber-500",
    },
  },
  {
    id: "apx",
    label: "APX-hard / 无 PTAS",
    sublabel: "最难近似的问题",
    definition: "不存在 PTAS（除非 P=NP 或 Unique Games Conjecture 不成立）。即使对任意固定 ε > 0，(1+ε)-近似也是 NP-hard 的。",
    time: "无多项式时间 PTAS",
    examples: [
      { name: "一般 TSP（无 △不等式）", ratio: "无任何常数近似", note: "HAM-CYCLE 规约排除一切近似" },
      { name: "最大独立集", ratio: "无 n^{1-ε} 近似", note: "inapproximability 结果极强" },
      { name: "集合覆盖", ratio: "≥ (1-ε)ln n", note: "Dinur-Steurer，贪心已是最优" },
      { name: "图着色（一般）", ratio: "很难近似", note: "Khot 猜想相关" },
    ],
    color: {
      bg: "bg-rose-50 dark:bg-rose-950/60",
      border: "border-rose-400 dark:border-rose-500",
      text: "text-rose-800 dark:text-rose-200",
      badge: "bg-rose-500 text-white",
      dot: "bg-rose-500",
    },
  },
];

export function ApproximationHierarchy() {
  const [selected, setSelected] = useState<string>("const");

  const info = LAYERS.find((l) => l.id === selected)!;

  return (
    <div className="w-full max-w-5xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-6 py-4">
        <h3 className="text-lg font-bold text-white">近似能力层次结构（Approximation Hierarchy）</h3>
        <p className="text-sm text-violet-100 mt-0.5">点击各层了解其定义、时间复杂度与代表性问题</p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-6">
        {/* === 左：嵌套椭圆图 === */}
        <div className="lg:w-1/2 flex items-center justify-center">
          <div className="relative w-full max-w-sm" style={{ aspectRatio: "1 / 1.08" }}>
            {/* APX-hard (outermost) */}
            <button
              onClick={() => setSelected("apx")}
              className={`absolute inset-0 rounded-[50%] border-4 transition-all duration-200 flex items-start justify-center pt-3 cursor-pointer ${
                selected === "apx"
                  ? "border-rose-500 bg-rose-100 dark:bg-rose-900/60 shadow-lg shadow-rose-200 dark:shadow-rose-900"
                  : "border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-950/40 hover:bg-rose-100 dark:hover:bg-rose-900/50"
              }`}
            >
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full mt-1 ${selected === "apx" ? "bg-rose-500 text-white" : "text-rose-600 dark:text-rose-400"}`}>
                APX-hard / 无PTAS
              </span>
            </button>

            {/* Constant (second) */}
            <button
              onClick={() => setSelected("const")}
              className={`absolute rounded-[50%] border-4 transition-all duration-200 flex items-start justify-center pt-3 cursor-pointer ${
                selected === "const"
                  ? "border-amber-500 bg-amber-100 dark:bg-amber-900/60 shadow-lg shadow-amber-200 dark:shadow-amber-900"
                  : "border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 hover:bg-amber-100 dark:hover:bg-amber-900/50"
              }`}
              style={{ inset: "9% 10%" }}
            >
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full mt-1 ${selected === "const" ? "bg-amber-500 text-white" : "text-amber-600 dark:text-amber-400"}`}>
                固定常数近似
              </span>
            </button>

            {/* PTAS (third) */}
            <button
              onClick={() => setSelected("ptas")}
              className={`absolute rounded-[50%] border-4 transition-all duration-200 flex items-start justify-center pt-3 cursor-pointer ${
                selected === "ptas"
                  ? "border-blue-500 bg-blue-100 dark:bg-blue-900/60 shadow-lg shadow-blue-200 dark:shadow-blue-900"
                  : "border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-950/40 hover:bg-blue-100 dark:hover:bg-blue-900/50"
              }`}
              style={{ inset: "20% 21%" }}
            >
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full mt-1 ${selected === "ptas" ? "bg-blue-500 text-white" : "text-blue-600 dark:text-blue-400"}`}>
                PTAS
              </span>
            </button>

            {/* FPTAS (innermost) */}
            <button
              onClick={() => setSelected("fptas")}
              className={`absolute rounded-[50%] border-4 transition-all duration-200 flex items-center justify-center cursor-pointer ${
                selected === "fptas"
                  ? "border-emerald-500 bg-emerald-100 dark:bg-emerald-900/60 shadow-lg shadow-emerald-200 dark:shadow-emerald-900"
                  : "border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/40 hover:bg-emerald-100 dark:hover:bg-emerald-900/50"
              }`}
              style={{ inset: "33% 34%" }}
            >
              <span className={`text-xs font-bold ${selected === "fptas" ? "text-emerald-800 dark:text-emerald-100" : "text-emerald-600 dark:text-emerald-400"}`}>
                FPTAS
              </span>
            </button>

            {/* Containment arrows */}
            <div className="absolute bottom-1 right-2 text-xs text-slate-400 dark:text-slate-500 font-mono select-none">
              FPTAS ⊂ PTAS ⊂ const ⊂ APX
            </div>
          </div>
        </div>

        {/* === 右：选中层信息 === */}
        <div className="lg:w-1/2">
          <AnimatePresence mode="wait">
            <motion.div
              key={selected}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className={`rounded-xl border-2 p-5 ${info.color.bg} ${info.color.border}`}
            >
              {/* Title */}
              <div className="flex items-center gap-3 mb-3">
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full ${info.color.badge}`}>
                  {info.label}
                </span>
                <span className={`text-sm font-medium ${info.color.text}`}>{info.sublabel}</span>
              </div>

              {/* Definition */}
              <p className={`text-sm leading-relaxed mb-3 ${info.color.text}`}>{info.definition}</p>

              {/* Time complexity */}
              <div className={`text-xs rounded-lg px-3 py-2 mb-4 font-mono ${info.color.text} bg-white/50 dark:bg-black/20 border ${info.color.border}`}>
                ⏱ 时间复杂度：{info.time}
              </div>

              {/* Examples */}
              <div className={`text-xs font-semibold mb-2 ${info.color.text} opacity-70`}>代表性问题</div>
              <div className="space-y-2">
                {info.examples.map((ex) => (
                  <div
                    key={ex.name}
                    className="flex items-start gap-2 bg-white/60 dark:bg-black/20 rounded-lg px-3 py-2"
                  >
                    <div className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${info.color.dot}`} />
                    <div>
                      <span className={`text-xs font-semibold ${info.color.text}`}>{ex.name}</span>
                      <span className={`text-xs ml-2 px-1.5 py-0.5 rounded ${info.color.badge} opacity-80`}>{ex.ratio}</span>
                      <div className={`text-xs ${info.color.text} opacity-60 mt-0.5`}>{ex.note}</div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Quick nav buttons */}
          <div className="flex flex-wrap gap-2 mt-3">
            {LAYERS.map((l) => (
              <button
                key={l.id}
                onClick={() => setSelected(l.id)}
                className={`text-xs px-3 py-1.5 rounded-full border-2 font-semibold transition-all ${
                  selected === l.id
                    ? `${l.color.badge} border-transparent`
                    : `${l.color.text} ${l.color.border} bg-transparent hover:opacity-80`
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

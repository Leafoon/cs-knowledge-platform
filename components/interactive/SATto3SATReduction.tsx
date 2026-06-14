"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, ChevronLeft, RotateCcw } from "lucide-react";

type CaseId = "k1" | "k2" | "k3" | "kn";

interface Literal {
  label: string;
  isAux?: boolean;
  negated?: boolean;
}

interface ClauseStep {
  title: string;
  desc: string;
  original: Literal[];
  result: Literal[][];
  auxVars: string[];
}

const CASES: Record<CaseId, { label: string; subtitle: string; steps: ClauseStep[] }> = {
  k1: {
    label: "k = 1", subtitle: "单文字子句",
    steps: [
      {
        title: "原始子句（1 个文字）",
        desc: "只有一个文字 l₁，它必须为真，否则整个公式为假。",
        original: [{ label: "l₁" }],
        result: [[{ label: "l₁" }]],
        auxVars: [],
      },
      {
        title: "引入两个辅助变量 y₁, y₂",
        desc: "需要将 1-文字子句填充到 3 个文字，引入辅助变量 y₁, y₂（fresh variables）。",
        original: [{ label: "l₁" }],
        result: [[{ label: "l₁" }]],
        auxVars: ["y₁", "y₂"],
      },
      {
        title: "展开为 4 个 3-文字子句",
        desc: "用 y₁, y₂ 的四种真值组合构造 4 个子句，保证 l₁=true 时均可满足，l₁=false 时无法同时满足。",
        original: [{ label: "l₁" }],
        result: [
          [{ label: "l₁" }, { label: "y₁", isAux: true }, { label: "y₂", isAux: true }],
          [{ label: "l₁" }, { label: "y₁", isAux: true }, { label: "¬y₂", isAux: true, negated: true }],
          [{ label: "l₁" }, { label: "¬y₁", isAux: true, negated: true }, { label: "y₂", isAux: true }],
          [{ label: "l₁" }, { label: "¬y₁", isAux: true, negated: true }, { label: "¬y₂", isAux: true, negated: true }],
        ],
        auxVars: ["y₁", "y₂"],
      },
      {
        title: "正确性验证",
        desc: "若 l₁=true：4 个子句均满足（第一个文字已真）✓\n若 l₁=false：4 个子句中第一个文字均为假，需要 y₁,y₂ 同时满足所有组合，矛盾 ✗",
        original: [{ label: "l₁" }],
        result: [
          [{ label: "l₁" }, { label: "y₁", isAux: true }, { label: "y₂", isAux: true }],
          [{ label: "l₁" }, { label: "y₁", isAux: true }, { label: "¬y₂", isAux: true, negated: true }],
          [{ label: "l₁" }, { label: "¬y₁", isAux: true, negated: true }, { label: "y₂", isAux: true }],
          [{ label: "l₁" }, { label: "¬y₁", isAux: true, negated: true }, { label: "¬y₂", isAux: true, negated: true }],
        ],
        auxVars: ["y₁", "y₂"],
      },
    ],
  },
  k2: {
    label: "k = 2", subtitle: "两文字子句",
    steps: [
      {
        title: "原始子句（2 个文字）",
        desc: "子句 (l₁ ∨ l₂)，需要至少一个文字为真。",
        original: [{ label: "l₁" }, { label: "l₂" }],
        result: [[{ label: "l₁" }, { label: "l₂" }]],
        auxVars: [],
      },
      {
        title: "引入 1 个辅助变量 y₁",
        desc: "引入一个 fresh 变量 y₁，将子句填充至 3 个文字。",
        original: [{ label: "l₁" }, { label: "l₂" }],
        result: [[{ label: "l₁" }, { label: "l₂" }]],
        auxVars: ["y₁"],
      },
      {
        title: "展开为 2 个 3-文字子句",
        desc: "y₁ 取 true 或 false 均可，因此两个子句的析取等价于原始 (l₁ ∨ l₂)。",
        original: [{ label: "l₁" }, { label: "l₂" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "y₁", isAux: true }],
          [{ label: "l₁" }, { label: "l₂" }, { label: "¬y₁", isAux: true, negated: true }],
        ],
        auxVars: ["y₁"],
      },
      {
        title: "正确性验证",
        desc: "若 l₁ ∨ l₂=true：无论 y₁ 取何值，两个子句均满足 ✓\n若 l₁=l₂=false：两个子句变成 (y₁) ∧ (¬y₁)，矛盾，无法同时满足 ✗",
        original: [{ label: "l₁" }, { label: "l₂" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "y₁", isAux: true }],
          [{ label: "l₁" }, { label: "l₂" }, { label: "¬y₁", isAux: true, negated: true }],
        ],
        auxVars: ["y₁"],
      },
    ],
  },
  k3: {
    label: "k = 3", subtitle: "三文字子句（已是 3-SAT）",
    steps: [
      {
        title: "原始子句（恰好 3 个文字）",
        desc: "子句已经是 3-文字形式，无需变换，直接保留。",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }],
        result: [[{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }]],
        auxVars: [],
      },
      {
        title: "无需变换，直接保留",
        desc: "k=3 的子句本身就满足 3-SAT 格式要求，规约函数对其原样输出。",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }],
        result: [[{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }]],
        auxVars: [],
      },
    ],
  },
  kn: {
    label: "k ≥ 4", subtitle: "长子句 — 递归链式分裂",
    steps: [
      {
        title: "原始子句（k≥4 个文字）",
        desc: "以 k=5 为例：(l₁ ∨ l₂ ∨ l₃ ∨ l₄ ∨ l₅)，必须拆分为3-文字子句链。",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [[{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }]],
        auxVars: [],
      },
      {
        title: "引入 k-3=2 个辅助变量",
        desc: "k=5 时，引入 z₁, z₂ 作为链式传递变量（k 个文字需要 k-3 个辅助变量）。",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [[{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }]],
        auxVars: ["z₁", "z₂"],
      },
      {
        title: "链式分裂：第 1 个子句",
        desc: "提取前两个文字，引入辅助变量 z₁ 作为「传递管道」：",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "z₁", isAux: true }],
        ],
        auxVars: ["z₁", "z₂"],
      },
      {
        title: "链式分裂：第 2 个子句（中间链）",
        desc: "¬z₁ 与下两个原始文字，再引出 z₂：",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "z₁", isAux: true }],
          [{ label: "¬z₁", isAux: true, negated: true }, { label: "l₃" }, { label: "z₂", isAux: true }],
        ],
        auxVars: ["z₁", "z₂"],
      },
      {
        title: "链式分裂：最后一个子句（收尾）",
        desc: "最后子句包含 ¬z_{k-3} 和最后两个原始文字，链式传递完成。",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "z₁", isAux: true }],
          [{ label: "¬z₁", isAux: true, negated: true }, { label: "l₃" }, { label: "z₂", isAux: true }],
          [{ label: "¬z₂", isAux: true, negated: true }, { label: "l₄" }, { label: "l₅" }],
        ],
        auxVars: ["z₁", "z₂"],
      },
      {
        title: "正确性验证",
        desc: "若原式可满足：某 lᵢ=true，在其所在子句令所有 z 适当取值，使链上所有子句满足 ✓\n若所有 lᵢ=false：无论 z 怎么取，链上存在某个子句三个文字全为 false ✗",
        original: [{ label: "l₁" }, { label: "l₂" }, { label: "l₃" }, { label: "l₄" }, { label: "l₅" }],
        result: [
          [{ label: "l₁" }, { label: "l₂" }, { label: "z₁", isAux: true }],
          [{ label: "¬z₁", isAux: true, negated: true }, { label: "l₃" }, { label: "z₂", isAux: true }],
          [{ label: "¬z₂", isAux: true, negated: true }, { label: "l₄" }, { label: "l₅" }],
        ],
        auxVars: ["z₁", "z₂"],
      },
    ],
  },
};

function LiteralPill({ lit, bright = false }: { lit: Literal; bright?: boolean }) {
  return (
    <motion.span
      layout
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`inline-flex items-center px-2.5 py-1 rounded-lg text-sm font-mono font-bold border ${
        lit.isAux
          ? "bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-600"
          : "bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 border-slate-300 dark:border-slate-600"
      }`}
    >
      {lit.label}
    </motion.span>
  );
}

function Clause({ lits, idx }: { lits: Literal[]; idx: number }) {
  return (
    <motion.div
      key={idx}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.08 }}
      className="flex items-center gap-2 flex-wrap"
    >
      <span className="text-slate-400 text-sm mr-1">C{idx + 1}:</span>
      <span className="text-slate-400">(</span>
      {lits.map((lit, j) => (
        <span key={j} className="flex items-center gap-1.5">
          <LiteralPill lit={lit} />
          {j < lits.length - 1 && <span className="text-slate-500 font-bold">∨</span>}
        </span>
      ))}
      <span className="text-slate-400">)</span>
    </motion.div>
  );
}

export function SATto3SATReduction() {
  const [caseId, setCaseId] = useState<CaseId>("kn");
  const [stepIdx, setStepIdx] = useState(0);

  const caseData = CASES[caseId];
  const currentStep = caseData.steps[Math.min(stepIdx, caseData.steps.length - 1)];
  const maxStep = caseData.steps.length - 1;

  const changeCase = (id: CaseId) => { setCaseId(id); setStepIdx(0); };

  return (
    <div className="w-full max-w-3xl mx-auto my-6 rounded-2xl overflow-hidden border border-amber-200 dark:border-amber-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-500 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">SAT → 3-SAT 规约演示</h3>
        <p className="text-sm text-amber-100 mt-0.5">四种子句情形的转换，每步观察辅助变量如何被引入</p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6 space-y-5">
        {/* Case selector */}
        <div className="grid grid-cols-4 gap-2">
          {(Object.keys(CASES) as CaseId[]).map(id => (
            <button
              key={id}
              onClick={() => changeCase(id)}
              className={`p-3 rounded-xl border text-center transition-all ${
                caseId === id
                  ? "bg-amber-50 dark:bg-amber-900/40 border-amber-400 dark:border-amber-600 shadow"
                  : "border-slate-200 dark:border-slate-700 hover:border-amber-300 dark:hover:border-amber-700"
              }`}
            >
              <div className={`text-sm font-bold ${caseId === id ? "text-amber-700 dark:text-amber-300" : "text-slate-700 dark:text-slate-300"}`}>
                {CASES[id].label}
              </div>
              <div className={`text-xs mt-0.5 ${caseId === id ? "text-amber-600 dark:text-amber-400" : "text-slate-500"}`}>
                {CASES[id].subtitle}
              </div>
            </button>
          ))}
        </div>

        {/* Step content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={`${caseId}-${stepIdx}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-4"
          >
            {/* Step title */}
            <div className="flex items-center gap-3">
              <div className="flex-shrink-0 w-7 h-7 rounded-full bg-amber-500 text-white text-xs font-bold flex items-center justify-center">
                {stepIdx + 1}
              </div>
              <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100">{currentStep.title}</h4>
            </div>

            <p className="text-sm text-slate-600 dark:text-slate-400 whitespace-pre-line">{currentStep.desc}</p>

            {/* Aux vars badge */}
            {currentStep.auxVars.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">引入辅助变量：</span>
                {currentStep.auxVars.map(v => (
                  <span key={v} className="px-2.5 py-1 text-xs rounded-lg bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-600 font-mono font-bold">
                    {v}
                  </span>
                ))}
              </div>
            )}

            {/* Original → Result */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Original */}
              <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
                <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">原始子句</p>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-slate-400">(</span>
                  {currentStep.original.map((lit, j) => (
                    <span key={j} className="flex items-center gap-1.5">
                      <LiteralPill lit={lit} />
                      {j < currentStep.original.length - 1 && <span className="text-slate-500 font-bold">∨</span>}
                    </span>
                  ))}
                  <span className="text-slate-400">)</span>
                </div>
              </div>

              {/* Arrow + Result */}
              <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
                <p className="text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-3">
                  3-SAT 等价形式（{currentStep.result.length} 个子句）
                </p>
                <div className="space-y-2">
                  {currentStep.result.length === 1 && currentStep.result[0].length <= 3 ? (
                    <Clause lits={currentStep.result[0]} idx={0} />
                  ) : currentStep.result.length === 1 && currentStep.result[0].length > 3 ? (
                    <div className="text-sm text-slate-500 italic">（待拆分…）</div>
                  ) : (
                    currentStep.result.map((clause, i) => (
                      <Clause key={i} lits={clause} idx={i} />
                    ))
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Step controls */}
        <div className="flex items-center gap-3 pt-1">
          <button
            onClick={() => setStepIdx(s => Math.max(0, s - 1))}
            disabled={stepIdx === 0}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all"
          >
            <ChevronLeft className="w-4 h-4" /> 上一步
          </button>

          <div className="flex gap-1.5 flex-1 justify-center">
            {caseData.steps.map((_, i) => (
              <button
                key={i}
                onClick={() => setStepIdx(i)}
                className={`w-2 h-2 rounded-full transition-all ${
                  i === stepIdx ? "bg-amber-500 w-4" : "bg-slate-300 dark:bg-slate-600 hover:bg-amber-300"
                }`}
              />
            ))}
          </div>

          {stepIdx < maxStep ? (
            <button
              onClick={() => setStepIdx(s => Math.min(maxStep, s + 1))}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium transition-all"
            >
              下一步 <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={() => setStepIdx(0)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all"
            >
              <RotateCcw className="w-4 h-4" /> 重置
            </button>
          )}
        </div>

        {/* Complexity summary */}
        <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 text-xs text-slate-600 dark:text-slate-400">
          <span className="font-semibold">规约复杂度：</span>多项式时间 O(m·k²)，其中 m 是子句数，k 是每个子句的最大文字数。规约后 3-SAT 公式大小至多是原始 SAT 公式的多项式倍。
        </div>
      </div>
    </div>
  );
}

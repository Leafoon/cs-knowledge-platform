"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

type Assignment = "TT" | "TF" | "FT" | "FF";

interface TableRow {
  name: string;
  type: "var" | "slack";
  cols: number[]; // [x₁, x₂, C₁, C₂]
  selectedIn: Assignment[];
}

// Formula: C₁=(x₁∨¬x₂), C₂=(¬x₁∨x₂)
// Target: [1, 1, 3, 3]
const ROWS: TableRow[] = [
  { name: "v₁", type: "var", cols: [1, 0, 1, 0], selectedIn: ["TT", "TF"] },
  { name: "v̄₁", type: "var", cols: [1, 0, 0, 1], selectedIn: ["FT", "FF"] },
  { name: "v₂", type: "var", cols: [0, 1, 0, 1], selectedIn: ["TT", "FT"] },
  { name: "v̄₂", type: "var", cols: [0, 1, 1, 0], selectedIn: ["TF", "FF"] },
  { name: "s₁",  type: "slack", cols: [0, 0, 1, 0], selectedIn: ["TT", "TF", "FF"] },
  { name: "s̄₁", type: "slack", cols: [0, 0, 1, 0], selectedIn: ["TT", "FF"] },
  { name: "s₂",  type: "slack", cols: [0, 0, 0, 1], selectedIn: ["TT", "FT", "FF"] },
  { name: "s̄₂", type: "slack", cols: [0, 0, 0, 1], selectedIn: ["TT", "FF"] },
];

const TARGET = [1, 1, 3, 3];
const COL_LABELS = ["x₁", "x₂", "C₁", "C₂"];

const ASSIGNMENTS: { id: Assignment; label: string; x1: boolean; x2: boolean }[] = [
  { id: "TT", label: "x₁=T, x₂=T", x1: true,  x2: true  },
  { id: "TF", label: "x₁=T, x₂=F", x1: true,  x2: false },
  { id: "FT", label: "x₁=F, x₂=T", x1: false, x2: true  },
  { id: "FF", label: "x₁=F, x₂=F", x1: false, x2: false },
];

function evalFormula(x1: boolean, x2: boolean) {
  const c1 = x1 || !x2;  // C₁ = x₁ ∨ ¬x₂
  const c2 = !x1 || x2;  // C₂ = ¬x₁ ∨ x₂
  return { c1, c2, sat: c1 && c2 };
}

export function SubsetSumReductionDemo() {
  const [assignment, setAssignment] = useState<Assignment>("TT");

  const asgn = ASSIGNMENTS.find(a => a.id === assignment)!;
  const ev = evalFormula(asgn.x1, asgn.x2);

  // compute column sums for selected rows
  const selected = ROWS.filter(r => r.selectedIn.includes(assignment));
  const colSums = [0, 1, 2, 3].map(ci => selected.reduce((acc, r) => acc + r.cols[ci], 0));
  const allMatch = colSums.every((s, i) => s === TARGET[i]);

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-cyan-200 dark:border-cyan-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-600 to-teal-600 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">3-SAT → Subset Sum 规约演示</h3>
        <p className="text-sm text-cyan-100 mt-0.5">
          公式：<span className="font-mono bg-white/15 px-1.5 py-0.5 rounded">C₁=(x₁∨¬x₂) ∧ C₂=(¬x₁∨x₂)</span>，
          选择真值赋值，观察对应的整数子集选取
        </p>
      </div>

      <div className="bg-white dark:bg-slate-900 p-6 space-y-5">
        {/* Assignment selector */}
        <div>
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">选择真值赋值</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {ASSIGNMENTS.map(a => {
              const e = evalFormula(a.x1, a.x2);
              return (
                <button
                  key={a.id}
                  onClick={() => setAssignment(a.id)}
                  className={`p-3 rounded-xl border text-center transition-all ${
                    assignment === a.id
                      ? e.sat
                        ? "bg-emerald-50 dark:bg-emerald-900/40 border-emerald-400 shadow"
                        : "bg-red-50 dark:bg-red-900/40 border-red-400 shadow"
                      : "border-slate-200 dark:border-slate-700 hover:border-cyan-300 dark:hover:border-cyan-700"
                  }`}
                >
                  <div className={`text-sm font-bold font-mono ${assignment === a.id ? (e.sat ? "text-emerald-700 dark:text-emerald-300" : "text-red-700 dark:text-red-300") : "text-slate-700 dark:text-slate-300"}`}>
                    {a.label}
                  </div>
                  <div className="flex items-center justify-center gap-1 mt-1">
                    {e.c1 ? <span className="text-xs text-emerald-600 dark:text-emerald-400">C₁✓</span> : <span className="text-xs text-red-500">C₁✗</span>}
                    {e.c2 ? <span className="text-xs text-emerald-600 dark:text-emerald-400">C₂✓</span> : <span className="text-xs text-red-500">C₂✗</span>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Encoding Table */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">整数编码表</p>
            <div className="flex gap-3 text-xs">
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-blue-400 inline-block"/><span>蓝色：变量行</span></span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-amber-400 inline-block"/><span>橙色：松弛行</span></span>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-800">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 dark:text-slate-300 w-32">行</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 dark:text-slate-300">类型</th>
                  {COL_LABELS.map((lbl, i) => (
                    <th key={i} className={`px-4 py-3 text-center text-xs font-semibold ${i >= 2 ? "text-teal-600 dark:text-teal-400" : "text-indigo-600 dark:text-indigo-400"}`}>
                      {lbl}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-center text-xs font-semibold text-slate-500 dark:text-slate-400">选取？</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {ROWS.map((row, ri) => {
                  const isSel = row.selectedIn.includes(assignment);
                  return (
                    <motion.tr
                      key={ri}
                      animate={{ backgroundColor: isSel ? (row.type === "var" ? "rgba(99,102,241,0.08)" : "rgba(245,158,11,0.08)") : "transparent" }}
                      transition={{ duration: 0.25 }}
                      className={`transition-colors ${isSel ? "" : "opacity-40"}`}
                    >
                      <td className="px-4 py-2.5">
                        <span className={`font-mono font-bold text-sm ${isSel ? (row.type === "var" ? "text-indigo-700 dark:text-indigo-300" : "text-amber-700 dark:text-amber-300") : "text-slate-500"}`}>
                          {row.name}
                        </span>
                        <span className="ml-2 text-xs text-slate-400">
                          {row.type === "var"
                            ? (row.name === "v₁" ? "(x₁=T)" : row.name === "v̄₁" ? "(x₁=F)" : row.name === "v₂" ? "(x₂=T)" : "(x₂=F)")
                            : "（松弛）"}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${row.type === "var" ? "bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400" : "bg-amber-100 dark:bg-amber-900/40 text-amber-600 dark:text-amber-400"}`}>
                          {row.type === "var" ? "变量" : "松弛"}
                        </span>
                      </td>
                      {row.cols.map((val, ci) => (
                        <td key={ci} className="px-4 py-2.5 text-center">
                          {val > 0 ? (
                            <span className={`font-bold font-mono ${isSel ? (ci >= 2 ? "text-teal-700 dark:text-teal-300" : "text-indigo-700 dark:text-indigo-300") : "text-slate-400"}`}>
                              {val}
                            </span>
                          ) : (
                            <span className="text-slate-300 dark:text-slate-600 font-mono">0</span>
                          )}
                        </td>
                      ))}
                      <td className="px-4 py-2.5 text-center">
                        <AnimatePresence mode="wait">
                          {isSel ? (
                            <motion.span key="yes" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}
                              className="inline-flex items-center justify-center w-6 h-6 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400 rounded-full text-xs font-bold">
                              ✓
                            </motion.span>
                          ) : (
                            <motion.span key="no" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}
                              className="inline-flex items-center justify-center w-6 h-6 bg-slate-100 dark:bg-slate-800 text-slate-400 rounded-full text-xs">
                              —
                            </motion.span>
                          )}
                        </AnimatePresence>
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>

              {/* Column sums */}
              <tfoot>
                <tr className="border-t-2 border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-800">
                  <td colSpan={2} className="px-4 py-3 text-xs font-bold text-slate-700 dark:text-slate-200">Σ 列和</td>
                  {colSums.map((s, i) => (
                    <td key={i} className="px-4 py-3 text-center">
                      <motion.span
                        key={`${assignment}-${i}`}
                        initial={{ scale: 0.5 }}
                        animate={{ scale: 1 }}
                        className={`text-sm font-bold font-mono ${s === TARGET[i] ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}
                      >
                        {s}
                      </motion.span>
                    </td>
                  ))}
                  <td />
                </tr>
                <tr className="bg-slate-100 dark:bg-slate-800/50">
                  <td colSpan={2} className="px-4 py-2.5 text-xs font-bold text-slate-500 dark:text-slate-400">目标 T</td>
                  {TARGET.map((t, i) => (
                    <td key={i} className="px-4 py-2.5 text-center">
                      <span className={`text-sm font-mono font-bold ${i >= 2 ? "text-teal-600 dark:text-teal-400" : "text-indigo-600 dark:text-indigo-400"}`}>{t}</span>
                    </td>
                  ))}
                  <td />
                </tr>
              </tfoot>
            </table>
          </div>
        </div>

        {/* Result */}
        <AnimatePresence mode="wait">
          <motion.div
            key={assignment}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className={`flex items-center gap-4 p-4 rounded-xl border ${
              allMatch
                ? "bg-emerald-50 dark:bg-emerald-900/30 border-emerald-300 dark:border-emerald-700"
                : "bg-red-50 dark:bg-red-900/30 border-red-300 dark:border-red-700"
            }`}
          >
            {allMatch
              ? <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
              : <XCircle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0" />}
            <div>
              <p className={`text-sm font-bold ${allMatch ? "text-emerald-700 dark:text-emerald-300" : "text-red-700 dark:text-red-300"}`}>
                {allMatch ? "找到合法子集！列和 = 目标 T → 原公式可满足" : "无合法子集——列和无法达到目标 T → 对应子句不满足"}
              </p>
              <p className={`text-xs mt-0.5 ${allMatch ? "text-emerald-600 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
                {allMatch
                  ? `赋值 ${asgn.label} 满足公式：C₁=${ev.c1 ? "✓" : "✗"}, C₂=${ev.c2 ? "✓" : "✗"}`
                  : `赋值 ${asgn.label} 不满足公式：C₁=${ev.c1 ? "✓" : "✗"}, C₂=${ev.c2 ? "✓" : "✗"}，对应列无法凑够目标值 3`}
              </p>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Key insight */}
        <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded-xl border border-cyan-200 dark:border-cyan-700 text-xs text-cyan-700 dark:text-cyan-300 space-y-1.5">
          <p className="font-bold text-sm mb-2">🔑 规约的核心编码精髓</p>
          <p><span className="font-semibold">变量列（x₁, x₂）：</span>每列目标为 1——确保对每个变量，恰好选 vᵢ 或 v̄ᵢ 之一（=选真或假，不能重复）。</p>
          <p><span className="font-semibold">子句列（C₁, C₂）：</span>每列目标为 3——字面量满足则贡献 1，松弛变量补足；若某子句完全不满足（贡献 0），松弛最多补 2 仍不够 3。</p>
          <p><span className="font-semibold">规约复杂度：</span>n 个变量 + m 个子句 → 2n+2m 个整数，每个整数至多 n+m 位 → 多项式规约 O(nm)。</p>
        </div>
      </div>
    </div>
  );
}

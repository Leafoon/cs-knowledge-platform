"use client";

import { useState } from "react";

const passes = [
  { name: "Constant Folding", type: "优化" },
  { name: "Dead Code Elimination", type: "优化" },
  { name: "Operator Fusion", type: "优化" },
  { name: "Layout Transform", type: "变换" },
  { name: "InferType", type: "分析" },
  { name: "ToANormalForm", type: "变换" },
];

const irBefore = `fn (%x: Tensor[(128, 128), float32],
    %y: Tensor[(128, 128), float32]) {
  %0 = add(%x, %y)
  %1 = multiply(%0, 0.5f)
  %2 = add(%1, 0.0f)   // 冗余操作
  %3 = nn.relu(%2)
  %3
}`;

const irAfter = `fn (%x: Tensor[(128, 128), float32],
    %y: Tensor[(128, 128), float32]) {
  %0 = add(%x, %y)
  %1 = multiply(%0, 0.5f)
  %2 = nn.relu(%1)     // 直接接 relu
  %2
}`;

const passDetails: Record<string, string> = {
  "Constant Folding": "常量折叠: 将编译时可计算的表达式提前计算\n0.0f + x → x\n减少了运行时加法操作",
  "Dead Code Elimination": "死代码消除: 移除不影响输出的计算\nadd(%1, 0.0f) 是冗余的\n被直接消除",
  "Operator Fusion": "算子融合: 将连续的小算子合并为大算子\nmultiply + relu → fused_mul_relu\n减少内存访问和 kernel launch",
  "Layout Transform": "布局变换: 转换数据布局以匹配硬件\nNCHW → NCHWc (channel last blocked)\n提升向量化效率",
  "InferType": "类型推断: 推导每个节点的类型\n输入 (128, 128) float32\n所有中间值类型一致",
  "ToANormalForm": "A-Normal Form: 使每个子表达式绑定到变量\n嵌套表达式 → 扁平化\n便于后续分析和变换",
};

export function PassDebugger() {
  const [selectedPass, setSelectedPass] = useState("Operator Fusion");
  const [showAfter, setShowAfter] = useState(false);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">Pass 调试器</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">IR_before → Pass → IR_after 逐步检查</p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700 md:col-span-1">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">可用 Pass</h4>
          <div className="space-y-1.5">
            {passes.map((p) => (
              <button
                key={p.name}
                onClick={() => setSelectedPass(p.name)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-all ${
                  selectedPass === p.name
                    ? "bg-indigo-100 dark:bg-indigo-900/40 border border-indigo-300 dark:border-indigo-600"
                    : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
                }`}
              >
                <span className="text-xs font-medium text-slate-700 dark:text-slate-200">{p.name}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                  p.type === "优化" ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-400" :
                  p.type === "变换" ? "bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400" :
                  "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400"
                }`}>
                  {p.type}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="md:col-span-2 space-y-4">
          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200">
                Pass 详情: {selectedPass}
              </h4>
              <button
                onClick={() => setShowAfter(!showAfter)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  showAfter
                    ? "bg-emerald-500 text-white"
                    : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
                }`}
              >
                {showAfter ? "IR After" : "IR Before"}
              </button>
            </div>
            <div className="bg-slate-900 dark:bg-slate-950 rounded-lg p-3 font-mono text-xs leading-relaxed text-green-400">
              <pre>{showAfter ? irAfter : irBefore}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">Pass 解释</h4>
            <p className="text-xs text-slate-600 dark:text-slate-300 whitespace-pre-wrap leading-relaxed">
              {passDetails[selectedPass]}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

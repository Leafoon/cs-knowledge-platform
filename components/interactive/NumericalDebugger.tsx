"use client";

import { useState } from "react";

const detectors = [
  {
    id: "nan",
    name: "NaN 检测",
    icon: "🔢",
    color: "from-blue-500 to-indigo-500",
    symptom: "输出全为 NaN",
    causes: ["除以零 (0/0)", "ln(0) 或 sqrt(-1)", "梯度爆炸", "初始化不当"],
    fix: "启用 NaN 检查点:\ntvm.transform.PassContext(\n  config={\"tir.debug_nan\": True}\n)\n# 运行时自动检测 NaN 出现位置",
    example: `# 检测 NaN 传播
A = tvm.nd.array([float('nan'), 1.0])
B = tvm.nd.empty((2,))
f(A, B)
# → 报错: NaN detected at B[0]`,
  },
  {
    id: "inf",
    name: "Inf 检测",
    icon: "♾️",
    color: "from-indigo-500 to-purple-500",
    symptom: "输出包含 Inf/-Inf",
    causes: ["指数溢出 exp(1000)", "数值范围过大", "未归一化的输入", "学习率过高"],
    fix: "启用 Inf 检查:\ntvm.transform.PassContext(\n  config={\"tir.debug_inf\": True}\n)\n# 溢出时立即报错并定位",
    example: `# 检测溢出
x = tvm.nd.array([1000.0])
y = tvm.nd.empty((1,))
exp_func(x, y)
# → 报错: Inf at y[0], caused by exp(1000)`,
  },
  {
    id: "overflow",
    name: "Overflow 检测",
    icon: "💥",
    color: "from-purple-500 to-pink-500",
    symptom: "整数溢出导致错误结果",
    causes: ["int32 累加超出范围", "索引计算溢出", "量化值超界", "batch_size 过大"],
    fix: "启用溢出检查:\ntvm.transform.PassContext(\n  config={\"tir.debug_overflow\": True}\n)\n# 检查每次算术操作",
    example: `# int32 溢出检测
# 2^31 - 1 + 1 = -2^31
# 检测到溢出:
# overflow detected at add(vi, 1)
# result: -2147483648`,
  },
];

export function NumericalDebugger() {
  const [active, setActive] = useState("nan");

  const detector = detectors.find((d) => d.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">数值调试器</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">NaN / Inf / Overflow 三种数值异常检测</p>

      <div className="grid grid-cols-3 gap-3 mb-6">
        {detectors.map((d) => (
          <button
            key={d.id}
            onClick={() => setActive(d.id)}
            className={`p-4 rounded-xl border-2 transition-all duration-300 text-center ${
              active === d.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span className="text-2xl">{d.icon}</span>
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200 mt-1">{d.name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{d.symptom}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">常见原因</h4>
          <ul className="space-y-1.5 mb-4">
            {detector.causes.map((c, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                {c}
              </li>
            ))}
          </ul>
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">修复方案</h4>
          <div className="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 font-mono text-xs text-slate-700 dark:text-slate-300 whitespace-pre-wrap">
            {detector.fix}
          </div>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <div className="text-slate-500 mb-2"># 检测示例</div>
          <pre>{detector.example}</pre>
        </div>
      </div>
    </div>
  );
}

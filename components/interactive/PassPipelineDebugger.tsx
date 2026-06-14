"use client";

import { useState } from "react";

const pipeline = [
  { id: 1, name: "InferType", type: "分析", desc: "推断所有节点的类型和 shape", effect: "标注每个张量的 dtype 和 shape" },
  { id: 2, name: "SimplifyExpr", type: "简化", desc: "简化冗余表达式", effect: "消除 x*1, x+0 等恒等操作" },
  { id: 3, name: "FuseOps", type: "融合", desc: "融合可合并的算子", effect: "conv+bn+relu → fused_conv_bn_relu" },
  { id: 4, name: "ToANormal", type: "变换", desc: "转换为 A-Normal Form", effect: "嵌套表达式扁平化" },
  { id: 5, name: "LowerToTIR", type: "降级", desc: "从 Relay 降到 TIR", effect: "高层算子 → 底层循环" },
  { id: 6, name: "OptimizeTIR", type: "优化", desc: "TIR 级别优化", effect: "循环展开、向量化、tiling" },
];

const typeColors: Record<string, string> = {
  "分析": "bg-blue-500",
  "简化": "bg-emerald-500",
  "融合": "bg-indigo-500",
  "变换": "bg-amber-500",
  "降级": "bg-purple-500",
  "优化": "bg-pink-500",
};

export function PassPipelineDebugger() {
  const [active, setActive] = useState(0);
  const [completed, setCompleted] = useState<number[]>([]);

  const handleStep = (idx: number) => {
    setActive(idx);
    setCompleted((prev) => {
      const next = [...new Set([...prev, ...Array.from({ length: idx + 1 }, (_, i) => i)])];
      return next;
    });
  };

  const step = pipeline[active];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">Pass Pipeline 调试</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">6 个 Pass 的链式变换，逐步检查 IR 变化</p>

      <div className="flex items-center justify-center gap-1 mb-6 flex-wrap">
        {pipeline.map((p, i) => (
          <div key={p.id} className="flex items-center">
            <button
              onClick={() => handleStep(i)}
              className={`flex flex-col items-center px-3 py-2 rounded-xl border-2 transition-all duration-300 min-w-[90px] ${
                active === i
                  ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                  : completed.includes(i)
                  ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
              }`}
            >
              <span className="text-xs font-bold text-slate-700 dark:text-slate-200">P{p.id}</span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">{p.name}</span>
              {completed.includes(i) && i !== active && (
                <span className="text-emerald-500 text-[10px]">✓</span>
              )}
            </button>
            {i < pipeline.length - 1 && (
              <span className={`mx-0.5 text-sm ${completed[i] ? "text-emerald-400" : "text-indigo-300"}`}>→</span>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <span className={`w-2 h-2 rounded-full ${typeColors[step.type]}`} />
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">{step.type}</span>
          </div>
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">{step.name}</h4>
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-3">{step.desc}</p>
          <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-3">
            <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400 mb-1">效果</div>
            <div className="text-xs text-slate-600 dark:text-slate-300">{step.effect}</div>
          </div>
        </div>

        <div className="md:col-span-2 space-y-4">
          <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
            <div className="text-slate-500 mb-2"># Pipeline 输出 (到 Pass {step.id} 为止)</div>
            <pre>{`# Pass Pipeline: ${pipeline.slice(0, active + 1).map((p) => p.name).join(" → ")}
\n# 阶段 {step.id}: {step.name}\n# {step.desc}\n\n$ tvm.run_passes(mod, passes=[${pipeline.slice(0, active + 1).map((p) => `"${p.name}"`).join(", ")}])\n\n# IR 状态: {active === 0 ? "已标注类型" : active < 3 ? "已简化" : active < 4 ? "已融合" : active < 5 ? "已降级到TIR" : "已优化"}`}</pre>
          </div>

          <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">Pipeline 进度</h4>
            <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${((active + 1) / pipeline.length) * 100}%` }}
              />
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {active + 1} / {pipeline.length} Passes 完成
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

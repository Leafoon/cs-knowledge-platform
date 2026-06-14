"use client";

export function PipelineFlowDiagram() {
  const steps = [
    { label: "PyTorch / ONNX", sub: "前端导入", icon: "🔥" },
    { label: "Relay IR", sub: "图级中间表示", icon: "📊" },
    { label: "优化 Pass", sub: "融合/常量折叠", icon: "⚡" },
    { label: "TIR", sub: "张量级别 IR", icon: "🔧" },
    { label: "Codegen", sub: "目标代码生成", icon: "💻" },
    { label: "Runtime", sub: "部署执行", icon: "🚀" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        编译管线流程
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {steps.map((step, i) => (
          <div key={i} className="relative flex flex-col items-center">
            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900 w-full text-center hover:shadow-xl transition-shadow">
              <div className="text-2xl mb-2">{step.icon}</div>
              <div className="text-sm font-bold text-slate-800 dark:text-slate-100">{step.label}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{step.sub}</div>
            </div>
            {i < steps.length - 1 && (
              <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 text-indigo-400 z-10">
                →
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>端到端：</strong>从深度学习框架导入模型，经过多层 IR 转换与优化，最终部署到目标硬件。
      </div>
    </div>
  );
}

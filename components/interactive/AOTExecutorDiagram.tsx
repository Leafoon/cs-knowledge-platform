"use client";

export function AOTExecutorDiagram() {
  const steps = [
    { label: "TIR", desc: "张量级别中间表示", color: "from-blue-500 to-blue-600" },
    { label: "C Source", desc: "生成纯 C 源代码", color: "from-indigo-500 to-indigo-600" },
    { label: "编译", desc: "GCC/LLVM 编译为目标码", color: "from-purple-500 to-purple-600" },
    { label: "执行", desc: "直接运行，无运行时开销", color: "from-violet-500 to-violet-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        AOT 执行器
      </h3>
      <div className="flex flex-col md:flex-row items-stretch gap-3">
        {steps.map((step, i) => (
          <div key={i} className="flex-1 flex items-center">
            <div className="w-full">
              <div className={`bg-gradient-to-br ${step.color} text-white rounded-xl p-5 shadow-lg text-center`}>
                <div className="text-sm font-bold mb-1">Step {i + 1}</div>
                <div className="text-lg font-bold">{step.label}</div>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 text-center mt-2">{step.desc}</p>
            </div>
            {i < steps.length - 1 && (
              <svg className="w-6 h-6 text-indigo-400 mx-2 shrink-0 hidden md:block" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>AOT 优势：</strong>无需运行时解释器，生成静态二进制，适合边缘设备部署（如 MicroTVM）。
      </div>
    </div>
  );
}

"use client";

export function CompileEngineCallChain() {
  const steps = [
    { label: "relay.build(mod, target)", color: "from-blue-500 to-blue-600" },
    { label: "Optimize Passes", color: "from-indigo-500 to-indigo-600" },
    { label: "FuseOps", color: "from-violet-500 to-violet-600" },
    { label: "Lower to TE", color: "from-purple-500 to-purple-600" },
    { label: "Schedule → TIR", color: "from-fuchsia-500 to-fuchsia-600" },
    { label: "Codegen → Library", color: "from-pink-500 to-pink-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        编译引擎调用链
      </h3>
      <div className="relative">
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-400 via-purple-400 to-pink-400 hidden md:block" />
        <div className="flex flex-col gap-3">
          {steps.map((step, i) => (
            <div key={i} className="flex items-center gap-4 md:ml-0">
              <div className={`bg-gradient-to-r ${step.color} text-white w-12 h-12 rounded-full flex items-center justify-center font-bold text-sm shrink-0 shadow-lg z-10`}>
                {i + 1}
              </div>
              <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl px-5 py-3 shadow-lg border border-indigo-100 dark:border-indigo-900">
                <code className="text-sm font-mono text-slate-800 dark:text-slate-100">{step.label}</code>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>六步编译：</strong>从 relay.build 入口开始，经过优化、融合、降级、调度、代码生成，输出可执行库。
      </div>
    </div>
  );
}

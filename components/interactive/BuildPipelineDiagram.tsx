"use client";

export function BuildPipelineDiagram() {
  const steps = [
    { label: "输入 (Relay IR)", color: "from-blue-500 to-blue-600" },
    { label: "FuseOps", color: "from-indigo-500 to-indigo-600" },
    { label: "TE Lower", color: "from-purple-500 to-purple-600" },
    { label: "Schedule", color: "from-violet-500 to-violet-600" },
    { label: "TIR", color: "from-fuchsia-500 to-fuchsia-600" },
    { label: "CodeGen", color: "from-pink-500 to-pink-600" },
    { label: "输出 (Library)", color: "from-rose-500 to-rose-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        relay.build 管线
      </h3>
      <div className="flex flex-col md:flex-row items-center justify-between gap-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center">
            <div
              className={`bg-gradient-to-br ${step.color} text-white px-4 py-3 rounded-xl text-sm font-semibold text-center shadow-lg min-w-[100px]`}
            >
              {step.label}
            </div>
            {i < steps.length - 1 && (
              <svg
                className="w-6 h-6 text-indigo-400 dark:text-indigo-300 mx-1 shrink-0 hidden md:block"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            )}
            {i < steps.length - 1 && (
              <svg
                className="w-6 h-6 text-indigo-400 dark:text-indigo-300 my-1 shrink-0 md:hidden"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 9l7 7 7-7" />
              </svg>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>核心流程：</strong>relay.build 接收 Relay IR，经过算子融合、TE 降级、调度优化、TIR 转换、代码生成，最终输出可执行库。
      </div>
    </div>
  );
}

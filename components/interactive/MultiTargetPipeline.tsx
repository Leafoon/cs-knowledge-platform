"use client";

export function MultiTargetPipeline() {
  const targets = [
    { name: "CPU (LLVM)", color: "from-blue-500 to-blue-600", arch: "x86_64 / ARM" },
    { name: "CUDA", color: "from-green-500 to-green-600", arch: "SM 7.0 / 8.0" },
    { name: "OpenCL", color: "from-orange-500 to-orange-600", arch: "Adreno / Mali" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        多 Target 编译管线
      </h3>
      <div className="flex flex-col gap-4">
        <div className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-xl p-4 text-center shadow-lg">
          <div className="font-bold">源码 (Relay IR)</div>
          <div className="text-xs opacity-80 mt-1">统一的中间表示</div>
        </div>

        <div className="flex justify-center">
          <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900 text-center">
          <div className="font-semibold text-slate-800 dark:text-slate-100 text-sm">Target 选择与分发</div>
          <code className="text-xs text-indigo-600 dark:text-indigo-400">tvm.target.Target(...)</code>
        </div>

        <div className="flex justify-center">
          <svg className="w-6 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {targets.map((t, i) => (
            <div key={i} className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-indigo-100 dark:border-indigo-900 overflow-hidden">
              <div className={`bg-gradient-to-r ${t.color} text-white px-4 py-2 text-sm font-bold text-center`}>
                {t.name}
              </div>
              <div className="p-3 text-center">
                <div className="text-xs text-slate-500 dark:text-slate-400">{t.arch}</div>
                <div className="mt-2 bg-slate-50 dark:bg-slate-700/50 rounded-lg px-3 py-1.5 text-xs text-slate-600 dark:text-slate-300">
                  分别编译 → 部署
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>多后端：</strong>同一份 Relay IR 可针对不同硬件分别编译，实现一次编写多处部署。
      </div>
    </div>
  );
}

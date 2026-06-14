"use client";

export function GraphRuntimeArchitecture() {
  const layers = [
    { name: "Executor", desc: "图执行器，管理节点执行顺序", color: "from-blue-500 to-blue-600" },
    { name: "Op", desc: "算子调度，调用具体计算内核", color: "from-indigo-500 to-indigo-600" },
    { name: "Memory", desc: "内存管理，池化分配与回收", color: "from-purple-500 to-purple-600" },
    { name: "Device", desc: "设备抽象层 (CPU/GPU/FPGA)", color: "from-violet-500 to-violet-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        图执行器架构
      </h3>
      <div className="flex flex-col gap-3">
        {layers.map((layer, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className={`bg-gradient-to-r ${layer.color} text-white px-6 py-4 rounded-xl text-sm font-semibold shadow-lg w-36 text-center`}>
              {layer.name}
            </div>
            <div className="flex-1 p-3 bg-white/60 dark:bg-slate-800/60 rounded-lg text-sm text-slate-600 dark:text-slate-300">
              {layer.desc}
            </div>
            {i < layers.length - 1 && (
              <svg className="w-5 h-5 text-indigo-400 absolute -bottom-3 left-1/2 -translate-x-1/2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>四层架构：</strong>Executor 驱动图遍历 → Op 层调度算子 → Memory 层管理存储 → Device 层执行硬件操作。
      </div>
    </div>
  );
}

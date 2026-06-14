"use client";

export function TVMEcosystemMap() {
  const areas = [
    {
      title: "核心 (Core)",
      color: "from-blue-500 to-blue-600",
      items: ["Relay", "TIR", "TE", "IRModule"],
    },
    {
      title: "工具 (Tools)",
      color: "from-indigo-500 to-indigo-600",
      items: ["AutoTVM", "Ansor", "RPC Tracker", "Profiler"],
    },
    {
      title: "社区 (Community)",
      color: "from-purple-500 to-purple-600",
      items: ["Frontends", "Topi", "MicroTVM", "Relax"],
    },
    {
      title: "硬件 (Hardware)",
      color: "from-violet-500 to-violet-600",
      items: ["CUDA", "ROCm", "OpenCL", "ARM CPU"],
    },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        TVM 生态地图
      </h3>
      <div className="grid grid-cols-2 gap-4">
        {areas.map((area, i) => (
          <div key={i} className={`bg-gradient-to-br ${area.color} text-white rounded-xl p-5 shadow-lg`}>
            <div className="font-bold text-lg mb-3">{area.title}</div>
            <div className="flex flex-wrap gap-2">
              {area.items.map((item, j) => (
                <span key={j} className="bg-white/20 px-3 py-1 rounded-full text-xs font-medium">
                  {item}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>四大领域：</strong>TVM 生态涵盖核心编译器、自动化调优工具、社区贡献和硬件后端支持。
      </div>
    </div>
  );
}

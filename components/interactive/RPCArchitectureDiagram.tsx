"use client";

export function RPCArchitectureDiagram() {
  const nodes = [
    { name: "Client", desc: "用户端，提交调优任务", color: "from-blue-500 to-blue-600", icon: "👤" },
    { name: "Tracker", desc: "调度中心，分配设备资源", color: "from-indigo-500 to-indigo-600", icon: "📡" },
    { name: "Server", desc: "设备端，执行远程调用", color: "from-purple-500 to-purple-600", icon: "🖥️" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        RPC 架构
      </h3>
      <div className="flex flex-col md:flex-row items-center justify-around gap-6">
        {nodes.map((node, i) => (
          <div key={i} className="flex items-center">
            <div className={`bg-gradient-to-br ${node.color} text-white rounded-2xl p-6 shadow-xl text-center min-w-[140px]`}>
              <div className="text-3xl mb-2">{node.icon}</div>
              <div className="font-bold text-lg">{node.name}</div>
              <div className="text-xs opacity-90 mt-2">{node.desc}</div>
            </div>
            {i < nodes.length - 1 && (
              <div className="flex flex-col items-center mx-3">
                <svg className="w-10 h-8 text-indigo-400 dark:text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
                <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {i === 0 ? "请求/响应" : "任务分发"}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>三节点架构：</strong>Client 提交请求 → Tracker 调度分配 → Server 远程执行，支持跨设备自动调优。
      </div>
    </div>
  );
}

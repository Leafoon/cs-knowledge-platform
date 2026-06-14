"use client";

import { useState } from "react";

const tools = [
  {
    id: "profiler",
    name: "Memory Profiler",
    icon: "📊",
    color: "from-blue-500 to-indigo-500",
    desc: "分析内存使用模式和热点",
    features: ["分配/释放频率统计", "内存占用时间线", "峰值使用检测", "热点函数定位"],
    usage: `from tvm.contrib import profiler
with profiler.Profile() as p:
    mod(a, b)
print(p.table())`,
  },
  {
    id: "tracker",
    name: "Allocation Tracker",
    icon: "🔍",
    color: "from-indigo-500 to-purple-500",
    desc: "追踪每次分配的来源和生命周期",
    features: ["分配调用栈记录", "Tensor 生命周期追踪", "泄漏检测", "引用计数监控"],
    usage: `tvm.instrument.PassTiming(True)
# 追踪每次内存分配
tvm.memory.enable_tracker()
# 打印活跃分配
tvm.memory.dump_allocations()`,
  },
  {
    id: "dump",
    name: "Memory Dump",
    icon: "💾",
    color: "from-purple-500 to-pink-500",
    desc: "导出内存快照用于离线分析",
    features: ["堆快照导出", "Diff 两个快照", "二进制数据检查", "可视化内存布局"],
    usage: `# 导出当前内存快照
tvm.memory.dump_snapshot("heap.bin")
# 对比两个快照
tvm.memory.diff_snapshots(
    "before.bin", "after.bin"
)`,
  },
];

export function MemoryDebuggingDiagram() {
  const [active, setActive] = useState("profiler");

  const tool = tools.find((t) => t.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">内存调试工具</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">TVM 内存调试三件套: Profiler / Tracker / Dump</p>

      <div className="grid grid-cols-3 gap-3 mb-6">
        {tools.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`p-4 rounded-xl border-2 transition-all duration-300 text-left ${
              active === t.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <div className="text-2xl mb-2">{t.icon}</div>
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200">{t.name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{t.desc}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-3">功能特性</h4>
          <ul className="space-y-2">
            {tool.features.map((f, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                {f}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <div className="text-slate-500 mb-2"># 使用示例</div>
          <pre>{tool.usage}</pre>
        </div>
      </div>

      <div className="mt-5 flex items-center gap-3 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-3">
        <span className="text-lg">💡</span>
        <p className="text-xs text-slate-600 dark:text-slate-300">
          <strong>最佳实践:</strong> 开发阶段用 Tracker 检测泄漏，优化阶段用 Profiler 找热点，部署前用 Dump 验证内存布局。
        </p>
      </div>
    </div>
  );
}

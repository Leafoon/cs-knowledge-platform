"use client";

import { useState } from "react";

const tools = [
  {
    id: "lower",
    name: "tvm.lower()",
    icon: "🔍",
    color: "from-blue-500 to-indigo-500",
    desc: "查看中间 IR 表示",
    features: ["TE → TIR 转换可视化", "检查调度效果", "验证循环结构", "调试 schedule 错误"],
    usage: `# 查看 lower 后的 TIR
sch = te.create_schedule(C.op)
mod = tvm.lower(sch, [A, B, C], name="main")
print(mod)
# 输出:
# primfn(A, B, C) ->
#   for (i, 0, 128):
#     for (j, 0, 128):
#       C[i*128+j] = A[i*128+j] * B[i*128+j]`,
    when: "开发新 schedule 或调试调度问题时",
  },
  {
    id: "passctx",
    name: "PassContext",
    icon: "⚙️",
    color: "from-indigo-500 to-purple-500",
    desc: "配置和控制 Pass 执行",
    features: ["设置优化级别 (opt_level)", "启用/禁用特定 Pass", "配置调试选项", "性能追踪开关"],
    usage: `# 使用 PassContext 控制优化
with tvm.transform.PassContext(
    opt_level=3,
    config={"tir.debug_nan": True},
    disabled_pass=["FoldScaleAxis"]
):
    lib = tvm.build(mod, target)

# 性能追踪
with tvm.transform.PassContext(
    trace=True
):
    lib = tvm.build(mod, target)`,
    when: "需要调整优化策略或调试编译问题时",
  },
  {
    id: "profiler",
    name: "TVM Profiler",
    icon: "📊",
    color: "from-purple-500 to-pink-500",
    desc: "性能分析和瓶颈定位",
    features: ["算子级耗时统计", "内存使用分析", "GPU kernel profiling", "端到端性能报告"],
    usage: `# 性能分析
from tvm.contrib import Profiler
profiler = Profiler()
profiler.start()
for _ in range(100):
    f(a, b)
profiler.stop()
print(profiler.table())

# GPU profiling
with tvm.autotvm.measure.measure_methods.gpu():
    result = measure_fn(input)`,
    when: "模型性能不达标需要定位瓶颈时",
  },
];

export function DebugToolchainSummary() {
  const [active, setActive] = useState("lower");

  const tool = tools.find((t) => t.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">调试工具链</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">tvm.lower / PassContext / Profiler 三件套</p>

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
            <span className="text-2xl">{t.icon}</span>
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200 mt-1">{t.name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{t.desc}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">功能</h4>
          <ul className="space-y-1.5 mb-3">
            {tool.features.map((f, i) => (
              <li key={i} className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                {f}
              </li>
            ))}
          </ul>
          <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-3">
            <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400 mb-1">使用场景</div>
            <div className="text-xs text-slate-600 dark:text-slate-300">{tool.when}</div>
          </div>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{tool.usage}</pre>
        </div>
      </div>

      <div className="bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">调试工作流</h4>
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {[
            { step: "1. lower()", desc: "检查 IR" },
            { step: "2. PassContext", desc: "配置优化" },
            { step: "3. build()", desc: "编译" },
            { step: "4. Profiler", desc: "性能分析" },
            { step: "5. 迭代", desc: "优化 schedule" },
          ].map((s, i) => (
            <div key={i} className="flex items-center">
              <div className="text-center">
                <div className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r from-indigo-500 to-purple-500 text-white">
                  {s.step}
                </div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">{s.desc}</div>
              </div>
              {i < 4 && <span className="mx-1 text-indigo-400">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";

const templateParts = [
  {
    id: "declaration",
    name: "模板声明",
    icon: "📋",
    desc: "定义 schedule 函数和参数空间",
    code: `@autotvm.template("matmul")
def matmul(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C"
    )`,
    detail: "使用 @autotvm.template 装饰器\n定义计算图和参数维度\nAutoTVM 在此空间内搜索最优配置",
  },
  {
    id: "configspace",
    name: "配置空间",
    icon: "🔧",
    desc: "定义可调参数 (knob)",
    code: `cfg = autotvm.get_config()
cfg.define_knob("tile_n", [32, 64, 128, 256])
cfg.define_knob("tile_m", [32, 64, 128])
cfg.define_knob("unroll_factor", [1, 2, 4, 8])`,
    detail: "每个 knob 定义一个搜索维度\ntotal_configs = ∏(每个 knob 的选项数)\n示例: 4×3×4 = 48 种配置",
  },
  {
    id: "schedule",
    name: "调度模板",
    icon: "📐",
    desc: "根据配置生成 schedule",
    code: `s = te.create_schedule(C.op)
bn, bm = cfg["tile_n"].val, cfg["tile_m"].val
ni, noi = s[C].split(C.op.axis[0], factor=bn)
mi, moi = s[C].split(C.op.axis[1], factor=bm)
s[C].reorder(ni, mi, noi, moi)
cfg.define_knob("unroll_factor", [1,2,4])
if cfg["unroll_factor"].val > 1:
    s[C].unroll(noi, cfg["unroll_factor"].val)`,
    detail: "模板根据 knob 值动态生成 schedule\ntvm.lower() 可查看生成的 TIR\nAutoTVM 搜索最优 knob 组合",
  },
];

export function ScheduleTemplateExplorer() {
  const [active, setActive] = useState("declaration");

  const part = templateParts.find((p) => p.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">Schedule Template 探索</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">AutoTVM 模板的三部分结构</p>

      <div className="grid grid-cols-3 gap-3 mb-6">
        {templateParts.map((p) => (
          <button
            key={p.id}
            onClick={() => setActive(p.id)}
            className={`p-4 rounded-xl border-2 transition-all duration-300 text-left ${
              active === p.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span className="text-xl">{p.icon}</span>
            <div className="text-sm font-bold text-slate-700 dark:text-slate-200 mt-1">{p.name}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{p.desc}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">说明</h4>
          <p className="text-sm text-slate-600 dark:text-slate-300 whitespace-pre-wrap leading-relaxed">
            {part.detail}
          </p>
          <div className="mt-3 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-3">
            <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400 mb-1">模板结构</div>
            <div className="text-xs text-slate-600 dark:text-slate-300">
              声明 → 配置空间 → 调度模板
              <br />
              AutoTVM 搜索器遍历配置空间，找到最优 schedule
            </div>
          </div>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{part.code}</pre>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";

const mapping = {
  teSchedule: {
    title: "TE Schedule (调度层)",
    code: `s = te.create_schedule(C.op)
# split: 将轴分为外层和内层
ni, noi = s[C].split(C.op.axis[0], factor=32)
mi, moi = s[C].split(C.op.axis[1], factor=32)
# reorder: 调整循环顺序
s[C].reorder(ni, mi, noi, moi)
# vectorize: 内层向量化
s[C].vectorize(moi)`,
  },
  tirResult: {
    title: "TIR 结果 (底层表示)",
    code: `@T.prim_func
def main(A: T.Buffer, B: T.Buffer):
    for ni in T.serial(4):        # outer
      for mi in T.serial(4):      # outer
        for noi in T.serial(32):  # inner
          for moi in T.vectorized(32):  # vectorized
            B[ni*32+noi, mi*32+moi] = \\
              A[ni*32+noi, mi*32+moi] * 2.0`,
  },
};

const transforms = [
  { te: "split(axis, 32)", tir: "外层 serial(4) + 内层 serial(32)", desc: "轴分割为外层循环和内层循环" },
  { te: "reorder(ni, mi, noi, moi)", tir: "循环顺序: ni → mi → noi → moi", desc: "调整嵌套循环顺序优化缓存" },
  { te: "vectorize(moi)", tir: "最内层 T.vectorized(32)", desc: "最内层循环向量化执行" },
  { te: "unroll(noi)", tir: "循环展开 32 次", desc: "内层循环完全展开减少开销" },
];

export function TIRScheduleMapping() {
  const [activeTransform, setActiveTransform] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">TE→TIR 调度映射</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">TE split → TIR For 循环的对应关系</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4">
          <h4 className="text-xs font-bold text-blue-400 mb-2">{mapping.teSchedule.title}</h4>
          <pre className="font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">{mapping.teSchedule.code}</pre>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4">
          <h4 className="text-xs font-bold text-purple-400 mb-2">{mapping.tirResult.title}</h4>
          <pre className="font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">{mapping.tirResult.code}</pre>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-5">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">变换对照表</h4>
        <div className="space-y-2">
          {transforms.map((t, i) => (
            <button
              key={i}
              onClick={() => setActiveTransform(i)}
              className={`w-full flex items-center gap-4 px-4 py-3 rounded-lg transition-all ${
                activeTransform === i
                  ? "bg-indigo-100 dark:bg-indigo-900/40 border border-indigo-300 dark:border-indigo-600"
                  : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
              }`}
            >
              <code className="text-xs text-indigo-600 dark:text-indigo-400 font-mono min-w-[180px]">{t.te}</code>
              <span className="text-indigo-400">→</span>
              <code className="text-xs text-purple-600 dark:text-purple-400 font-mono flex-1">{t.tir}</code>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">TE 层</div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            用户友好的调度 API，描述"怎么分"
          </div>
        </div>
        <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <div className="text-sm font-bold text-purple-600 dark:text-purple-400">TIR 层</div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            精确的循环结构，描述"怎么跑"
          </div>
        </div>
      </div>
    </div>
  );
}

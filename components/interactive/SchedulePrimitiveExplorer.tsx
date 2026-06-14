"use client";

export function SchedulePrimitiveExplorer() {
  const primitives = [
    { name: "split", desc: "将循环拆分为内外两层", example: "split(i, 4) → i.outer, i.inner", color: "from-blue-500 to-blue-600" },
    { name: "reorder", desc: "改变循环嵌套顺序", example: "reorder(i, j) → j, i", color: "from-indigo-500 to-indigo-600" },
    { name: "fuse", desc: "合并相邻循环为单层", example: "fuse(i, j) → fused", color: "from-purple-500 to-purple-600" },
    { name: "unroll", desc: "循环展开，消除循环开销", example: "unroll(i) → 展开所有迭代", color: "from-violet-500 to-violet-600" },
    { name: "vectorize", desc: "向量化循环，利用 SIMD", example: "vectorize(j) → SIMD 指令", color: "from-fuchsia-500 to-fuchsia-600" },
    { name: "parallel", desc: "并行化循环，多线程执行", example: "parallel(i) → 多线程", color: "from-pink-500 to-pink-600" },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        调度原语探索器
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {primitives.map((p, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-indigo-100 dark:border-indigo-900 overflow-hidden hover:shadow-xl transition-shadow">
            <div className={`bg-gradient-to-r ${p.color} text-white px-4 py-2 font-bold text-sm`}>
              {p.name}
            </div>
            <div className="p-4">
              <p className="text-sm text-slate-700 dark:text-slate-200 mb-2">{p.desc}</p>
              <code className="text-xs bg-slate-100 dark:bg-slate-700 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded block">
                {p.example}
              </code>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>六种原语：</strong>split/reorder/fuse/unroll/vectorize/parallel 是 TVM 调度优化的基础构建块。
      </div>
    </div>
  );
}

"use client";

export function FusionMotivationDiagram() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        算子融合动机
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-red-200 dark:border-red-900 overflow-hidden">
          <div className="bg-gradient-to-r from-red-500 to-red-600 text-white px-5 py-3 font-bold text-center">
            内存瓶颈 (无融合)
          </div>
          <div className="p-5 space-y-3">
            <div className="flex items-center gap-3">
              <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-red-700 dark:text-red-300">Conv2D → 写入内存 → 读取</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-red-700 dark:text-red-300">BatchNorm → 写入内存 → 读取</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-red-100 dark:bg-red-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-red-700 dark:text-red-300">ReLU → 写入内存</div>
              </div>
            </div>
            <div className="text-center text-sm font-semibold text-red-600 dark:text-red-400 mt-3">
              3 次写入 + 2 次读取 = 5 次内存访问
            </div>
            <div className="w-full bg-red-100 dark:bg-red-900/30 rounded-full h-3 mt-2">
              <div className="bg-red-500 h-3 rounded-full w-full" />
            </div>
            <div className="text-xs text-center text-slate-500 dark:text-slate-400">带宽利用率低</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-green-200 dark:border-green-900 overflow-hidden">
          <div className="bg-gradient-to-r from-green-500 to-green-600 text-white px-5 py-3 font-bold text-center">
            计算密集 (融合后)
          </div>
          <div className="p-5 space-y-3">
            <div className="flex items-center gap-3">
              <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-green-700 dark:text-green-300">Conv2D + BN + ReLU</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-green-700 dark:text-green-300">在寄存器中直接传递</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="bg-green-100 dark:bg-green-900/30 rounded-lg p-2 w-full">
                <div className="text-xs font-mono text-green-700 dark:text-green-300">一次写入最终结果</div>
              </div>
            </div>
            <div className="text-center text-sm font-semibold text-green-600 dark:text-green-400 mt-3">
              0 次中间读写 = 1 次内存访问
            </div>
            <div className="w-full bg-green-100 dark:bg-green-900/30 rounded-full h-3 mt-2">
              <div className="bg-green-500 h-3 rounded-full w-1/5" />
            </div>
            <div className="text-xs text-center text-slate-500 dark:text-slate-400">带宽利用率高</div>
          </div>
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>核心动机：</strong>算子融合将内存密集型转换为计算密集型，消除中间张量的内存读写开销。
      </div>
    </div>
  );
}

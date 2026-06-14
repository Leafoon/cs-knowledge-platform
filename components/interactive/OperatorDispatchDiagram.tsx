"use client";

export function OperatorDispatchDiagram() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        算子分发机制
      </h3>
      <div className="flex flex-col gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900">
          <div className="font-bold text-blue-600 dark:text-blue-400 mb-2">Relay Op</div>
          <div className="text-sm text-slate-600 dark:text-slate-300">高层算子描述（如 nn.conv2d）</div>
          <code className="text-xs text-indigo-600 dark:text-indigo-400 mt-1 block">relay.nn.conv2d(data, weight, ...)</code>
        </div>

        <div className="flex justify-center">
          <div className="flex items-center gap-2">
            <svg className="w-8 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            <span className="text-xs text-slate-500 dark:text-slate-400">compute lowering</span>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900">
          <div className="font-bold text-purple-600 dark:text-purple-400 mb-2">TE Compute</div>
          <div className="text-sm text-slate-600 dark:text-slate-300">张量表达式定义计算语义</div>
          <code className="text-xs text-indigo-600 dark:text-indigo-400 mt-1 block">te.compute((N, C, H, W), lambda n, c, h, w: ...)</code>
        </div>

        <div className="flex justify-center">
          <div className="flex items-center gap-2">
            <svg className="w-8 h-6 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            <span className="text-xs text-slate-500 dark:text-slate-400">schedule + lower</span>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border border-indigo-100 dark:border-indigo-900">
          <div className="font-bold text-violet-600 dark:text-violet-400 mb-2">TIR PrimFunc</div>
          <div className="text-sm text-slate-600 dark:text-slate-300">底层循环表示，可直接代码生成</div>
          <code className="text-xs text-indigo-600 dark:text-indigo-400 mt-1 block">for (i, 0, N): for (j, 0, C): ...</code>
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>分发链路：</strong>Relay Op → TE compute 定义语义 → Schedule 变换 → TIR PrimFunc 生成代码。
      </div>
    </div>
  );
}

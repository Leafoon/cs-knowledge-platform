"use client";

export function FuseVisualization() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        算子融合可视化
      </h3>
      <div className="flex flex-col md:flex-row items-center justify-around gap-8">
        <div className="flex flex-col items-center gap-3">
          <span className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-2">融合前</span>
          <div className="flex flex-col items-center gap-2">
            <div className="bg-blue-500 text-white px-6 py-2 rounded-lg text-sm font-semibold shadow-md w-32 text-center">
              Conv2D
            </div>
            <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            <div className="bg-indigo-500 text-white px-6 py-2 rounded-lg text-sm font-semibold shadow-md w-32 text-center">
              BatchNorm
            </div>
            <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            <div className="bg-purple-500 text-white px-6 py-2 rounded-lg text-sm font-semibold shadow-md w-32 text-center">
              ReLU
            </div>
          </div>
          <span className="text-xs text-red-500 dark:text-red-400 mt-1">3 次内存读写</span>
        </div>

        <div className="flex flex-col items-center">
          <svg className="w-12 h-12 text-indigo-400 dark:text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
          <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">FuseOps</span>
        </div>

        <div className="flex flex-col items-center gap-3">
          <span className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-2">融合后</span>
          <div className="bg-gradient-to-br from-blue-500 to-purple-600 text-white px-6 py-4 rounded-xl text-sm font-semibold shadow-lg w-44 text-center">
            <div>Conv2D</div>
            <div className="text-xs opacity-80">+ BatchNorm</div>
            <div className="text-xs opacity-80">+ ReLU</div>
          </div>
          <span className="text-xs text-green-500 dark:text-green-400 mt-1">1 次内存读写</span>
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>优势：</strong>融合减少中间张量的内存分配和拷贝，显著提升执行效率。
      </div>
    </div>
  );
}

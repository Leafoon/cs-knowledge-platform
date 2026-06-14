"use client";

export function MemoryArchitectureDiagram() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        TVM 内存架构
      </h3>
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 bg-gradient-to-b from-blue-500 to-blue-600 text-white rounded-xl p-4 shadow-lg text-center">
          <div className="font-bold text-lg mb-2">Host</div>
          <div className="text-xs opacity-90 space-y-1">
            <p>CPU 内存</p>
            <p>模型参数</p>
            <p>控制流逻辑</p>
          </div>
        </div>
        <div className="flex items-center justify-center">
          <div className="flex flex-col items-center gap-1">
            <svg className="w-8 h-8 text-indigo-400 rotate-90 md:rotate-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
            <span className="text-xs text-slate-500 dark:text-slate-400">数据传输</span>
          </div>
        </div>
        <div className="flex-1 bg-gradient-to-b from-indigo-500 to-indigo-600 text-white rounded-xl p-4 shadow-lg text-center">
          <div className="font-bold text-lg mb-2">Device</div>
          <div className="text-xs opacity-90 space-y-1">
            <p>GPU / NPU 内存</p>
            <p>计算中间结果</p>
            <p>输入输出张量</p>
          </div>
        </div>
        <div className="flex items-center justify-center">
          <div className="flex flex-col items-center gap-1">
            <svg className="w-8 h-8 text-purple-400 rotate-90 md:rotate-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
            <span className="text-xs text-slate-500 dark:text-slate-400">池化管理</span>
          </div>
        </div>
        <div className="flex-1 bg-gradient-to-b from-purple-500 to-purple-600 text-white rounded-xl p-4 shadow-lg text-center">
          <div className="font-bold text-lg mb-2">Memory Pool</div>
          <div className="text-xs opacity-90 space-y-1">
            <p>空闲列表管理</p>
            <p>内存块复用</p>
            <p>碎片整理</p>
          </div>
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>三层架构：</strong>Host 管理控制，Device 执行计算，Pool 优化内存分配效率。
      </div>
    </div>
  );
}

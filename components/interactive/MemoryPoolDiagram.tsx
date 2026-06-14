"use client";

export function MemoryPoolDiagram() {
  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-6">
        内存池机制
      </h3>
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg border border-indigo-100 dark:border-indigo-900">
          <div className="font-bold text-blue-600 dark:text-blue-400 mb-3">FreeList</div>
          <div className="space-y-2">
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-2 text-xs font-mono text-slate-700 dark:text-slate-300">
              [0x1000, 256B] → [0x2000, 512B] → [0x4000, 1KB]
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">维护空闲内存块链表</p>
          </div>
        </div>
        <div className="flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <div className="bg-green-500 text-white px-4 py-2 rounded-lg text-xs font-semibold shadow-md">Alloc</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">申请内存</div>
            <div className="bg-orange-500 text-white px-4 py-2 rounded-lg text-xs font-semibold shadow-md">Recycle</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">归还内存</div>
          </div>
        </div>
        <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg border border-indigo-100 dark:border-indigo-900">
          <div className="font-bold text-purple-600 dark:text-purple-400 mb-3">流程</div>
          <div className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
            <div className="flex items-center gap-2">
              <span className="bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">1</span>
              <span>查找 ≥ 请求大小的空闲块</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">2</span>
              <span>分裂或整块分配</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">3</span>
              <span>使用后归还 FreeList</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold">4</span>
              <span>合并相邻空闲块</span>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 p-3 bg-white/60 dark:bg-slate-800/60 rounded-xl text-sm text-slate-600 dark:text-slate-300">
        <strong>池化优势：</strong>通过 FreeList 管理，避免频繁系统调用，减少内存碎片，提升分配效率。
      </div>
    </div>
  );
}

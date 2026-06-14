"use client";

export function GPUMemoryBreakdownChart() {
    const layers = [
        { name: "Global Memory", size: "40 GB", latency: "~400 cycles", bandwidth: "1.5 TB/s", color: "from-slate-500 to-slate-600", pct: 100 },
        { name: "L2 Cache", size: "40 MB", latency: "~200 cycles", bandwidth: "5 TB/s", color: "from-blue-500 to-blue-600", pct: 80 },
        { name: "Shared Memory", size: "164 KB", latency: "~20 cycles", bandwidth: "19 TB/s", color: "from-purple-500 to-purple-600", pct: 60 },
        { name: "L1 Cache", size: "128 KB", latency: "~20 cycles", bandwidth: "19 TB/s", color: "from-indigo-500 to-indigo-600", pct: 55 },
        { name: "Register File", size: "256 KB", latency: "~1 cycle", bandwidth: "~80 TB/s", color: "from-blue-400 to-indigo-500", pct: 40 },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">GPU 内存层次</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="flex items-end justify-center gap-3 h-56 mb-4">
                    {layers.map((layer, i) => (
                        <div key={i} className="flex flex-col items-center" style={{ width: `${layer.pct}%` }}>
                            <div
                                className={`w-full bg-gradient-to-t ${layer.color} rounded-t-lg flex items-center justify-center text-white font-bold text-xs p-2 text-center`}
                                style={{ height: `${(layers.length - i) * 45}px` }}
                            >
                                {layer.name}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="flex items-center justify-center gap-1 text-xs text-slate-500">
                    <span>容量小/快</span>
                    <div className="w-32 h-1 bg-gradient-to-r from-indigo-500 to-slate-400 rounded mx-2" />
                    <span>容量大/慢</span>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {layers.map((layer, i) => (
                    <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg border-l-4 border-indigo-500">
                        <h5 className="font-bold text-slate-800 dark:text-slate-100 text-sm mb-2">{layer.name}</h5>
                        <div className="space-y-1 text-xs">
                            <div className="flex justify-between"><span className="text-slate-500">容量</span><span className="font-mono text-indigo-600 dark:text-indigo-400">{layer.size}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">延迟</span><span className="font-mono text-purple-600 dark:text-purple-400">{layer.latency}</span></div>
                            <div className="flex justify-between"><span className="text-slate-500">带宽</span><span className="font-mono text-blue-600 dark:text-blue-400">{layer.bandwidth}</span></div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-4 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-4 text-white text-center text-sm">
                💡 TVM 通过自动调度将数据放置在最优的内存层级中
            </div>
        </div>
    );
}

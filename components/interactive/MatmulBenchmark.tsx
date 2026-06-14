"use client";

export function MatmulBenchmark() {
    const benchmarks = [
        { name: "朴素循环", perf: 12, color: "bg-red-500", desc: "三重循环，无优化" },
        { name: "循环重排", perf: 28, color: "bg-orange-500", desc: "优化数据局部性" },
        { name: "分块(Tiling)", perf: 65, color: "bg-yellow-500", desc: "利用缓存分块" },
        { name: "向量化", perf: 78, color: "bg-green-500", desc: "SIMD指令加速" },
        { name: "多线程", perf: 85, color: "bg-blue-500", desc: "并行化外层循环" },
        { name: "TVM自动调优", perf: 95, color: "bg-indigo-500", desc: "自动搜索最优调度" },
        { name: "cuBLAS", perf: 100, color: "bg-purple-500", desc: "厂商高度优化库" },
    ];

    const matrixSizes = [
        { size: "256×256", naive: 2.1, tvm: 0.08, cublas: 0.06 },
        { size: "512×512", naive: 16.8, tvm: 0.45, cublas: 0.38 },
        { size: "1024×1024", naive: 134.2, tvm: 2.8, cublas: 2.3 },
        { size: "2048×2048", naive: 1073.6, tvm: 18.5, cublas: 15.2 },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">矩阵乘法基准测试</h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">不同调度策略的性能对比（A100 GPU）</p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-4 text-center">性能对比条形图 (1024×1024)</h4>
                <div className="space-y-3">
                    {benchmarks.map((b, i) => (
                        <div key={i} className="flex items-center gap-3">
                            <div className="w-28 shrink-0 text-right text-sm font-semibold text-slate-700 dark:text-slate-300">{b.name}</div>
                            <div className="flex-1 bg-slate-100 dark:bg-slate-700 rounded-full h-8 overflow-hidden">
                                <div className={`${b.color} h-full rounded-full flex items-center justify-end pr-3 transition-all duration-500`}
                                    style={{ width: `${b.perf}%` }}>
                                    <span className="text-white text-xs font-bold">{b.perf}%</span>
                                </div>
                            </div>
                            <div className="w-32 text-xs text-slate-500 dark:text-slate-400">{b.desc}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-4">📊 不同矩阵大小的运行时间 (ms)</h4>
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-slate-200 dark:border-slate-700">
                                <th className="py-2 text-left text-slate-600 dark:text-slate-400">大小</th>
                                <th className="py-2 text-right text-red-500">朴素</th>
                                <th className="py-2 text-right text-indigo-500">TVM</th>
                                <th className="py-2 text-right text-purple-500">cuBLAS</th>
                            </tr>
                        </thead>
                        <tbody>
                            {matrixSizes.map((r, i) => (
                                <tr key={i} className="border-b border-slate-100 dark:border-slate-700/50">
                                    <td className="py-2 font-mono text-slate-700 dark:text-slate-300">{r.size}</td>
                                    <td className="py-2 text-right font-mono text-red-600">{r.naive}</td>
                                    <td className="py-2 text-right font-mono text-indigo-600">{r.tvm}</td>
                                    <td className="py-2 text-right font-mono text-purple-600">{r.cublas}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 dark:text-blue-400 mb-4">🔍 关键观察</h4>
                    <div className="space-y-3">
                        {[
                            { icon: "📈", text: "TVM自动调优接近cuBLAS性能(95%)", color: "indigo" },
                            { icon: "⚡", text: "相比朴素实现提升50-70倍", color: "green" },
                            { icon: "🎯", text: "分块(Tiling)是最重要的优化", color: "blue" },
                            { icon: "🔄", text: "调度空间搜索是TVM的核心优势", color: "purple" },
                        ].map((obs, i) => (
                            <div key={i} className={`flex items-center gap-3 p-3 bg-${obs.color}-50 dark:bg-${obs.color}-900/20 rounded-lg`}>
                                <span className="text-xl">{obs.icon}</span>
                                <span className="text-sm text-slate-700 dark:text-slate-300">{obs.text}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

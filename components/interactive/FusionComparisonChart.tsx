"use client";

export function FusionComparisonChart() {
    const rows = [
        {
            strategy: "水平融合",
            applicable: "多个独立算子共享相同输入",
            example: "多头注意力中的Q/K/V投影",
            benefit: "减少内核启动次数",
            limitation: "算子间无数据依赖",
            color: "from-blue-500 to-indigo-500",
        },
        {
            strategy: "垂直融合",
            applicable: "串联的逐元素算子链",
            example: "Conv → BN → Relu",
            benefit: "消除中间张量分配",
            limitation: "需要逐元素操作",
            color: "from-purple-500 to-indigo-500",
        },
        {
            strategy: "Chunk融合",
            applicable: "大张量分块并行计算",
            example: "大矩阵乘法的分块",
            benefit: "提升并行度+减少缓冲",
            limitation: "需要可分块的数据",
            color: "from-amber-500 to-orange-500",
        },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">融合策略对比</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden mb-6">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="bg-indigo-50 dark:bg-indigo-900/30">
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">策略</th>
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">适用场景</th>
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">典型例子</th>
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">收益</th>
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">限制</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row, i) => (
                                <tr key={i} className="border-t border-slate-100 dark:border-slate-700 hover:bg-indigo-50/50 dark:hover:bg-indigo-900/10 transition-colors">
                                    <td className="px-4 py-3">
                                        <span className={`px-2 py-1 rounded-lg text-xs font-bold text-white bg-gradient-to-r ${row.color}`}>
                                            {row.strategy}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-slate-700 dark:text-slate-300">{row.applicable}</td>
                                    <td className="px-4 py-3 font-mono text-xs text-purple-600 dark:text-purple-400">{row.example}</td>
                                    <td className="px-4 py-3 text-emerald-600 dark:text-emerald-400">{row.benefit}</td>
                                    <td className="px-4 py-3 text-red-500 dark:text-red-400">{row.limitation}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    💡 实际编译器中，三种融合策略通常<strong className="text-indigo-600 dark:text-indigo-400">组合使用</strong>：先垂直融合消除中间张量，再水平融合减少启动开销，最后用 Chunk 融合提升并行度。
                </p>
            </div>
        </div>
    );
}

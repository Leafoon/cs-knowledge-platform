"use client";

export function IRTypeSignatureTable() {
    const rows = [
        {
            feature: "类型系统",
            relay: "代数数据类型 (ADT)",
            te: "基于张量的类型推导",
            tir: "显式类型标注",
        },
        {
            feature: "函数签名",
            relay: "fn(x: Tensor[(m, n), float32]) -> Tensor",
            te: "def compute(shape, fcompute)",
            tir: "@T.prim_func def f(A: T.handle)",
        },
        {
            feature: "张量类型",
            relay: "Tensor[(shape), dtype]",
            te: "Tensor(shape, dtype)",
            tir: "T.Buffer(shape, dtype)",
        },
        {
            feature: "控制流",
            relay: "If/Match/Let",
            te: "无（纯计算图）",
            tir: "For/If/Assert",
        },
        {
            feature: "内存标注",
            relay: "无（自动管理）",
            te: "无（由调度决定）",
            tir: "T.alloc_buffer(scope)",
        },
        {
            feature: "典型用途",
            relay: "高层图优化",
            te: "算子定义+调度",
            tir: "底层代码生成",
        },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">IR 类型签名对比</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden mb-6">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="bg-indigo-50 dark:bg-indigo-900/30">
                                <th className="px-4 py-3 text-left font-bold text-indigo-700 dark:text-indigo-300">特性</th>
                                <th className="px-4 py-3 text-left font-bold text-blue-700 dark:text-blue-300">Relay</th>
                                <th className="px-4 py-3 text-left font-bold text-purple-700 dark:text-purple-300">TE</th>
                                <th className="px-4 py-3 text-left font-bold text-emerald-700 dark:text-emerald-300">TIR</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row, i) => (
                                <tr key={i} className="border-t border-slate-100 dark:border-slate-700 hover:bg-indigo-50/50 dark:hover:bg-indigo-900/10 transition-colors">
                                    <td className="px-4 py-3 font-bold text-slate-800 dark:text-slate-100 text-xs">{row.feature}</td>
                                    <td className="px-4 py-3 font-mono text-xs text-blue-700 dark:text-blue-300">{row.relay}</td>
                                    <td className="px-4 py-3 font-mono text-xs text-purple-700 dark:text-purple-300">{row.te}</td>
                                    <td className="px-4 py-3 font-mono text-xs text-emerald-700 dark:text-emerald-300">{row.tir}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
                {[
                    { name: "Relay", desc: "高层 IR，函数式风格", color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-50 dark:bg-blue-900/20" },
                    { name: "TE", desc: "张量表达式，调度原语", color: "text-purple-600 dark:text-purple-400", bg: "bg-purple-50 dark:bg-purple-900/20" },
                    { name: "TIR", desc: "底层 IR，代码生成", color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-900/20" },
                ].map((ir, i) => (
                    <div key={i} className={`${ir.bg} rounded-xl p-4 text-center`}>
                        <div className={`text-lg font-bold ${ir.color}`}>{ir.name}</div>
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{ir.desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}

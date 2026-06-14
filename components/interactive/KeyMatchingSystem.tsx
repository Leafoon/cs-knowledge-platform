"use client";

export function KeyMatchingSystem() {
    const operators = ["nn.conv2d", "nn.dense", "nn.relu", "add", "multiply"];
    const targets = ["cuda", "llvm", "opencl", "metal"];
    const codegens = ["cuda_source", "llvm_ir", "opencl_source", "metal_source"];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">Target Key 匹配系统图</h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">算子 → Target → CodeGen 的三层匹配机制</p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid grid-cols-3 gap-4 items-start">
                    <div>
                        <h4 className="text-center font-bold text-indigo-600 dark:text-indigo-400 mb-3">算子</h4>
                        <div className="space-y-2">
                            {operators.map((op, i) => (
                                <div key={i} className="p-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg text-center text-sm font-mono text-indigo-700 dark:text-indigo-300">
                                    {op}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div>
                        <h4 className="text-center font-bold text-purple-600 dark:text-purple-400 mb-3">Target</h4>
                        <div className="space-y-2">
                            {targets.map((t, i) => (
                                <div key={i} className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg text-center text-sm font-mono text-purple-700 dark:text-purple-300">
                                    {t}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div>
                        <h4 className="text-center font-bold text-blue-600 dark:text-blue-400 mb-3">CodeGen</h4>
                        <div className="space-y-2">
                            {codegens.map((c, i) => (
                                <div key={i} className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg text-center text-sm font-mono text-blue-700 dark:text-blue-300">
                                    {c}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-4">🔑 Key 匹配规则</h4>
                    <div className="space-y-3">
                        {[
                            { rule: "精确匹配", desc: "算子名 + Target完全匹配", icon: "🎯" },
                            { rule: "模糊匹配", desc: "算子名匹配，Target通配", icon: "🔍" },
                            { rule: "回退机制", desc: "无匹配时使用默认实现", icon: "↩️" },
                        ].map((r, i) => (
                            <div key={i} className="flex items-start gap-3 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                                <span className="text-xl">{r.icon}</span>
                                <div>
                                    <div className="font-semibold text-sm text-slate-800 dark:text-slate-100">{r.rule}</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">{r.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-4">⚡ 匹配流程</h4>
                    <pre className="text-xs bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto">
{`# 匹配流程示意
def match(op, target):
    # 1. 构建key
    key = f"{op}::{target}"
    
    # 2. 查找调度表
    if key in schedule_table:
        return schedule_table[key]
    
    # 3. 通配符回退
    if f"{op}::*" in schedule_table:
        return schedule_table[f"{op}::*"]
    
    # 4. 默认实现
    return default_schedule(op)`}
                    </pre>
                </div>
            </div>

            <div className="mt-6 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-4 text-white text-center">
                <p className="text-sm">💡 TVM通过Target Key匹配系统实现算子到硬件后端的最优映射</p>
            </div>
        </div>
    );
}

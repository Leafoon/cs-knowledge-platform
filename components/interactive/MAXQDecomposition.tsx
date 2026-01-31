"use client";

export function MAXQDecomposition() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-lime-50 to-emerald-50 dark:from-slate-900 dark:to-lime-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MAXQ 分解
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">值函数分解</h4>
                <div className="font-mono text-center p-4 bg-lime-50 dark:bg-lime-900/20 rounded border-2 border-lime-500">
                    Q(s, a) = V(a, s) + C(a, s)
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                        <strong>V(a, s):</strong> 执行子任务 a 的值
                    </div>
                    <div className="p-3 bg-lime-50 dark:bg-lime-900/20 rounded">
                        <strong>C(a, s):</strong> 子任务完成后的值
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">层次化任务树示例</h4>
                <div className="p-4 bg-slate-50 dark:bg-slate-700 rounded">
                    <div className="font-mono text-sm">
                        <div className="mb-2">Root Task</div>
                        <div className="ml-4 mb-2">├─ Navigate to Room A</div>
                        <div className="ml-8 mb-2">│   ├─ Move North (primitive)</div>
                        <div className="ml-8 mb-2">│   └─ Move East (primitive)</div>
                        <div className="ml-4 mb-2">└─ Navigate to Room B</div>
                        <div className="ml-8 mb-2">    ├─ Move South (primitive)</div>
                        <div className="ml-8">    └─ Move West (primitive)</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-lime-600 mb-4">Composite Action</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-lime-50 dark:bg-lime-900/20 rounded">
                            <strong>定义:</strong> 由子任务组成
                        </div>
                        <div className="p-3 bg-lime-50 dark:bg-lime-900/20 rounded">
                            <strong>Q-value:</strong><br />
                            Q(M<sub>i</sub>, s, a) = V(M<sub>a</sub>, s) + C(M<sub>i</sub>, s, M<sub>a</sub>)
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-emerald-600 mb-4">Primitive Action</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>定义:</strong> 原始动作
                        </div>
                        <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                            <strong>Q-value:</strong><br />
                            Q(M<sub>i</sub>, s, a) = R(s,a) + γΣ P(s'|s,a) V(M<sub>i</sub>, s')
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">MAXQ 的优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 模块化学习</strong><br />
                        子任务独立学习
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 技能复用</strong><br />
                        子任务可用于多个任务
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 状态抽象</strong><br />
                        子任务隐藏内部状态
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 加速学习</strong><br />
                        分而治之降低复杂度
                    </div>
                </div>
            </div>
        </div>
    );
}

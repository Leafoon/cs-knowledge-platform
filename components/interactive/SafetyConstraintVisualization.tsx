"use client";

export function SafetyConstraintVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    安全约束可视化
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">CMDP定义</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>奖励函数 R(s,a):</strong> 最大化
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                        <strong>成本函数 C(s,a):</strong> 约束 E[Σγᵗc_t] ≤ d
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-blue-600 mb-4">标准RL</h4>
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="font-mono mb-2">max E[R]</div>
                        <div className="text-xs">无约束</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">安全RL (CMDP)</h4>
                    <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <div className="font-mono mb-2">max E[R]</div>
                        <div className="font-mono text-xs">s.t. E[C] ≤ d</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

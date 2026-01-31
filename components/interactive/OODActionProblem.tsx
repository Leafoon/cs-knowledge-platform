"use client";

export function OODActionProblem() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-red-50 to-pink-50 dark:from-slate-900 dark:to-red-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    OOD 动作问题
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Out-of-Distribution Actions
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">问题示例</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>数据集:</strong> 保守驾驶员数据<br />
                        <div className="text-sm mt-2">速度: [50, 60] km/h ✓</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <strong>Q-learning:</strong> max Q(s,a)<br />
                        <div className="text-sm mt-2">可能选择: 速度 = 120 km/h ❌</div>
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                        <strong>问题:</strong> OOD 动作的 Q 值不可靠<br />
                        <div className="text-sm mt-2">从未在数据中出现 → 外推误差 → 高估</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">数据内动作</h4>
                    <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">✓</div>
                        <div className="font-mono mb-2">(s,a) ∈ D</div>
                        <div className="text-sm">Q 值估计准确</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">OOD 动作</h4>
                    <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <div className="text-2xl mb-2">✗</div>
                        <div className="font-mono mb-2">(s,a) ∉ D</div>
                        <div className="text-sm">Q 值可能被高估</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

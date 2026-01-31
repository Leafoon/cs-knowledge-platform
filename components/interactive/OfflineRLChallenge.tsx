"use client";

export function OfflineRLChallenge() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Offline RL 挑战
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="text-4xl mb-3 text-center">⚠️</div>
                    <h4 className="text-lg font-bold text-violet-600 mb-3">OOD 动作</h4>
                    <div className="text-sm space-y-2">
                        <div className="p-2 bg-violet-50 dark:bg-violet-900/20 rounded">
                            数据未见过的 (s,a) 对
                        </div>
                        <div className="p-2 bg-violet-50 dark:bg-violet-900/20 rounded">
                            Q 值估计不准确
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="text-4xl mb-3 text-center">📈</div>
                    <h4 className="text-lg font-bold text-purple-600 mb-3">外推误差</h4>
                    <div className="text-sm space-y-2">
                        <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                            max Q(s,a) 可能选 OOD
                        </div>
                        <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                            Q 值被高估
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <div className="text-4xl mb-3 text-center">☠️</div>
                    <h4 className="text-lg font-bold text-red-600 mb-3">Deadly Triad</h4>
                    <div className="text-sm space-y-2">
                        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded">
                            函数逼近 ✓
                        </div>
                        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded">
                            Bootstrapping ✓
                        </div>
                        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded">
                            Off-policy ✓
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">解决方案对比</h4>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                            <th className="text-left p-3">方法</th>
                            <th className="text-center p-3">核心思想</th>
                            <th className="text-center p-3">复杂度</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3 font-bold">BCQ</td>
                            <td className="text-center p-3">限制动作在数据分布内</td>
                            <td className="text-center p-3">中</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3 font-bold">CQL</td>
                            <td className="text-center p-3">保守估计 Q 值</td>
                            <td className="text-center p-3">中</td>
                        </tr>
                        <tr className="border-b border-slate-100 dark:border-slate-800">
                            <td className="p-3 font-bold">IQL</td>
                            <td className="text-center p-3">期望值学习（避免 max）</td>
                            <td className="text-center p-3 text-green-600 font-bold">低</td>
                        </tr>
                        <tr>
                            <td className="p-3 font-bold">Decision Transformer</td>
                            <td className="text-center p-3">序列建模</td>
                            <td className="text-center p-3">高</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
}

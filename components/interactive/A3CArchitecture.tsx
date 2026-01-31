"use client";

export function A3CArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-slate-900 dark:to-orange-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    A3C 异步架构
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="text-center mb-6">
                    <div className="inline-block p-6 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-xl shadow-lg">
                        <div className="text-2xl font-bold">全局共享网络</div>
                        <div className="text-sm mt-2">参数: θ<sub>global</sub></div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="border-2 border-orange-300 dark:border-orange-700 rounded-xl p-4">
                            <div className="text-center font-bold text-orange-600 mb-3">
                                Worker {i}
                            </div>
                            <div className="space-y-2 text-sm">
                                <div className="p-2 bg-orange-50 dark:bg-orange-900/20 rounded">
                                    1. 复制全局参数
                                </div>
                                <div className="p-2 bg-orange-50 dark:bg-orange-900/20 rounded">
                                    2. 本地收集经验
                                </div>
                                <div className="p-2 bg-orange-50 dark:bg-orange-900/20 rounded">
                                    3. 计算梯度
                                </div>
                                <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                                    4. 异步更新全局
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-orange-600 mb-4">A3C (异步)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>✅ 高吞吐量:</strong> Worker 独立运行
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>⚠️ 梯度可能过时:</strong> 基于旧参数
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>🔧 实现复杂:</strong> 多线程同步
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">A2C (同步)</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 更稳定:</strong> 梯度基于当前参数
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 易实现:</strong> 无需多线程
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>💡 主流选择:</strong> 现代实践更常用
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

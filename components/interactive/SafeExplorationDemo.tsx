"use client";

export function SafeExplorationDemo() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    安全探索演示
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Shield机制</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <strong>策略提议:</strong> 动作a
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <strong>Shield检查:</strong> is_safe(s, a)?
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                            <strong>✓ 安全:</strong> 执行a
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded border-2 border-red-500">
                            <strong>✗ 不安全:</strong> 执行a_safe
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">安全集合</h4>
                <div className="text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded">
                    <div className="text-sm">S_safe = (s | 存在策略能保证安全)</div>
                </div>
            </div>
        </div>
    );
}

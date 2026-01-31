"use client";

export function MultiObjectiveTradeoff() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-slate-900 dark:to-orange-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    多目标权衡
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">自动驾驶示例</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong>保守用户:</strong> w = [0.8, 0.2]<br />
                        <div className="text-sm mt-2">
                            80% 安全性，20% 速度 → 慢但安全
                        </div>
                    </div>
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <strong>平衡用户:</strong> w = [0.5, 0.5]<br />
                        <div className="text-sm mt-2">
                            50% 安全性，50% 速度 → 适中
                        </div>
                    </div>
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <strong>激进用户:</strong> w = [0.2, 0.8]<br />
                        <div className="text-sm mt-2">
                            20% 安全性，80% 速度 → 快但有风险
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">加权奖励</h4>
                <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded font-mono">
                    r = w₁·r_safety + w₂·r_speed
                </div>
            </div>
        </div>
    );
}

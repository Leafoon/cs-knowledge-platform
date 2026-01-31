"use client";

export function MonotonicImprovement() {
    const rewards = [50, 75, 90, 110, 125, 140, 155, 165, 175, 185];
    const unstableRewards = [50, 80, 60, 120, 70, 140, 90, 100, 150, 130];

    const maxReward = Math.max(...rewards, ...unstableRewards);

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    单调改进曲线
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">✅ TRPO (单调改进)</h4>
                    <div className="h-48 bg-slate-50 dark:bg-slate-700 rounded relative">
                        <svg width="100%" height="100%" viewBox="0 0 300 150" preserveAspectRatio="none">
                            <polyline
                                fill="none"
                                stroke="rgb(34, 197, 94)"
                                strokeWidth="3"
                                points={rewards.map((r, i) => {
                                    const x = (i / (rewards.length - 1)) * 300;
                                    const y = 150 - (r / maxReward) * 130;
                                    return `${x},${y}`;
                                }).join(' ')}
                            />
                            {rewards.map((r, i) => {
                                const x = (i / (rewards.length - 1)) * 300;
                                const y = 150 - (r / maxReward) * 130;
                                return <circle key={i} cx={x} cy={y} r="3" fill="rgb(34, 197, 94)" />;
                            })}
                        </svg>
                    </div>
                    <div className="mt-3 text-sm text-center text-green-700 dark:text-green-400">
                        性能稳定上升，无崩溃
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">❌ 无约束 PG (不稳定)</h4>
                    <div className="h-48 bg-slate-50 dark:bg-slate-700 rounded relative">
                        <svg width="100%" height="100%" viewBox="0 0 300 150" preserveAspectRatio="none">
                            <polyline
                                fill="none"
                                stroke="rgb(239, 68, 68)"
                                strokeWidth="3"
                                points={unstableRewards.map((r, i) => {
                                    const x = (i / (unstableRewards.length - 1)) * 300;
                                    const y = 150 - (r / maxReward) * 130;
                                    return `${x},${y}`;
                                }).join(' ')}
                            />
                            {unstableRewards.map((r, i) => {
                                const x = (i / (unstableRewards.length - 1)) * 300;
                                const y = 150 - (r / maxReward) * 130;
                                return <circle key={i} cx={x} cy={y} r="3" fill="rgb(239, 68, 68)" />;
                            })}
                        </svg>
                    </div>
                    <div className="mt-3 text-sm text-center text-red-700 dark:text-red-400">
                        性能波动大，频繁崩溃
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">单调改进保证</h4>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                    <div className="font-mono text-center mb-2">
                        J(π<sub>new</sub>) ≥ J(π<sub>old</sub>)
                    </div>
                    <div className="text-sm text-center text-slate-600 dark:text-slate-400">
                        TRPO 通过 KL 约束保证每次更新后性能不会变差
                    </div>
                </div>
            </div>
        </div>
    );
}

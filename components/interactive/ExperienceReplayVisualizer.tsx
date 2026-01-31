"use client";

export function ExperienceReplayVisualizer() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-900 dark:to-green-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Experience Replay 机制
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-red-600 mb-4">❌ 无 Replay</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题 1:</strong> 样本强相关<br />
                            连续样本来自同一 episode
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题 2:</strong> 样本效率低<br />
                            每个经验只使用一次
                        </div>
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                            <strong>问题 3:</strong> 灾难性遗忘<br />
                            新经验覆盖旧知识
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-green-600 mb-4">✅ 有 Replay</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势 1:</strong> 打破相关性<br />
                            随机采样 → i.i.d.
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势 2:</strong> 提高效率<br />
                            重复使用经验
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>优势 3:</strong> 平滑学习<br />
                            稳定梯度更新
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Buffer 操作</h4>
                <div className="grid grid-cols-3 gap-4 font-mono text-sm">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded text-center">
                        <div className="font-bold mb-2">存储</div>
                        <div>buffer.push(s,a,r,s')</div>
                    </div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded text-center">
                        <div className="font-bold mb-2">采样</div>
                        <div>batch = buffer.sample(64)</div>
                    </div>
                    <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded text-center">
                        <div className="font-bold mb-2">容量</div>
                        <div>max_size = 1M</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

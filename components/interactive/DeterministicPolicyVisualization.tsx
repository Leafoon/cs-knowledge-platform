"use client";

export function DeterministicPolicyVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-blue-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    确定性策略 vs 随机策略
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 mb-4">随机策略 π(a|s;θ)</h4>
                    <div className="space-y-4 text-sm">
                        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <div className="font-bold mb-2">输出:</div>
                            <div>动作的<strong>概率分布</strong></div>
                        </div>
                        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <div className="font-bold mb-2">示例:</div>
                            <div className="font-mono text-xs">
                                mean, std = network(s)<br />
                                a ~ N(mean, std)
                            </div>
                        </div>
                        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <div className="font-bold mb-2">优势:</div>
                            <div>• 内在探索<br />• 适合离散动作</div>
                        </div>
                        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <div className="font-bold mb-2">劣势:</div>
                            <div>• 连续空间方差大<br />• 样本效率低</div>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-sky-500">
                    <h4 className="text-lg font-bold text-sky-600 mb-4">确定性策略 μ(s;θ)</h4>
                    <div className="space-y-4 text-sm">
                        <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-bold mb-2">输出:</div>
                            <div><strong>确定的动作</strong></div>
                        </div>
                        <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-bold mb-2">示例:</div>
                            <div className="font-mono text-xs">
                                a = network(s)<br />
                                (直接输出动作值)
                            </div>
                        </div>
                        <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-bold mb-2">优势:</div>
                            <div>• 低方差梯度<br />• 高样本效率<br />• 适合连续控制</div>
                        </div>
                        <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                            <div className="font-bold mb-2">劣势:</div>
                            <div>• 需要外部探索噪声</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">探索机制对比</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>随机策略:</strong><br />
                        从分布采样 → 内在探索
                    </div>
                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded">
                        <strong>确定性策略:</strong><br />
                        a = μ(s) + 噪声 → 外在探索
                    </div>
                </div>
            </div>
        </div>
    );
}

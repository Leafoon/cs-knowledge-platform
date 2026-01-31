"use client";

export function DistributedRLArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    分布式 RL 架构
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-blue-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🎮</div>
                        <h4 className="text-xl font-bold text-blue-600">Actors</h4>
                        <div className="text-sm text-slate-500">(并行环境交互)</div>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>数量:</strong> N 个 (例如 100+)
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>任务:</strong> 收集经验
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>硬件:</strong> CPU
                        </div>
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <strong>策略:</strong> 可能稍旧
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-indigo-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">💾</div>
                        <h4 className="text-xl font-bold text-indigo-600">Replay Buffer</h4>
                        <div className="text-sm text-slate-500">(经验存储)</div>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>容量:</strong> 1M+ transitions
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>功能:</strong> 存储、采样
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>优先级:</strong> PER (可选)
                        </div>
                        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                            <strong>共享:</strong> 所有 Actors
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-purple-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🧠</div>
                        <h4 className="text-xl font-bold text-purple-600">Learner</h4>
                        <div className="text-sm text-slate-500">(中心化训练)</div>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>数量:</strong> 1 个
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>任务:</strong> 梯度更新
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>硬件:</strong> GPU
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <strong>Batch:</strong> 大 (512+)
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">数据流</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div>Actors 与环境交互 → 收集 (s,a,r,s')</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-indigo-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div>经验 → Replay Buffer (异步存储)</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div>Learner 从 Buffer 采样 batch</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div>Learner 更新网络参数 θ</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                        <div>新参数 θ → Actors (周期性同步)</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">关键优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 样本收集速度 ×N</strong><br />
                        并行环境交互
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 墙钟时间大幅缩短</strong><br />
                        分布式加速
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 探索多样性</strong><br />
                        不同 Actors 探索不同区域
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 硬件利用率高</strong><br />
                        CPU 采样 + GPU 训练
                    </div>
                </div>
            </div>
        </div>
    );
}

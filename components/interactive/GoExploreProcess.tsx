"use client";

export function GoExploreProcess() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-slate-900 dark:to-amber-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Go-Explore 过程
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">核心流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>探索:</strong> 随机/策略探索，记录所有访问的状态</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>归档:</strong> 将状态降采样为 cell，存入 archive</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>选择:</strong> 从 archive 选择一个有趣的 cell</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>返回:</strong> 重放轨迹，返回到该 cell</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                        <div><strong>继续探索:</strong> 从该 cell 继续探索新区域</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-amber-600 mb-4">Cell Archive</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>存储:</strong> cell → (trajectory, reward)
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>Cell:</strong> 低分辨率状态（例如位置）
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>更新:</strong> 发现更高奖励时更新
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-orange-600 mb-4">选择策略</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>优先:</strong> 访问次数少的 cell
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>边界:</strong> 接近探索边界的 cell
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>奖励:</strong> 高奖励但未充分探索
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">关键成就：Montezuma's Revenge</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 系统探索</strong><br />
                        不依赖随机性
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 记忆机制</strong><br />
                        能精确返回有趣状态
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 突破难题</strong><br />
                        达到人类水平
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 稀疏奖励</strong><br />
                        有效处理极稀疏奖励
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">局限性</h4>
                <div className="space-y-2 text-sm">
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        ❌ 需要确定性环境（或低随机性）
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        ❌ 需要合适的状态表示（cell 定义）
                    </div>
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                        ❌ 返回机制可能不适用于所有环境
                    </div>
                </div>
            </div>
        </div>
    );
}

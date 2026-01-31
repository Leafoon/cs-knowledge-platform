"use client";

export function FeudalArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-green-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Feudal RL 架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Manager-Worker 层次结构
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-teal-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">👔</div>
                        <h4 className="text-xl font-bold text-teal-600">Manager (高层)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>任务:</strong> 设定目标 g<sub>t</sub>
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>时间尺度:</strong> 长（例如每 c 步）
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>奖励:</strong> 外部奖励
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>输出:</strong> g<sub>t</sub> ∈ ℝ<sup>d</sup>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-green-500">
                    <div className="text-center mb-4">
                        <div className="text-4xl mb-2">🔧</div>
                        <h4 className="text-xl font-bold text-green-600">Worker (低层)</h4>
                    </div>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>任务:</strong> 实现目标 g<sub>t</sub>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>时间尺度:</strong> 短（每步）
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>奖励:</strong> 内在奖励
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>输出:</strong> a<sub>t</sub> ∈ A
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">信息流</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-teal-50 to-green-50 dark:from-teal-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center font-bold">M</div>
                        <div><strong>Manager → Worker:</strong> 目标向量 g<sub>t</sub></div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-teal-50 to-green-50 dark:from-teal-900/20 dark:to-green-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">W</div>
                        <div><strong>Worker → Manager:</strong> 状态信息</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">内在奖励（Worker）</h4>
                <div className="font-mono text-center p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                    r<sub>intrinsic</sub> = cos(s<sub>t+1</sub> - s<sub>t</sub>, g<sub>t</sub>)
                </div>
                <div className="mt-4 text-sm text-center text-slate-600 dark:text-slate-400">
                    Worker 因朝向目标方向移动而获得奖励
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">FuN (FeUdal Networks) 特点</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong className="text-teal-700 dark:text-teal-400">✅ 层次化决策</strong><br />
                        不同时间尺度
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 目标设定</strong><br />
                        Manager 提供方向
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <strong className="text-teal-700 dark:text-teal-400">✅ 内在驱动</strong><br />
                        Worker 的内在奖励
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 端到端训练</strong><br />
                        同时学习 Manager 和 Worker
                    </div>
                </div>
            </div>
        </div>
    );
}

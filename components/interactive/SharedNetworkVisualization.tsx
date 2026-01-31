"use client";

export function SharedNetworkVisualization() {
    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-blue-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    共享网络结构
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <div className="space-y-6">
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="font-bold">输入状态 s</div>
                    </div>

                    <div className="text-center">↓</div>

                    <div className="p-6 bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-xl shadow-lg">
                        <div className="text-xl font-bold mb-2">共享特征层</div>
                        <div className="text-sm">参数共享，提取通用特征</div>
                        <div className="mt-3 space-y-1 text-xs">
                            <div>• Linear(state_dim → 256) + ReLU</div>
                            <div>• Linear(256 → 256) + ReLU</div>
                        </div>
                    </div>

                    <div className="text-center">↓</div>

                    <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded border-2 border-purple-500">
                        <div className="font-bold">特征 φ(s)</div>
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        <div className="space-y-3">
                            <div className="text-center text-2xl">🎭</div>
                            <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded border-2 border-indigo-500">
                                <div className="font-bold text-indigo-600 mb-2">Actor 头</div>
                                <div className="text-sm">Linear(256 → action_dim)</div>
                                <div className="text-sm">+ Softmax</div>
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded text-center">
                                <strong>π(a|s)</strong><br />
                                <span className="text-xs">策略分布</span>
                            </div>
                        </div>

                        <div className="space-y-3">
                            <div className="text-center text-2xl">🎓</div>
                            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded border-2 border-purple-500">
                                <div className="font-bold text-purple-600 mb-2">Critic 头</div>
                                <div className="text-sm">Linear(256 → 1)</div>
                            </div>
                            <div className="text-center">↓</div>
                            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded text-center">
                                <strong>V(s)</strong><br />
                                <span className="text-xs">状态价值</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">共享网络的优势</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">📦</div>
                        <div className="font-bold">参数共享</div>
                        <div className="text-xs mt-2">减少模型大小</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">⚡</div>
                        <div className="font-bold">特征复用</div>
                        <div className="text-xs mt-2">提高学习效率</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <div className="text-2xl mb-2">🎯</div>
                        <div className="font-bold">联合训练</div>
                        <div className="text-xs mt-2">Actor-Critic 互相帮助</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

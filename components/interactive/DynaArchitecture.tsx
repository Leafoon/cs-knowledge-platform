"use client";

export function DynaArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-slate-900 dark:to-orange-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    Dyna 架构
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    真实经验 + 模拟经验
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Dyna 循环</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>真实交互:</strong> Agent → Environment → (s,a,r,s')</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>Direct RL:</strong> 用真实经验更新 Q(s,a)</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>Model Learning:</strong> 用真实经验训练 Model</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-amber-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>Planning:</strong> 用 Model 生成模拟经验</div>
                    </div>
                    <div className="p-3 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                        <div><strong>Indirect RL:</strong> 用模拟经验更新 Q(s,a)</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-orange-600 mb-4">真实经验</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>来源:</strong> 环境交互
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>质量:</strong> 100% 准确
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <strong>数量:</strong> 有限（昂贵）
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 保证数据真实性</strong>
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-amber-500">
                    <h4 className="text-lg font-bold text-amber-600 mb-4">模拟经验</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>来源:</strong> 学习的模型
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>质量:</strong> 近似（有误差）
                        </div>
                        <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded">
                            <strong>数量:</strong> 无限（免费）
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <strong>✅ 提升样本效率</strong>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">Dyna-Q 伪代码</h4>
                <div className="font-mono text-xs p-4 bg-slate-50 dark:bg-slate-700 rounded overflow-x-auto">
                    <div className="text-orange-600 mb-2">// 每一步循环</div>
                    <div>(s, a) → Environment → (r, s')</div>
                    <div className="text-green-600 mt-2">// Direct RL</div>
                    <div>Q[s,a] += α(r + γ max Q[s'] - Q[s,a])</div>
                    <div className="text-green-600 mt-2">// Model Learning</div>
                    <div>Model[s,a] = (r, s')</div>
                    <div className="text-green-600 mt-2">// Planning (n 步)</div>
                    <div className="ml-2">for i in range(n):</div>
                    <div className="ml-4">s, a = random_visited_pair()</div>
                    <div className="ml-4">r, s' = Model[s,a]</div>
                    <div className="ml-4">Q[s,a] += α(r + γ max Q[s'] - Q[s,a])</div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">Planning Steps 选择</h4>
                <div className="grid grid-cols-3 gap-4 text-sm text-center">
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded">
                        <div className="text-2xl mb-2">⚠️</div>
                        <div className="font-bold">n 太小</div>
                        <div className="text-xs mt-2">接近 model-free<br />提升有限</div>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border-2 border-green-500">
                        <div className="text-2xl mb-2">✅</div>
                        <div className="font-bold">n 适中 (5-50)</div>
                        <div className="text-xs mt-2">平衡效率<br />与计算</div>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded">
                        <div className="text-2xl mb-2">⚠️</div>
                        <div className="font-bold">n 太大</div>
                        <div className="text-xs mt-2">计算开销大<br />模型误差累积</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

"use client";

export function WorldModelVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    世界模型可视化
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    在潜在空间学习与规划
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">World Model 架构 (Ha & Schmidhuber, 2018)</h4>
                <div className="space-y-4">
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-teal-500 text-white rounded-full flex items-center justify-center font-bold">V</div>
                            <strong>VAE (Vision Model)</strong>
                        </div>
                        <div className="text-sm ml-13 space-y-1">
                            <div>• 编码器: 图像 o<sub>t</sub> → 潜在编码 z<sub>t</sub></div>
                            <div>• 解码器: z<sub>t</sub> → 重建图像 ô<sub>t</sub></div>
                            <div>• 压缩高维观察到低维潜在空间</div>
                        </div>
                    </div>

                    <div className="flex justify-center">
                        <div className="text-2xl">↓</div>
                    </div>

                    <div className="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">M</div>
                            <strong>MDN-RNN (Memory Model)</strong>
                        </div>
                        <div className="text-sm ml-13 space-y-1">
                            <div>• 输入: z<sub>t</sub>, a<sub>t</sub></div>
                            <div>• 输出: z<sub>t+1</sub> 的分布 (混合高斯)</div>
                            <div>• 预测下一个潜在状态</div>
                        </div>
                    </div>

                    <div className="flex justify-center">
                        <div className="text-2xl">↓</div>
                    </div>

                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">C</div>
                            <strong>Controller (Policy)</strong>
                        </div>
                        <div className="text-sm ml-13 space-y-1">
                            <div>• 输入: z<sub>t</sub> (潜在状态)</div>
                            <div>• 输出: a<sub>t</sub> (动作)</div>
                            <div>• <strong>仅在潜在空间训练</strong>，无需真实环境</div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-teal-600 mb-4">训练阶段</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>1. 收集数据:</strong><br />
                            用随机策略收集轨迹
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>2. 训练 VAE:</strong><br />
                            学习潜在表示 z
                        </div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">
                            <strong>3. 训练 RNN:</strong><br />
                            预测 z<sub>t+1</sub> 的转移
                        </div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg border-4 border-cyan-500">
                    <h4 className="text-lg font-bold text-cyan-600 mb-4">规划阶段</h4>
                    <div className="space-y-3 text-sm">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>4. 在梦中训练:</strong><br />
                            Controller 在 World Model 中想象
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>5. 进化策略:</strong><br />
                            用 CMA-ES 优化 Controller
                        </div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">
                            <strong>6. 零真实样本:</strong><br />
                            完全在想象中学习
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">World Model vs 传统 Model-based</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-slate-50 dark:bg-slate-700 rounded">
                        <strong>传统:</strong><br />
                        在原始状态空间建模<br />
                        高维、难训练
                    </div>
                    <div className="p-4 bg-teal-50 dark:bg-teal-900/20 rounded border-2 border-teal-500">
                        <strong>World Model:</strong><br />
                        在潜在空间建模<br />
                        低维、高效
                    </div>
                </div>
            </div>
        </div>
    );
}

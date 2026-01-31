"use client";

export function PEARLArchitecture() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-slate-900 dark:to-violet-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    PEARL 架构图
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                    Probabilistic Embeddings for Actor-Critic RL
                </p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">上下文编码流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                        <strong>输入:</strong> 上下文 C = (s,a,r,s&apos;), ...
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                        <strong>编码器:</strong> Context Encoder
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded">
                        <strong>采样:</strong> z ~ N(μ, σ²)
                    </div>
                    <div className="flex justify-center text-2xl">↓</div>
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded border-2 border-purple-500">
                        <strong>输出:</strong> 任务嵌入 z
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">条件策略</h4>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded font-mono">
                    π(a | s, z) 和 Q(s, a, z)
                </div>
                <div className="text-sm text-center mt-3 text-slate-600 dark:text-slate-400">
                    策略和Q函数都以任务嵌入z为条件
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">PEARL 优势</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ Off-Policy</strong><br />
                        样本效率高
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                        <strong className="text-green-700 dark:text-green-400">✅ 隐式推断</strong><br />
                        自动推断任务参数
                    </div>
                </div>
            </div>
        </div>
    );
}

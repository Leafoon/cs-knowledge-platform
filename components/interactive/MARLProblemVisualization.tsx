"use client";

export function MARLProblemVisualization() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-sky-50 to-blue-50 dark:from-slate-900 dark:to-sky-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    MARL 问题可视化
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">多智能体MDP</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded text-center">
                        <div className="text-2xl mb-2">🤖</div>
                        <strong>智能体 1</strong>
                        <div className="text-xs mt-2">观测 o¹, 动作 a¹</div>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded text-center">
                        <div className="text-2xl mb-2">🤖</div>
                        <strong>智能体 2</strong>
                        <div className="text-xs mt-2">观测 o², 动作 a²</div>
                    </div>
                    <div className="p-4 bg-sky-50 dark:bg-sky-900/20 rounded text-center">
                        <div className="text-2xl mb-2">🤖</div>
                        <strong>智能体 N</strong>
                        <div className="text-xs mt-2">观测 oᴺ, 动作 aᴺ</div>
                    </div>
                </div>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                <h4 className="text-lg font-bold mb-4">环境转移</h4>
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <div className="font-mono mb-2">P(s' | s, a¹, a², ..., aᴺ)</div>
                    <div className="text-xs mt-2 text-slate-600 dark:text-slate-400">
                        下一个状态取决于所有智能体的联合动作
                    </div>
                </div>
            </div>
        </div>
    );
}

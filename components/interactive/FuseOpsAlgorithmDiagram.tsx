"use client";

export function FuseOpsAlgorithmDiagram() {
    const phases = [
        { title: "遍历节点", icon: "🔍", color: "from-indigo-500 to-blue-500", items: ["遍历计算图节点", "分析数据依赖关系", "构建算子拓扑图", "标记融合候选项"] },
        { title: "检查模式", icon: "🧠", color: "from-purple-500 to-indigo-500", items: ["识别可融合算子对", "匹配融合模式模板", "检查约束条件", "评估内存收益"] },
        { title: "决定融合", icon: "⚙️", color: "from-blue-500 to-purple-500", items: ["计算并行度", "评估代价模型", "选择最优策略", "生成融合计划"] },
        { title: "执行变换", icon: "🚀", color: "from-pink-500 to-indigo-500", items: ["合并循环体", "消除中间缓冲", "优化内存访问", "生成融合内核"] },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">FuseOps 算法流程</h3>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                {phases.map((phase, i) => (
                    <div key={i} className="relative">
                        <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg h-full">
                            <div className={`w-14 h-14 mx-auto mb-3 rounded-full bg-gradient-to-r ${phase.color} flex items-center justify-center text-2xl shadow-lg`}>
                                {phase.icon}
                            </div>
                            <h4 className="text-base font-bold text-center text-slate-800 dark:text-slate-100 mb-1">步骤 {i + 1}</h4>
                            <h5 className="text-center text-indigo-600 dark:text-indigo-400 font-semibold mb-3 text-sm">{phase.title}</h5>
                            <div className="space-y-2">
                                {phase.items.map((item, j) => (
                                    <div key={j} className="flex items-center gap-2 text-xs">
                                        <div className="w-5 h-5 rounded-full bg-indigo-100 dark:bg-indigo-900/40 flex items-center justify-center text-[10px] font-bold text-indigo-600 dark:text-indigo-400 shrink-0">
                                            {j + 1}
                                        </div>
                                        <span className="text-slate-700 dark:text-slate-300">{item}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        {i < phases.length - 1 && (
                            <div className="hidden md:flex absolute top-1/2 -right-2 transform -translate-y-1/2 z-10">
                                <div className="w-5 h-5 bg-indigo-500 rounded-full flex items-center justify-center text-white text-xs shadow-lg">→</div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-4 shadow-lg">
                <div className="flex items-center gap-3 text-sm text-slate-600 dark:text-slate-400">
                    <span className="text-indigo-500">💡</span>
                    <span>FuseOps 通过<strong className="text-indigo-600 dark:text-indigo-400">模式匹配</strong>识别可融合算子，利用<strong className="text-purple-600 dark:text-purple-400">代价模型</strong>评估收益，最终生成高效<strong className="text-blue-600 dark:text-blue-400">融合内核</strong>。</span>
                </div>
            </div>
        </div>
    );
}

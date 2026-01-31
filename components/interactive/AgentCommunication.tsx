"use client";

export function AgentCommunication() {
    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-slate-900 dark:to-teal-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
                    智能体通信机制
                </h3>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-lg font-bold mb-4">通信流程</h4>
                <div className="space-y-3 text-sm">
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                        <div><strong>观测:</strong> 每个智能体观测环境</div>
                    </div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                        <div><strong>生成消息:</strong> m^i = encoder(o^i)</div>
                    </div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                        <div><strong>广播:</strong> 发送消息到其他智能体</div>
                    </div>
                    <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                        <div><strong>聚合:</strong> 接收并聚合消息</div>
                    </div>
                    <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded flex items-center gap-3">
                        <div className="flex-shrink-0 w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                        <div><strong>决策:</strong> π^i(o^i, messages)</div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-teal-600 mb-4">显式通信</h4>
                    <div className="text-sm space-y-2">
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">发送消息</div>
                        <div className="p-3 bg-teal-50 dark:bg-teal-900/20 rounded">CommNet</div>
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-cyan-600 mb-4">隐式协调</h4>
                    <div className="text-sm space-y-2">
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">观测他人</div>
                        <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded">历史动作</div>
                    </div>
                </div>
            </div>
        </div>
    );
}

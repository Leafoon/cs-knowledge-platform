"use client";

export function MLCLLMOverviewDiagram() {
    const stages = [
        { title: "模型导入", icon: "📥", color: "from-indigo-500 to-blue-500", items: ["HuggingFace模型", "PyTorch权重", "模型配置解析"] },
        { title: "量化压缩", icon: "🗜️", color: "from-blue-500 to-purple-500", items: ["INT4/INT8量化", "GPTQ/AWQ", "权重校准"] },
        { title: "编译优化", icon: "⚙️", color: "from-purple-500 to-indigo-500", items: ["算子融合", "内存优化", "TensorIR调度"] },
        { title: "部署运行", icon: "🚀", color: "from-indigo-500 to-blue-500", items: ["WebGPU/iOS/Android", "流式推理", "对话管理"] },
    ];

    const supported = [
        { name: "Llama", icon: "🦙" },
        { name: "Mistral", icon: "💨" },
        { name: "GPT-NeoX", icon: "🤖" },
        { name: "RedPajama", icon: "🔴" },
        { name: "ChatGLM", icon: "💬" },
        { name: "RWKV", icon: "⚡" },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">MLC-LLM 架构概览图</h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">大语言模型的通用部署框架</p>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {stages.map((stage, i) => (
                        <div key={i} className="relative">
                            <div className={`bg-gradient-to-br ${stage.color} rounded-xl p-4 text-white h-full`}>
                                <div className="text-3xl text-center mb-2">{stage.icon}</div>
                                <div className="text-lg font-bold text-center mb-3">{stage.title}</div>
                                <div className="space-y-1">
                                    {stage.items.map((item, j) => (
                                        <div key={j} className="text-xs bg-white/20 rounded px-2 py-1 text-center">
                                            {item}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            {i < stages.length - 1 && (
                                <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10 text-2xl text-indigo-500">
                                    →
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-4">🤖 支持的模型</h4>
                    <div className="grid grid-cols-2 gap-2">
                        {supported.map((m, i) => (
                            <div key={i} className="flex items-center gap-2 p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                                <span className="text-xl">{m.icon}</span>
                                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">{m.name}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-4">📱 部署平台</h4>
                    <div className="space-y-2">
                        {[
                            { platform: "WebGPU", desc: "浏览器端运行", icon: "🌐" },
                            { platform: "iOS (Metal)", desc: "iPhone/iPad原生", icon: "🍎" },
                            { platform: "Android (OpenCL)", desc: "安卓设备", icon: "🤖" },
                            { platform: "CLI (CUDA)", desc: "服务器端推理", icon: "💻" },
                        ].map((p, i) => (
                            <div key={i} className="flex items-center gap-3 p-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                                <span className="text-xl">{p.icon}</span>
                                <div>
                                    <div className="font-semibold text-sm text-slate-800 dark:text-slate-100">{p.platform}</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">{p.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-4 text-white text-center">
                <p className="text-sm">💡 MLC-LLM = TVM编译优化 + 量化压缩 + 多平台部署，实现LLM的端侧运行</p>
            </div>
        </div>
    );
}

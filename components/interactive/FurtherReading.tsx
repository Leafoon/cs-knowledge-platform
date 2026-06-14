"use client";

export function FurtherReading() {
    const papers = [
        { title: "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", authors: "Chen et al., OSDI 2018", desc: "TVM原始论文，介绍端到端编译优化框架", link: "#" },
        { title: "Relay: A High-Level IR for Deep Learning", authors: "Roesch et al., 2019", desc: "Relay高级中间表示的设计与实现", link: "#" },
        { title: "Ansor: Generating High-Performance Tensor Programs", authors: "Zheng et al., OSDI 2020", desc: "自动搜索张量程序的调度空间", link: "#" },
    ];

    const docs = [
        { title: "TVM 官方文档", desc: "完整的API参考与教程", icon: "📖" },
        { title: "TVM 教程 - TensorIR", desc: "张量IR入门与实践", icon: "📝" },
        { title: "MLC-LLM 文档", desc: "大语言模型部署指南", icon: "🤖" },
        { title: "TVMCon 会议视频", desc: "年度TVM开发者大会", icon: "🎥" },
    ];

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">扩展阅读推荐</h3>
                <p className="text-slate-600 dark:text-slate-400 text-sm">深入学习TVM编译器的核心论文与资源</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-4">📄 核心论文</h4>
                    <div className="space-y-4">
                        {papers.map((p, i) => (
                            <div key={i} className="border border-indigo-200 dark:border-indigo-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                                <div className="font-semibold text-slate-800 dark:text-slate-100 text-sm">{p.title}</div>
                                <div className="text-xs text-indigo-500 mt-1">{p.authors}</div>
                                <div className="text-xs text-slate-600 dark:text-slate-400 mt-2">{p.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg">
                    <h4 className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-4">🔗 学习资源</h4>
                    <div className="space-y-3">
                        {docs.map((d, i) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg hover:from-indigo-100 hover:to-purple-100 dark:hover:from-indigo-900/40 dark:hover:to-purple-900/40 transition-colors cursor-pointer">
                                <span className="text-2xl">{d.icon}</span>
                                <div>
                                    <div className="font-semibold text-slate-800 dark:text-slate-100 text-sm">{d.title}</div>
                                    <div className="text-xs text-slate-600 dark:text-slate-400">{d.desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-6 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-4 text-white text-center">
                <p className="text-sm">💡 建议学习路径：TVM论文 → 官方教程 → Relay/TensorIR → MLC-LLM部署</p>
            </div>
        </div>
    );
}

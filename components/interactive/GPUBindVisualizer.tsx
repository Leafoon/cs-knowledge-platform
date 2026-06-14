"use client";

export function GPUBindVisualizer() {
    const blocks = [
        { bx: 0, threads: [0, 1, 2, 3] },
        { bx: 1, threads: [0, 1, 2, 3] },
        { bx: 2, threads: [0, 1, 2, 3] },
    ];

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">GPU 绑定映射</h3>

            <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-lg mb-6">
                <h4 className="text-base font-bold text-indigo-600 dark:text-indigo-400 mb-4 text-center">blockIdx.x → i, threadIdx.x → j</h4>
                <div className="space-y-3">
                    {blocks.map((block) => (
                        <div key={block.bx} className="flex items-center gap-3">
                            <div className="w-28 shrink-0 text-right">
                                <span className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/40 rounded text-sm font-mono text-indigo-700 dark:text-indigo-300">
                                    blockIdx.x={block.bx}
                                </span>
                            </div>
                            <div className="flex gap-2 flex-1">
                                {block.threads.map((tx) => (
                                    <div key={tx} className="flex-1 p-3 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-center text-sm font-semibold shadow">
                                        <div className="text-xs opacity-75">threadIdx.x={tx}</div>
                                        <div className="font-mono">[{block.bx * 4 + tx}]</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="mt-4 text-center text-sm text-slate-600 dark:text-slate-400 font-mono">
                    i = blockIdx.x × blockDim.x + threadIdx.x
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-purple-600 dark:text-purple-400 mb-3">TVM 绑定方式</h4>
                    <div className="space-y-2">
                        {[
                            { bind: "blockIdx.x", desc: "数据的块维度 i" },
                            { bind: "threadIdx.x", desc: "数据的线程维度 j" },
                            { bind: "vthread", desc: "虚拟线程，映射到循环" },
                        ].map((item, i) => (
                            <div key={i} className="flex items-center gap-3 p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded">
                                <code className="px-2 py-1 bg-indigo-200 dark:bg-indigo-800 rounded text-xs font-mono text-indigo-700 dark:text-indigo-300">{item.bind}</code>
                                <span className="text-sm text-slate-700 dark:text-slate-300">{item.desc}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white dark:bg-slate-800 rounded-xl p-5 shadow-lg">
                    <h4 className="text-base font-bold text-blue-600 dark:text-blue-400 mb-3">TIR 示例</h4>
                    <pre className="text-xs bg-slate-900 text-green-400 p-4 rounded-lg overflow-x-auto">
{`@T.prim_func
def matmul(A, B, C):
    for bx in T.thread_binding(0, 4, "blockIdx.x"):
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            for k in range(128):
                C[bx, tx] += A[bx, k] * B[k, tx]`}
                    </pre>
                </div>
            </div>
        </div>
    );
}

"use client";

import { useState } from "react";

const targets = [
    { id: "llvm", name: "LLVM CPU", icon: "🖥️", desc: "x86/ARM CPU 后端", features: ["AVX/NEON 向量化", "循环展开", "指令选择"] },
    { id: "cuda", name: "CUDA GPU", icon: "🎮", desc: "NVIDIA GPU 后端", features: ["线程/块映射", "共享内存", "Tensor Core"] },
    { id: "opencl", name: "OpenCL", icon: "📱", desc: "异构计算后端", features: ["设备适配", "工作组大小", "内存层次"] },
];

const integrationSteps = [
    { step: 1, title: "定义搜索空间", desc: "通过 Schedule 模板定义可调参数", icon: "📐" },
    { step: 2, title: "特征提取", desc: "提取计算图结构特征和硬件特征", icon: "🔍" },
    { step: 3, title: "成本模型", desc: "基于 XGBoost/LSTM 预测执行时间", icon: "🤖" },
    { step: 4, title: "搜索策略", desc: "模拟退火/遗传算法搜索最优配置", icon: "🔎" },
    { step: 5, title: "验证与缓存", desc: "远程编译验证并缓存最优调度", icon: "✅" },
];

export function AutoTVMTargetIntegration() {
    const [selectedTarget, setSelectedTarget] = useState("llvm");

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                AutoTVM 与 Target 系统集成
            </h2>
            <p className="text-center text-sm text-slate-500 dark:text-slate-400 mb-6">
                AutoTVM 如何根据目标硬件自动搜索最优调度
            </p>

            {/* Target selector */}
            <div className="flex justify-center gap-4 mb-6 flex-wrap">
                {targets.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setSelectedTarget(t.id)}
                        className={`flex flex-col items-center px-5 py-3 rounded-xl border-2 transition-all duration-300 min-w-[130px] ${
                            selectedTarget === t.id
                                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 shadow-lg"
                                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
                        }`}
                    >
                        <span className="text-2xl mb-1">{t.icon}</span>
                        <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{t.name}</span>
                        <span className="text-xs text-slate-500 dark:text-slate-400">{t.desc}</span>
                    </button>
                ))}
            </div>

            {/* Target features */}
            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-6">
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">
                    {targets.find((t) => t.id === selectedTarget)?.icon} Target 特化优化
                </h3>
                <div className="flex flex-wrap gap-3">
                    {targets
                        .find((t) => t.id === selectedTarget)
                        ?.features.map((f, i) => (
                            <span key={i} className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-indigo-100 to-purple-100 dark:from-indigo-900/40 dark:to-purple-900/40 text-xs font-medium text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
                                {f}
                            </span>
                        ))}
                </div>
            </div>

            {/* Integration pipeline */}
            <div className="relative">
                <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300 dark:from-indigo-700 dark:via-purple-700 dark:to-pink-700 -translate-y-1/2 hidden md:block" />
                <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative z-10">
                    {integrationSteps.map((s) => (
                        <div key={s.step} className="flex flex-col items-center">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 text-white flex items-center justify-center text-sm font-bold shadow-lg mb-2">
                                {s.step}
                            </div>
                            <div className="bg-white dark:bg-slate-800/80 rounded-xl p-3 border border-slate-200 dark:border-slate-700 text-center w-full">
                                <span className="text-lg">{s.icon}</span>
                                <div className="text-xs font-bold text-slate-700 dark:text-slate-200 mt-1">{s.title}</div>
                                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">{s.desc}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Code example */}
            <div className="mt-6 bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400">
                <div className="text-slate-500 mb-2"># AutoTVM 调优示例</div>
                <pre>{`@autotvm.template("matmul")
def matmul(N, M, K):
    cfg = tvm.autotvm.get_config()
    cfg.define_knob("tile_x", [16, 32, 64])
    cfg.define_knob("tile_y", [16, 32, 64])
    # ... 调度模板定义

task = autotvm.task.create("matmul", args=(1024,1024,1024), target="${selectedTarget}")
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=100)`}</pre>
            </div>
        </div>
    );
}

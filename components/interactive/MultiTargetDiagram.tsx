"use client";

import { useState } from "react";

const targets = [
  { id: "llvm", name: "LLVM (x86)", icon: "🖥️", color: "from-blue-500 to-indigo-500", perf: "100ms", code: `target = "llvm -mcpu=skylake"\nlib = tvm.build(mod, target)\n# x86 AVX2 向量化\n# 多核并行` },
  { id: "cuda", name: "CUDA (NVIDIA)", icon: "🎮", color: "from-indigo-500 to-purple-500", perf: "5ms", code: `target = "cuda"\nlib = tvm.build(mod, target)\n# Warp 并行\n# 共享内存 tiling` },
  { id: "rocm", name: "ROCm (AMD)", icon: "🔴", color: "from-purple-500 to-pink-500", perf: "7ms", code: `target = "rocm"\nlib = tvm.build(mod, target)\n# AMD GPU 后端\n# HIP 代码生成` },
  { id: "vulkan", name: "Vulkan", icon: "🌋", color: "from-pink-500 to-rose-500", perf: "15ms", code: `target = "vulkan"\nlib = tvm.build(mod, target)\n# SPIR-V 代码\n# 跨平台 GPU` },
  { id: "webgpu", name: "WebGPU", icon: "🌐", color: "from-rose-500 to-red-500", perf: "30ms", code: `target = "webgpu"\nlib = tvm.build(mod, target)\n# WGSL 着色器\n# 浏览器部署` },
];

export function MultiTargetDiagram() {
  const [active, setActive] = useState("cuda");

  const target = targets.find((t) => t.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">多 Target 编译</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">同一模型编译到不同硬件 Target</p>

      <div className="flex items-center justify-center mb-6">
        <div className="bg-white dark:bg-slate-800 px-5 py-3 rounded-xl border border-slate-200 dark:border-slate-700">
          <span className="text-lg">📦</span>
          <span className="text-sm font-bold text-slate-700 dark:text-slate-200 ml-2">统一 IR Module</span>
        </div>
      </div>

      <div className="flex items-center justify-center gap-1 mb-2">
        <div className="w-0.5 h-4 bg-indigo-300" />
      </div>

      <div className="grid grid-cols-5 gap-2 mb-6">
        {targets.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`flex flex-col items-center p-3 rounded-xl border-2 transition-all duration-300 ${
              active === t.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span className="text-2xl mb-1">{t.icon}</span>
            <span className="text-xs font-bold text-slate-700 dark:text-slate-200 text-center">{t.name}</span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">{t.perf}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-2">
            {target.icon} {target.name} 编译
          </h4>
          <div className="space-y-2">
            {targets.filter((t) => t.id !== active).map((t) => (
              <div key={t.id} className="flex items-center gap-2 text-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-600" />
                <span className="text-slate-500 dark:text-slate-400">{t.name}</span>
                <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">{t.perf}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
            <div className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
              选定延迟: {target.perf}
            </div>
          </div>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{target.code}</pre>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-3">
        {[
          { title: "代码复用", desc: "同一个 IR，不同后端" },
          { title: "自动调优", desc: "每个 Target 独立 AutoTVM" },
          { title: "统一 API", desc: "tvm.build(mod, target)" },
        ].map((f, i) => (
          <div key={i} className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
            <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{f.title}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

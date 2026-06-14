"use client";

import { useState } from "react";

const devices = [
  {
    id: "cpu",
    name: "CPU",
    icon: "🖥️",
    color: "from-blue-500 to-indigo-500",
    features: ["x86/ARM/RISC-V", "AVX2/NEON 向量化", "多线程并行", "LLVM 后端"],
    config: `target = "llvm -mcpu=skylake"\nmod = tvm.IRModule.from_expr(func)\nlib = tvm.build(mod, target)`,
    runtime: "CPU Runtime (~100ms)",
  },
  {
    id: "gpu",
    name: "GPU",
    icon: "🎮",
    color: "from-indigo-500 to-purple-500",
    features: ["CUDA/ROCm/OpenCL", "Warp 级并行", "共享内存优化", "Tensor Core 支持"],
    config: `target = "cuda"\nmod = tvm.IRModule.from_expr(func)\nlib = tvm.build(mod, target)`,
    runtime: "GPU Runtime (~5ms)",
  },
  {
    id: "fpga",
    name: "FPGA",
    icon: "⚡",
    color: "from-purple-500 to-pink-500",
    features: ["Vitis HLS 后端", "流水线并行", "自定义数据流", "低延迟推理"],
    config: `target = "sdaccel"\nmod = tvm.IRModule.from_expr(func)\nlib = tvm.build(mod, target)`,
    runtime: "FPGA Runtime (~1ms)",
  },
  {
    id: "mcu",
    name: "MCU",
    icon: "📡",
    color: "from-pink-500 to-rose-500",
    features: ["ARM Cortex-M", "CMSIS-NN 加速", "极小内存 footprint", "裸机部署"],
    config: `target = "c -runtime=c"\nmod = tvm.IRModule.from_expr(func)\nlib = tvm.build(mod, target)`,
    runtime: "MCU Runtime (~200ms)",
  },
];

export function MultiDeviceDeployment() {
  const [active, setActive] = useState("cpu");

  const device = devices.find((d) => d.id === active)!;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">多设备部署</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">同一模型部署到 CPU / GPU / FPGA / MCU 四种目标</p>

      <div className="grid grid-cols-4 gap-3 mb-6">
        {devices.map((d) => (
          <button
            key={d.id}
            onClick={() => setActive(d.id)}
            className={`flex flex-col items-center p-4 rounded-xl border-2 transition-all duration-300 ${
              active === d.id
                ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
            }`}
          >
            <span className="text-3xl mb-2">{d.icon}</span>
            <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{d.name}</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">{d.runtime}</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-3">
            {device.icon} {device.name} 特性
          </h4>
          <ul className="space-y-2">
            {device.features.map((f, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                {f}
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <div className="text-slate-500 mb-2"># 编译配置</div>
          <pre>{device.config}</pre>
        </div>
      </div>

      <div className="mt-5 bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">统一部署流程</h4>
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {["模型定义", "统一 IR", "Target 优化", "代码生成", "设备部署"].map((step, i) => (
            <div key={i} className="flex items-center">
              <span className={`px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r ${device.color} text-white`}>
                {step}
              </span>
              {i < 4 && <span className="mx-1 text-indigo-400">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

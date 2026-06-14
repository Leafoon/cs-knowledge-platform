"use client";

import { useState } from "react";

const plugins = [
  { name: "conv2d", device: "cuda", kernel: "conv2d_cuda_kernel", status: "registered" },
  { name: "dense", device: "cuda", kernel: "dense_cuda_kernel", status: "registered" },
  { name: "softmax", device: "cuda", kernel: "softmax_cuda_kernel", status: "registered" },
  { name: "conv2d", device: "cpu", kernel: "conv2d_llvm_kernel", status: "registered" },
  { name: "batch_norm", device: "cuda", kernel: null, status: "fallback" },
];

export function ExecutorExtensionDiagram() {
  const [activePlugin, setActivePlugin] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        执行器扩展: 插件式算子注册
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        通过注册表机制，运行时可以动态查找和调用特定设备的算子实现
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            算子注册表
          </h4>
          <div className="space-y-2">
            {plugins.map((p, i) => (
              <button
                key={`${p.name}-${p.device}`}
                onClick={() => setActivePlugin(activePlugin === i ? null : i)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all border ${
                  activePlugin === i
                    ? "bg-white dark:bg-slate-800 shadow-md border-indigo-300 dark:border-indigo-700"
                    : "bg-white/50 dark:bg-slate-800/50 border-transparent hover:border-slate-300"
                }`}
              >
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold text-white ${
                  p.device === "cuda" ? "bg-green-500" : "bg-blue-500"
                }`}>
                  {p.device}
                </span>
                <span className="font-mono text-sm text-slate-800 dark:text-slate-200">
                  {p.name}
                </span>
                <span className="ml-auto text-xs text-slate-400">
                  {p.status === "registered" ? "✓" : "⚠ fallback"}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            注册代码
          </h4>
          <pre className="bg-slate-900 text-green-400 p-4 rounded-xl text-xs font-mono">
{`// 注册 CUDA conv2d 实现
TVM_REGISTER_OP("nn.conv2d")
  .set_attr("FTVMCompute",
    Conv2DCompute, kCUDA);

// 注册 CPU 实现
TVM_REGISTER_OP("nn.conv2d")
  .set_attr("FTVMCompute",
    Conv2DCompute, kCPU);

// 运行时查找
auto kernel = GetKernel(
  "nn.conv2d", device_type);`}
          </pre>

          {activePlugin !== null && (
            <div className="mt-3 p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <p className="text-xs text-indigo-700 dark:text-indigo-300">
                <strong>{plugins[activePlugin].name}</strong> @ {plugins[activePlugin].device}:
                {plugins[activePlugin].status === "registered"
                  ? ` 使用 ${plugins[activePlugin].kernel}`
                  : " 回退到通用实现"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

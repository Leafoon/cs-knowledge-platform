"use client";

import { useState } from "react";

const devices = [
  { name: "CPU", color: "from-blue-500 to-indigo-600", alloc: "malloc/free", copy: "memcpy", sync: "无 (同步)", stream: "不支持", special: "NUMA 感知" },
  { name: "CUDA", color: "from-green-500 to-emerald-600", alloc: "cudaMalloc/cudaFree", copy: "cudaMemcpyAsync", sync: "cudaStreamSynchronize", stream: "cudaStream_t", special: "Unified Memory" },
  { name: "OpenCL", color: "from-orange-500 to-red-500", alloc: "clCreateBuffer", copy: "clEnqueueRead/Write", sync: "clFinish", stream: "cl_command_queue", special: "SVM 支持" },
  { name: "Metal", color: "from-purple-500 to-pink-600", alloc: "newBufferWithLength", copy: "copyFromBuffer", sync: "waitUntilCompleted", stream: "MTLCommandBuffer", special: "Shared Memory" },
];

const apis = ["AllocMemory", "FreeMemory", "CopyDataFromTo", "StreamSync", "SetDevice"];

export function DeviceAPITable() {
  const [highlight, setHighlight] = useState<number | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        DeviceAPI 接口对比
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-slate-300 dark:border-slate-600">
              <th className="text-left p-3 text-slate-700 dark:text-slate-300 font-bold">
                API 方法
              </th>
              {devices.map((d, i) => (
                <th
                  key={d.name}
                  className={`p-3 cursor-pointer transition-all ${
                    highlight === i ? "bg-indigo-100 dark:bg-indigo-900/40" : ""
                  }`}
                  onMouseEnter={() => setHighlight(i)}
                  onMouseLeave={() => setHighlight(null)}
                >
                  <span
                    className={`px-2 py-1 rounded text-xs font-bold text-white bg-gradient-to-r ${d.color}`}
                  >
                    {d.name}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-mono text-xs text-violet-600 dark:text-violet-400">
                AllocMemory
              </td>
              {devices.map((d) => (
                <td key={d.name} className="p-3 text-xs text-slate-600 dark:text-slate-400 font-mono">
                  {d.alloc}
                </td>
              ))}
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-mono text-xs text-violet-600 dark:text-violet-400">
                CopyDataFromTo
              </td>
              {devices.map((d) => (
                <td key={d.name} className="p-3 text-xs text-slate-600 dark:text-slate-400 font-mono">
                  {d.copy}
                </td>
              ))}
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-mono text-xs text-violet-600 dark:text-violet-400">
                StreamSync
              </td>
              {devices.map((d) => (
                <td key={d.name} className="p-3 text-xs text-slate-600 dark:text-slate-400 font-mono">
                  {d.sync}
                </td>
              ))}
            </tr>
            <tr className="border-b border-slate-200 dark:border-slate-700">
              <td className="p-3 font-mono text-xs text-violet-600 dark:text-violet-400">
                Stream
              </td>
              {devices.map((d) => (
                <td key={d.name} className="p-3 text-xs text-slate-600 dark:text-slate-400 font-mono">
                  {d.stream}
                </td>
              ))}
            </tr>
            <tr>
              <td className="p-3 font-mono text-xs text-violet-600 dark:text-violet-400">
                特性
              </td>
              {devices.map((d) => (
                <td key={d.name} className="p-3 text-xs text-amber-600 dark:text-amber-400">
                  {d.special}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

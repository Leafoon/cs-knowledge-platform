"use client";

import { useState } from "react";

export function NDArrayMemoryLayout() {
  const [hoverField, setHoverField] = useState<string | null>(null);

  const fields = [
    { name: "shape", value: "[2, 3, 4]", color: "from-violet-500 to-purple-600", desc: "张量各维度大小", offset: 0 },
    { name: "strides", value: "[12, 4, 1]", color: "from-indigo-500 to-blue-600", desc: "各维度步长 (元素数)", offset: 1 },
    { name: "data_ptr", value: "0x7f3a...", color: "from-blue-500 to-cyan-600", desc: "指向实际数据的指针", offset: 2 },
    { name: "device", value: "GPU:0", color: "from-cyan-500 to-teal-600", desc: "数据所在的设备", offset: 3 },
    { name: "dtype", value: "float32", color: "from-teal-500 to-green-600", desc: "元素数据类型", offset: 4 },
    { name: "byte_offset", value: "0", color: "from-green-500 to-emerald-600", desc: "数据起始偏移量", offset: 5 },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
        NDArray 内存布局 (DLTensor)
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            DLTensor 结构体
          </h4>
          <div className="bg-slate-900 rounded-xl p-4 font-mono text-sm">
            <span className="text-purple-400">struct</span>{" "}
            <span className="text-green-400">DLTensor</span>{" "}
            <span className="text-slate-400">{"{"}</span>
            {fields.map((f) => (
              <div
                key={f.name}
                className={`pl-6 py-1 cursor-pointer transition-all rounded ${
                  hoverField === f.name ? "bg-slate-700/50" : ""
                }`}
                onMouseEnter={() => setHoverField(f.name)}
                onMouseLeave={() => setHoverField(null)}
              >
                <span className="text-blue-300">{f.name}</span>
                <span className="text-slate-500">{"  // "}{f.desc}</span>
              </div>
            ))}
            <span className="text-slate-400">{"}"}</span>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
            内存视图
          </h4>
          <div className="space-y-2">
            {fields.map((f) => (
              <div
                key={f.name}
                className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                  hoverField === f.name
                    ? "bg-white dark:bg-slate-800 shadow-md scale-[1.02]"
                    : "bg-white/50 dark:bg-slate-800/50"
                }`}
                onMouseEnter={() => setHoverField(f.name)}
                onMouseLeave={() => setHoverField(null)}
              >
                <div className={`w-24 text-center px-2 py-1 rounded bg-gradient-to-r ${f.color} text-white text-xs font-bold`}>
                  {f.name}
                </div>
                <span className="font-mono text-sm text-slate-700 dark:text-slate-300">
                  {f.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">
          shape=[2,3,4] 的 3D 张量内存布局
        </h4>
        <div className="flex flex-wrap gap-1">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className={`w-8 h-8 rounded flex items-center justify-center text-[10px] font-mono text-white ${
                i < 12 ? "bg-indigo-500" : "bg-purple-500"
              }`}
            >
              {i}
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
          行优先 (C-contiguous): strides=[12,4,1], 地址 = i*12 + j*4 + k*1
        </p>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from "react";

interface Scenario {
  title: string;
  problem: string;
  solution: string;
  code: string;
  safe: boolean;
}

const scenarios: Scenario[] = [
  {
    title: "多线程同时 AllocDataSpace",
    problem: "多个推理线程同时调用 cudaMalloc，可能导致内存碎片或分配冲突。",
    solution: "TVM 内部使用线程局部的内存池（MemoryPool），每个线程维护独立的 free list。",
    code: `// 线程安全的内存池\nvoid* MemoryPool::Alloc(size_t size) {\n  std::lock_guard<std::mutex> lock(mtx_);\n  // 从 free list 查找合适的块\n  auto it = free_list_.lower_bound(size);\n  if (it != free_list_.end()) {\n    void* ptr = it->second;\n    free_list_.erase(it);\n    return ptr;\n  }\n  return device_api_->AllocDataSpace(...);\n}`,
    safe: true,
  },
  {
    title: "主线程释放 + 工作线程读取",
    problem: "主线程释放 buffer 后，工作线程仍在使用（Use-After-Free）。",
    solution: "TVM 使用引用计数（Reference Counting）+ Stream Sync 确保安全释放。",
    code: `// 引用计数保护\nvoid NDArray::InternalDecRef() {\n  if (--ref_counter_ == 0) {\n    // 等待所有流完成\n    TVMStreamSync(ctx_.device_type,\n                   ctx_.device_id, stream_);\n    device_api_->FreeDataSpace(ctx_, dptr_);\n  }\n}`,
    safe: true,
  },
  {
    title: "跨设备 CopyDataFromTo",
    problem: "在不同 GPU 间传输数据时的竞态条件。",
    solution: "使用 Peer-to-Peer 访问 + 事件同步机制。",
    code: `// 跨设备拷贝 with 事件同步\ncudaEvent_t event;\ncudaEventCreate(&event);\ncudaMemcpyAsync(dst, src, size, kind, stream);\ncudaEventRecord(event, stream);\n// 在目标设备等待\ncudaStreamWaitEvent(dst_stream, event, 0);`,
    safe: true,
  },
];

const DeviceAPIThreadSafetyDiagram = () => {
  const [active, setActive] = useState(0);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-indigo-950/80 via-purple-950/60 to-blue-950/80 backdrop-blur rounded-xl border border-indigo-700/30 shadow-lg my-6">
      <h3 className="text-xl font-bold mb-2 text-indigo-200">DeviceAPI 线程安全模型</h3>
      <p className="text-sm text-indigo-300/70 mb-4">多线程环境下 DeviceAPI 的并发安全问题与解决方案。</p>

      <div className="flex gap-2 mb-4">
        {scenarios.map((s, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              active === i
                ? "bg-indigo-600 text-white border-indigo-400"
                : "bg-indigo-900/40 text-indigo-300 border-indigo-700/50 hover:bg-indigo-800/60"
            }`}
          >
            {s.title}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-red-950/30 rounded-lg p-4 border border-red-800/30">
            <div className="text-xs text-red-400 font-semibold mb-2">⚠️ 问题</div>
            <div className="text-sm text-red-200/80">{scenarios[active].problem}</div>
          </div>
          <div className="bg-green-950/30 rounded-lg p-4 border border-green-800/30">
            <div className="text-xs text-green-400 font-semibold mb-2">✅ 解决方案</div>
            <div className="text-sm text-green-200/80">{scenarios[active].solution}</div>
          </div>
        </div>

        <div className="bg-indigo-950/60 rounded-lg p-4 border border-indigo-800/40">
          <div className="text-xs text-indigo-400 mb-2">C++ 实现</div>
          <pre className="text-xs text-cyan-300/80 font-mono whitespace-pre-wrap overflow-x-auto">{scenarios[active].code}</pre>
        </div>

        <div className="flex items-center gap-3 text-xs text-indigo-300/70">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-400" />
            <span>线程安全机制</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <span>潜在竞态条件</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-yellow-400" />
            <span>需要显式同步</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeviceAPIThreadSafetyDiagram;

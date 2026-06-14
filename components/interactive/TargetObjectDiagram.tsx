'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

export default function TargetObjectDiagram() {
  const [selectedField, setSelectedField] = useState<string | null>(null);

  const objectFields = [
    {
      id: 'kind',
      name: 'kind',
      type: 'TargetKind',
      desc: 'Target类型标识，包含名称、设备类型和默认属性',
      color: 'from-blue-500 to-indigo-600',
      details: [
        'name: "cuda" / "llvm" / "opencl"',
        'device_type: kDLCPU / kDLGPU / ...',
        'default_keys: 默认匹配标签集',
      ],
      example: '{ name: "cuda", device_type: kDLGPU }',
    },
    {
      id: 'keys',
      name: 'keys',
      type: 'Array<String>',
      desc: '用于匹配调度策略和优化Pass的标签集合',
      color: 'from-purple-500 to-violet-600',
      details: [
        '标识目标能力（如 "gpu", "cpu", "arm"）',
        '调度注册通过 key 匹配',
        '一个 Target 可有多个 keys',
      ],
      example: '["gpu", "cuda", "tensorcore"]',
    },
    {
      id: 'attrs',
      name: 'attrs',
      type: 'Map<String, Value>',
      desc: '硬件特定属性键值对，影响优化决策和代码生成',
      color: 'from-indigo-500 to-blue-600',
      details: [
        'arch: 硬件架构版本',
        'max_threads_per_block: 线程上限',
        'vector_width: 向量宽度',
        '自定义属性可扩展',
      ],
      example: '{ arch: "sm_86", max_threads: 1024 }',
    },
  ];

  const fullObject = {
    'target': 'cuda -arch=sm_86',
    'kind.name': 'cuda',
    'kind.device_type': 2,
    'keys': ['gpu', 'cuda'],
    'attrs.arch': 'sm_86',
    'attrs.max_shared_memory_per_block': 49152,
    'attrs.thread_warp_size': 32,
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        Target 对象结构图
      </h2>
      <p className="text-gray-400 text-center text-sm mb-8">
        Target 对象由 kind、keys、attrs 三部分组成
      </p>

      <div className="flex flex-col items-center mb-8">
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl border-2 border-indigo-500/30 p-6 w-full max-w-md shadow-lg shadow-indigo-500/10"
        >
          <div className="text-center mb-4">
            <span className="text-xs text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full">
              tvm.target.Target
            </span>
          </div>

          <div className="space-y-3">
            {objectFields.map((field) => (
              <motion.div
                key={field.id}
                whileHover={{ x: 4 }}
                onClick={() =>
                  setSelectedField(selectedField === field.id ? null : field.id)
                }
                className={`rounded-lg p-3 border cursor-pointer transition-all ${
                  selectedField === field.id
                    ? 'bg-gray-800 border-purple-500/40'
                    : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded bg-gradient-to-br ${field.color}`} />
                    <code className="text-white font-mono text-sm font-semibold">
                      {field.name}
                    </code>
                  </div>
                  <span className="text-xs text-gray-500 font-mono">{field.type}</span>
                </div>
                <p className="text-gray-400 text-xs mt-1">{field.desc}</p>

                {selectedField === field.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 pt-3 border-t border-gray-700"
                  >
                    <ul className="space-y-1.5 mb-3">
                      {field.details.map((d) => (
                        <li key={d} className="text-xs text-gray-400 flex items-start gap-2">
                          <span className="text-purple-400 mt-0.5">▸</span>
                          {d}
                        </li>
                      ))}
                    </ul>
                    <div className="bg-gray-900/60 rounded px-3 py-2">
                      <code className="text-xs font-mono text-emerald-400">{field.example}</code>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      <div className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700">
          <span className="text-sm font-medium text-gray-300">具体实例：CUDA Target</span>
        </div>
        <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {Object.entries(fullObject).map(([key, value], i) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="flex items-center gap-3 bg-gray-900/50 rounded-lg px-3 py-2"
            >
              <code className="text-purple-400 font-mono text-xs min-w-[160px]">{key}</code>
              <code className="text-emerald-400 font-mono text-xs">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </code>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="mt-6 text-center">
        <p className="text-gray-500 text-xs">
          Target 对象是不可变的（immutable），创建后属性不可修改
        </p>
      </div>
    </div>
  );
}

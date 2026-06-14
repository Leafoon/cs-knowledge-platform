'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const stages = [
  { id: 'frontend', name: '前端导入', icon: '📥' },
  { id: 'graph', name: '图优化', icon: '🔄' },
  { id: 'operator', name: '算子调度', icon: '⚙️' },
  { id: 'codegen', name: '代码生成', icon: '🔧' },
  { id: 'runtime', name: '运行时', icon: '🚀' },
];

const targets = [
  {
    id: 'cpu',
    name: 'CPU (x86)',
    color: 'from-blue-500 to-cyan-600',
    influences: {
      frontend: '无特殊影响，标准ONNX/PyTorch导入',
      graph: '算子融合策略偏向CPU友好的模式，减少内存带宽需求',
      operator: '使用AVX/SSE向量化，多线程Tiling，缓存友好调度',
      codegen: '生成LLVM IR，x86特有指令集优化（BMI, AES等）',
      runtime: 'pthread线程池，CPU内存分配器，NUMA感知',
    },
  },
  {
    id: 'cuda',
    name: 'NVIDIA GPU',
    color: 'from-green-500 to-emerald-600',
    influences: {
      frontend: '识别可并行的算子，标记GPU适用的子图',
      graph: 'GPU友好的融合模式，消除不必要的CPU-GPU数据拷贝',
      operator: 'CUDA Thread/Block层次调度，Shared Memory利用，Warp级别优化',
      codegen: '生成PTX或CUDA Kernel代码，register分配优化',
      runtime: 'CUDA Stream管理，GPU显存池分配器，异步执行',
    },
  },
  {
    id: 'arm',
    name: 'ARM (移动端)',
    color: 'from-orange-500 to-amber-600',
    influences: {
      frontend: '模型量化感知导入，支持int8/uint8输入',
      graph: '针对移动端的算子裁剪与融合，减少算子数量',
      operator: 'NEON向量化（float32x4），int8量化调度，Winograd变换',
      codegen: '生成ARM汇编或LLVM IR（aarch64 target）',
      runtime: '轻量级运行时（~100KB），内存受限分配策略',
    },
  },
  {
    id: 'vulkan',
    name: 'Vulkan (通用GPU)',
    color: 'from-purple-500 to-violet-600',
    influences: {
      frontend: '标准导入，无特殊处理',
      graph: 'SPIR-V友好的图变换，workgroup大小适配',
      operator: 'Vulkan Compute Shader调度，local_size优化',
      codegen: '生成SPIR-V字节码，支持descriptor set绑定',
      runtime: 'Vulkan Command Buffer管理，device memory分配',
    },
  },
];

export default function TargetInfluenceDiagram() {
  const [selectedTarget, setSelectedTarget] = useState(0);
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  const currentTarget = targets[selectedTarget];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        Target 如何影响各编译阶段
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        不同硬件目标在每个编译阶段产生不同的优化决策
      </p>

      <div className="flex gap-3 justify-center mb-8">
        {targets.map((t, i) => (
          <button
            key={t.id}
            onClick={() => { setSelectedTarget(i); setSelectedStage(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              i === selectedTarget
                ? `bg-gradient-to-r ${t.color} text-white shadow-lg`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <div className="relative">
        <div className="flex items-center justify-between gap-2">
          {stages.map((stage, i) => (
            <React.Fragment key={stage.id}>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedStage(i === selectedStage ? null : i)}
                className={`flex-1 relative rounded-xl p-4 border transition-all cursor-pointer ${
                  i === selectedStage
                    ? 'bg-gray-800 border-purple-500/50 shadow-lg shadow-purple-500/10'
                    : 'bg-gray-800/60 border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="text-2xl mb-2">{stage.icon}</div>
                <div className="text-white text-sm font-medium">{stage.name}</div>
                {i === selectedStage && (
                  <motion.div
                    layoutId="stage-glow"
                    className="absolute inset-0 rounded-xl border-2 border-purple-400/30"
                    transition={{ type: 'spring', bounce: 0.2 }}
                  />
                )}
              </motion.button>
              {i < stages.length - 1 && (
                <div className="text-gray-600 flex-shrink-0">→</div>
              )}
            </React.Fragment>
          ))}
        </div>

        {selectedStage !== null && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 bg-gray-800/80 rounded-xl p-5 border border-purple-500/20"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">{stages[selectedStage].icon}</span>
              <span className="text-white font-medium">{stages[selectedStage].name}</span>
              <span className="text-gray-500 mx-2">→</span>
              <span className={`bg-gradient-to-r ${currentTarget.color} bg-clip-text text-transparent font-medium`}>
                {currentTarget.name}
              </span>
            </div>
            <p className="text-gray-300 text-sm">
              {currentTarget.influences[stages[selectedStage].id as keyof typeof currentTarget.influences]}
            </p>
          </motion.div>
        )}
      </div>

      <div className="mt-8 bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700">
          <span className="text-sm font-medium text-gray-300">
            {currentTarget.name} 对各阶段的影响一览
          </span>
        </div>
        <div className="divide-y divide-gray-700/50">
          {stages.map((stage, i) => (
            <div key={stage.id} className="flex items-start gap-4 px-4 py-3">
              <div className="flex items-center gap-2 min-w-[120px]">
                <span>{stage.icon}</span>
                <span className="text-sm text-gray-300">{stage.name}</span>
              </div>
              <motion.p
                key={`${currentTarget.id}-${stage.id}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="text-sm text-gray-400 flex-1"
              >
                {currentTarget.influences[stage.id as keyof typeof currentTarget.influences]}
              </motion.p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

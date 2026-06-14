'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const tags = [
  {
    id: 'cuda-tensorcore',
    name: 'cuda -tensorcore',
    label: 'CUDA + TensorCore',
    properties: [
      { key: 'backend', value: 'cuda', source: 'TargetKind' },
      { key: 'arch', value: 'sm_75+', source: 'Tag' },
      { key: 'tensorcore', value: 'true', source: 'Tag' },
      { key: 'max_threads', value: '1024', source: 'TargetKind默认' },
      { key: 'warp_size', value: '32', source: 'TargetKind默认' },
    ],
    description: '启用Tensor Core加速矩阵运算，适用于Ampere/Turing架构',
    color: 'from-green-500 to-emerald-600',
  },
  {
    id: 'llvm-avx512',
    name: 'llvm -mattr=+avx512f',
    label: 'LLVM + AVX-512',
    properties: [
      { key: 'backend', value: 'llvm', source: 'TargetKind' },
      { key: 'mattr', value: '+avx512f', source: '用户指定' },
      { key: 'vector_width', value: '16', source: 'Tag推导' },
      { key: 'device_type', value: 'kDLCPU', source: 'TargetKind默认' },
    ],
    description: '启用AVX-512向量指令集，512位SIMD宽度',
    color: 'from-blue-500 to-indigo-600',
  },
  {
    id: 'arm-neon',
    name: 'llvm -device=arm_cpu -mattr=+neon',
    label: 'ARM + NEON',
    properties: [
      { key: 'backend', value: 'llvm', source: 'TargetKind' },
      { key: 'device', value: 'arm_cpu', source: 'Tag' },
      { key: 'mattr', value: '+neon', source: '用户指定' },
      { key: 'vector_width', value: '4', source: 'Tag推导' },
      { key: 'mtriple', value: 'aarch64', source: 'Tag' },
    ],
    description: 'ARM平台NEON SIMD优化，适合移动端部署',
    color: 'from-orange-500 to-amber-600',
  },
  {
    id: 'vulkan-spirv',
    name: 'vulkan -from_device=0',
    label: 'Vulkan + SPIR-V',
    properties: [
      { key: 'backend', value: 'vulkan', source: 'TargetKind' },
      { key: 'from_device', value: '0', source: '用户指定' },
      { key: 'max_threads', value: '256', source: 'TargetKind默认' },
      { key: 'supports_float32', value: 'true', source: '运行时查询' },
    ],
    description: 'Vulkan计算着色器，跨平台GPU通用目标',
    color: 'from-purple-500 to-violet-600',
  },
];

const sourceColors: Record<string, string> = {
  'TargetKind': 'text-blue-400',
  'Tag': 'text-purple-400',
  '用户指定': 'text-emerald-400',
  'TargetKind默认': 'text-gray-400',
  'Tag推导': 'text-amber-400',
  '运行时查询': 'text-cyan-400',
};

export default function TargetTagSystem() {
  const [selectedTag, setSelectedTag] = useState(0);
  const tag = tags[selectedTag];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        Target 标签系统
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        Tag 如何组合多个属性形成完整 Target 配置
      </p>

      <div className="flex flex-wrap gap-3 justify-center mb-8">
        {tags.map((t, i) => (
          <button
            key={t.id}
            onClick={() => setSelectedTag(i)}
            className={`px-4 py-2 rounded-lg text-sm font-mono transition-all ${
              i === selectedTag
                ? `bg-gradient-to-r ${t.color} text-white shadow-lg`
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedTag}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <div className="bg-gray-800/40 rounded-xl border border-gray-700 p-6 mb-6">
            <div className="text-center mb-4">
              <span className="text-lg font-semibold text-white">{tag.label}</span>
            </div>
            <p className="text-gray-400 text-sm text-center mb-6">{tag.description}</p>

            <div className="flex items-center justify-center gap-3 flex-wrap">
              <div className="bg-blue-900/40 rounded-lg px-4 py-2 border border-blue-500/30">
                <div className="text-blue-400 text-xs font-medium mb-1">TargetKind</div>
                <code className="text-white text-sm font-mono">{tag.properties[0].value}</code>
              </div>
              <span className="text-gray-600 text-xl">+</span>
              {tag.properties.filter(p => p.source !== 'TargetKind').slice(0, 2).map((p) => (
                <React.Fragment key={p.key}>
                  <div className="bg-purple-900/40 rounded-lg px-4 py-2 border border-purple-500/30">
                    <div className="text-purple-400 text-xs font-medium mb-1">{p.source}</div>
                    <code className="text-white text-sm font-mono">
                      {p.key}={p.value}
                    </code>
                  </div>
                  <span className="text-gray-600 text-xl">+</span>
                </React.Fragment>
              ))}
              <div className="bg-gray-700/50 rounded-lg px-4 py-2 border border-gray-600">
                <div className="text-gray-400 text-xs font-medium mb-1">合并</div>
                <span className="text-white text-sm">→ Target 对象</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
            <div className="px-4 py-3 bg-gray-800/80 border-b border-gray-700">
              <span className="text-sm font-medium text-gray-300">属性来源分析</span>
            </div>
            <div className="divide-y divide-gray-700/50">
              {tag.properties.map((prop, i) => (
                <motion.div
                  key={prop.key}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center px-4 py-3 gap-4"
                >
                  <code className="text-white font-mono text-sm min-w-[120px]">{prop.key}</code>
                  <code className="text-emerald-400 font-mono text-sm min-w-[80px]">{prop.value}</code>
                  <span className="text-gray-600 mx-2">←</span>
                  <span className={`text-xs font-medium ${sourceColors[prop.source] || 'text-gray-400'}`}>
                    {prop.source}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="mt-6 flex flex-wrap gap-4 justify-center">
        {Object.entries(sourceColors).map(([name, color]) => (
          <div key={name} className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${color.replace('text-', 'bg-')}`} />
            <span className={`text-xs ${color}`}>{name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

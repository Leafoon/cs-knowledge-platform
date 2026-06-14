'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const layers = [
  {
    id: 'user',
    name: '用户接口层',
    color: 'from-blue-500 to-cyan-600',
    borderColor: 'border-blue-500/30',
    components: [
      { name: 'Target 字符串', desc: '"cuda -arch=sm_86"' },
      { name: 'Target Python API', desc: 'tvm.target.Target(...)' },
      { name: 'Target 构建器', desc: 'tvm.target.Target.current()' },
    ],
    description: '用户通过字符串或API创建和管理Target对象',
  },
  {
    id: 'registry',
    name: '注册管理层',
    color: 'from-indigo-500 to-blue-600',
    borderColor: 'border-indigo-500/30',
    components: [
      { name: 'TargetKindRegistry', desc: '全局注册表单例' },
      { name: 'TargetKind', desc: '目标类型定义' },
      { name: 'TargetTag', desc: '预定义标签组合' },
    ],
    description: '管理所有已注册的Target类型和标签',
  },
  {
    id: 'object',
    name: 'Target 对象层',
    color: 'from-purple-500 to-indigo-600',
    borderColor: 'border-purple-500/30',
    components: [
      { name: 'Target Object', desc: 'kind + keys + attrs' },
      { name: '属性查询', desc: 'target.attrs[...]' },
      { name: '设备信息', desc: 'target.kind.device_type' },
    ],
    description: '不可变的Target实例，封装了编译目标的全部信息',
  },
  {
    id: 'consumer',
    name: '编译消费层',
    color: 'from-violet-500 to-purple-600',
    borderColor: 'border-violet-500/30',
    components: [
      { name: 'Pass 系统', desc: '根据Target选择优化策略' },
      { name: '调度系统', desc: '匹配key选择调度方案' },
      { name: 'CodeGen', desc: '生成目标硬件代码' },
    ],
    description: '编译流程的各个组件消费Target信息做决策',
  },
];

export default function TargetSystemOverview() {
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [hoveredComp, setHoveredComp] = useState<string | null>(null);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        Target 系统层次概览
      </h2>
      <p className="text-gray-400 text-center text-sm mb-8">
        从用户接口到编译消费的四层架构
      </p>

      <div className="space-y-3">
        {layers.map((layer, layerIdx) => (
          <motion.div
            key={layer.id}
            layout
            onClick={() =>
              setSelectedLayer(selectedLayer === layerIdx ? null : layerIdx)
            }
            className={`rounded-xl border cursor-pointer transition-all overflow-hidden ${
              selectedLayer === layerIdx
                ? `${layer.borderColor} bg-gray-800/80 shadow-lg`
                : 'border-gray-700 bg-gray-800/40 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between px-5 py-4">
              <div className="flex items-center gap-3">
                <div
                  className={`w-8 h-8 rounded-lg bg-gradient-to-br ${layer.color} flex items-center justify-center text-white text-sm font-bold`}
                >
                  {layerIdx + 1}
                </div>
                <div>
                  <h3 className="text-white font-medium text-sm">{layer.name}</h3>
                  <p className="text-gray-500 text-xs">{layer.description}</p>
                </div>
              </div>
              <motion.span
                animate={{ rotate: selectedLayer === layerIdx ? 180 : 0 }}
                className="text-gray-500"
              >
                ▾
              </motion.span>
            </div>

            <AnimatePresence>
              {selectedLayer === layerIdx && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="border-t border-gray-700/50"
                >
                  <div className="p-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {layer.components.map((comp, i) => (
                      <motion.div
                        key={comp.name}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        onMouseEnter={() => setHoveredComp(`${layerIdx}-${i}`)}
                        onMouseLeave={() => setHoveredComp(null)}
                        className={`bg-gray-900/60 rounded-lg p-3 border transition-all ${
                          hoveredComp === `${layerIdx}-${i}`
                            ? 'border-purple-500/30'
                            : 'border-gray-700/50'
                        }`}
                      >
                        <div className="text-white text-sm font-medium mb-1">
                          {comp.name}
                        </div>
                        <code className="text-gray-400 text-xs">{comp.desc}</code>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      <div className="flex justify-center my-4">
        <div className="flex flex-col items-center gap-1 text-gray-600 text-xs">
          <span>数据流向</span>
          <div className="flex flex-col items-center">
            <span>↓</span>
            <span>↓</span>
            <span>↓</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2">
        {layers.map((layer) => (
          <div key={layer.id} className="text-center">
            <div className={`h-1 rounded-full bg-gradient-to-r ${layer.color} mb-2`} />
            <span className="text-xs text-gray-500">{layer.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

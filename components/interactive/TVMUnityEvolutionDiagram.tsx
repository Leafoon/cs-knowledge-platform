'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const versions = [
  {
    id: 'v0.6',
    name: 'v0.6',
    year: '2019',
    title: '传统分离架构',
    description: 'Relay IR与TE（Tensor Expression）分离，前端与后端解耦不够紧密',
    features: [
      { name: 'Relay IR', desc: '高级图表示，支持控制流' },
      { name: 'TE调度', desc: '手动定义张量表达式与调度策略' },
      { name: 'AutoTVM', desc: '基于模板的自动调优' },
    ],
    limitations: [
      '前端与后端耦合松散',
      'Pass之间缺乏统一管理',
      '图优化与算子优化独立',
    ],
    color: 'from-slate-600 to-slate-700',
    accent: 'border-slate-500',
  },
  {
    id: 'v0.8',
    name: 'v0.8',
    year: '2020',
    title: 'Meta Schedule引入',
    description: '引入Meta Schedule框架，统一搜索空间与调度策略，开始向Unity架构演进',
    features: [
      { name: 'Meta Schedule', desc: '基于IR的程序调优框架' },
      { name: '统一搜索空间', desc: '自动化的调度空间定义' },
      { name: 'TOPI改进', desc: '算子模板库增强' },
    ],
    limitations: [
      'IR层面仍有割裂',
      '图级与算子级优化未统一',
    ],
    color: 'from-blue-600 to-indigo-700',
    accent: 'border-blue-500',
  },
  {
    id: 'v0.10',
    name: 'v0.10',
    year: '2022',
    title: 'Unity初期',
    description: '提出TVMScript，统一IR表示，图优化与调度开始融合',
    features: [
      { name: 'TVMScript', desc: 'Python嵌入式IR描述语言' },
      { name: 'TIR', desc: '底层张量IR统一表示' },
      { name: '统一Pass框架', desc: '结构化IR变换管理' },
    ],
    limitations: [
      '部分遗留代码未迁移',
      '生态工具链仍在完善',
    ],
    color: 'from-indigo-600 to-purple-700',
    accent: 'border-indigo-500',
  },
  {
    id: 'v0.14',
    name: 'v0.14',
    year: '2023',
    title: 'Unity成熟',
    description: '完整Unity架构：统一IR、统一Pass、Meta Schedule成熟，端到端优化',
    features: [
      { name: 'Relax IR', desc: '新一代高级IR，支持动态shape' },
      { name: '统一编译栈', desc: '图/算子/硬件全栈统一' },
      { name: 'End-to-End', desc: '从模型到部署的完整流水线' },
    ],
    limitations: [],
    color: 'from-purple-600 to-violet-700',
    accent: 'border-purple-500',
  },
];

export default function TVMUnityEvolutionDiagram() {
  const [selected, setSelected] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gray-900 rounded-2xl">
      <h2 className="text-2xl font-bold text-center mb-2 bg-gradient-to-r from-indigo-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
        TVM Unity 演进图
      </h2>
      <p className="text-gray-400 text-center text-sm mb-6">
        从分离架构到统一编译栈的演进历程
      </p>

      {/* Timeline */}
      <div className="flex items-center justify-between mb-8 px-4">
        {versions.map((v, i) => (
          <React.Fragment key={v.id}>
            <button
              onClick={() => setSelected(i)}
              className={`relative flex flex-col items-center transition-all duration-300 ${
                i === selected ? 'scale-110' : 'opacity-60 hover:opacity-80'
              }`}
            >
              <div
                className={`w-14 h-14 rounded-full bg-gradient-to-br ${v.color} flex items-center justify-center text-white font-bold text-sm border-2 ${
                  i === selected ? v.accent : 'border-transparent'
                } shadow-lg ${i === selected ? 'shadow-purple-500/30' : ''}`}
              >
                {v.name}
              </div>
              <span className="text-xs text-gray-400 mt-1">{v.year}</span>
              {i === selected && (
                <motion.div
                  layoutId="timeline-indicator"
                  className="absolute -bottom-3 w-2 h-2 bg-purple-400 rounded-full"
                />
              )}
            </button>
            {i < versions.length - 1 && (
              <div className="flex-1 h-0.5 mx-2 bg-gradient-to-r from-gray-700 via-gray-600 to-gray-700 relative overflow-hidden">
                <motion.div
                  className="absolute inset-y-0 left-0 bg-gradient-to-r from-indigo-500 to-purple-500"
                  initial={{ width: '0%' }}
                  animate={{ width: i < selected ? '100%' : '0%' }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          className="bg-gray-800/60 rounded-xl border border-gray-700 p-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <div
              className={`px-3 py-1 rounded-full bg-gradient-to-r ${versions[selected].color} text-white text-sm font-medium`}
            >
              {versions[selected].name}
            </div>
            <h3 className="text-xl font-semibold text-white">
              {versions[selected].title}
            </h3>
          </div>
          <p className="text-gray-300 mb-5">{versions[selected].description}</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
            {versions[selected].features.map((f, i) => (
              <motion.div
                key={f.name}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
                className="bg-gray-900/50 rounded-lg p-4 border border-indigo-500/20"
              >
                <div className="text-indigo-400 font-semibold text-sm mb-1">{f.name}</div>
                <div className="text-gray-400 text-xs">{f.desc}</div>
              </motion.div>
            ))}
          </div>

          {versions[selected].limitations.length > 0 && (
            <div className="bg-gray-900/40 rounded-lg p-4 border border-amber-500/20">
              <div className="text-amber-400 text-xs font-medium mb-2">局限性</div>
              <ul className="space-y-1">
                {versions[selected].limitations.map((l) => (
                  <li key={l} className="text-gray-400 text-xs flex items-start gap-2">
                    <span className="text-amber-500 mt-0.5">•</span>
                    {l}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Architecture comparison */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700">
          <div className="text-slate-400 text-xs font-medium mb-3 text-center">传统架构（v0.6）</div>
          <div className="flex flex-col gap-2">
            <div className="bg-slate-700/50 rounded px-3 py-2 text-center text-xs text-slate-300">Relay IR（图级）</div>
            <div className="text-center text-gray-600 text-xs">↕ 松耦合</div>
            <div className="bg-slate-700/50 rounded px-3 py-2 text-center text-xs text-slate-300">TE / TOPI（算子级）</div>
            <div className="text-center text-gray-600 text-xs">↕</div>
            <div className="bg-slate-700/50 rounded px-3 py-2 text-center text-xs text-slate-300">CodeGen（代码生成）</div>
          </div>
        </div>
        <div className="bg-gray-800/40 rounded-xl p-4 border border-purple-500/20">
          <div className="text-purple-400 text-xs font-medium mb-3 text-center">Unity架构（v0.14）</div>
          <div className="flex flex-col gap-2">
            <div className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded px-3 py-2 text-center text-xs text-purple-300">Relax IR（统一高级IR）</div>
            <div className="text-center text-purple-500 text-xs">↕ 紧耦合</div>
            <div className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded px-3 py-2 text-center text-xs text-purple-300">TIR + Meta Schedule</div>
            <div className="text-center text-purple-500 text-xs">↕</div>
            <div className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded px-3 py-2 text-center text-xs text-purple-300">统一 CodeGen</div>
          </div>
        </div>
      </div>
    </div>
  );
}

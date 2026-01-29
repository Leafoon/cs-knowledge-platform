"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type ChainType = 'sequential' | 'parallel' | 'router' | 'mapreduce';

type Edge = {
  from: string;
  to: string;
  label?: string;
};

type NodeType = 'input' | 'output' | 'process' | 'router' | 'merge' | 'split';

type Node = {
  id: string;
  label: string;
  type: NodeType;
};

type Architecture = {
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
  code: string;
};

const architectures: Record<ChainType, Architecture> = {
  sequential: {
    name: '顺序链',
    description: '串行执行多个步骤，前一步输出作为后一步输入',
    nodes: [
      { id: 'input', label: '输入数据', type: 'input' },
      { id: 'step1', label: 'Step 1: 翻译', type: 'process' },
      { id: 'step2', label: 'Step 2: 摘要', type: 'process' },
      { id: 'step3', label: 'Step 3: 关键词', type: 'process' },
      { id: 'output', label: '最终输出', type: 'output' }
    ],
    edges: [
      { from: 'input', to: 'step1' },
      { from: 'step1', to: 'step2' },
      { from: 'step2', to: 'step3' },
      { from: 'step3', to: 'output' }
    ],
    code: `chain = (
    translate_prompt | model
    | summary_prompt | model
    | keyword_prompt | model
)`
  },
  parallel: {
    name: '并行链',
    description: '同时执行多个独立任务，结果合并为字典',
    nodes: [
      { id: 'input', label: '输入数据', type: 'input' },
      { id: 'branch1', label: '翻译', type: 'process' },
      { id: 'branch2', label: '情感分析', type: 'process' },
      { id: 'branch3', label: '实体提取', type: 'process' },
      { id: 'merge', label: '结果合并', type: 'merge' },
      { id: 'output', label: '输出字典', type: 'output' }
    ],
    edges: [
      { from: 'input', to: 'branch1' },
      { from: 'input', to: 'branch2' },
      { from: 'input', to: 'branch3' },
      { from: 'branch1', to: 'merge' },
      { from: 'branch2', to: 'merge' },
      { from: 'branch3', to: 'merge' },
      { from: 'merge', to: 'output' }
    ],
    code: `chain = RunnableParallel(
    translation=translate_chain,
    sentiment=sentiment_chain,
    entities=entity_chain
)`
  },
  router: {
    name: '路由链',
    description: '根据条件动态选择执行路径',
    nodes: [
      { id: 'input', label: '输入数据', type: 'input' },
      { id: 'router', label: '路由决策', type: 'router' },
      { id: 'route1', label: '技术支持', type: 'process' },
      { id: 'route2', label: '账单查询', type: 'process' },
      { id: 'route3', label: '通用客服', type: 'process' },
      { id: 'output', label: '输出', type: 'output' }
    ],
    edges: [
      { from: 'input', to: 'router' },
      { from: 'router', to: 'route1', label: 'technical' },
      { from: 'router', to: 'route2', label: 'billing' },
      { from: 'router', to: 'route3', label: 'general' },
      { from: 'route1', to: 'output' },
      { from: 'route2', to: 'output' },
      { from: 'route3', to: 'output' }
    ],
    code: `chain = RunnableBranch(
    (lambda x: x["intent"] == "tech", tech_chain),
    (lambda x: x["intent"] == "bill", bill_chain),
    general_chain
)`
  },
  mapreduce: {
    name: 'Map-Reduce',
    description: '分布式处理大规模数据集合',
    nodes: [
      { id: 'input', label: '长文档', type: 'input' },
      { id: 'split', label: '文档分割', type: 'split' },
      { id: 'map1', label: 'Map: Chunk 1', type: 'process' },
      { id: 'map2', label: 'Map: Chunk 2', type: 'process' },
      { id: 'map3', label: 'Map: Chunk 3', type: 'process' },
      { id: 'reduce', label: 'Reduce: 合并', type: 'merge' },
      { id: 'output', label: '最终摘要', type: 'output' }
    ],
    edges: [
      { from: 'input', to: 'split' },
      { from: 'split', to: 'map1' },
      { from: 'split', to: 'map2' },
      { from: 'split', to: 'map3' },
      { from: 'map1', to: 'reduce' },
      { from: 'map2', to: 'reduce' },
      { from: 'map3', to: 'reduce' },
      { from: 'reduce', to: 'output' }
    ],
    code: `# Map阶段
summaries = map_chain.batch(chunks)

# Reduce阶段
final = reduce_chain.invoke({"summaries": summaries})`
  }
};

export default function ChainOrchestrationDiagram() {
  const [selectedType, setSelectedType] = useState<ChainType>('sequential');
  const architecture = architectures[selectedType];

  // 计算节点位置
  const getNodePosition = (nodeId: string, index: number, total: number) => {
    const arch = architectures[selectedType];
    const node = arch.nodes.find(n => n.id === nodeId);
    
    if (node?.type === 'input') return { x: 50, y: 200 };
    if (node?.type === 'output') return { x: 750, y: 200 };
    
    if (selectedType === 'sequential') {
      return { x: 150 + index * 150, y: 200 };
    } else if (selectedType === 'parallel') {
      if (node?.type === 'process') {
        return { x: 300, y: 80 + index * 100 };
      }
      if (node?.type === 'merge') return { x: 550, y: 200 };
    } else if (selectedType === 'router') {
      if (node?.type === 'router') return { x: 250, y: 200 };
      if (node?.type === 'process') {
        return { x: 450, y: 80 + (index - 2) * 100 };
      }
    } else if (selectedType === 'mapreduce') {
      if (node?.type === 'split') return { x: 200, y: 200 };
      if (node?.type === 'process') {
        return { x: 400, y: 80 + (index - 2) * 100 };
      }
      if (node?.type === 'merge') return { x: 600, y: 200 };
    }
    
    return { x: 400, y: 200 };
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        链编排架构可视化
      </h3>
      
      {/* 类型选择器 */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {(Object.keys(architectures) as ChainType[]).map((type) => (
          <button
            key={type}
            onClick={() => setSelectedType(type)}
            className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${
              selectedType === type
                ? 'bg-blue-500 text-white shadow-lg'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
            }`}
          >
            {architectures[type].name}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={selectedType}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >
          {/* 描述 */}
          <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-slate-700 dark:text-slate-300">
              {architecture.description}
            </p>
          </div>

          {/* 流程图 */}
          <div className="relative h-96 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700 mb-6 overflow-hidden">
            <svg className="absolute inset-0 w-full h-full">
              {/* 绘制边 */}
              {architecture.edges.map((edge, idx) => {
                const fromNode = architecture.nodes.find(n => n.id === edge.from);
                const toNode = architecture.nodes.find(n => n.id === edge.to);
                const fromIdx = architecture.nodes.findIndex(n => n.id === edge.from);
                const toIdx = architecture.nodes.findIndex(n => n.id === edge.to);
                
                const fromPos = getNodePosition(edge.from, fromIdx, architecture.nodes.length);
                const toPos = getNodePosition(edge.to, toIdx, architecture.nodes.length);
                
                return (
                  <motion.g
                    key={idx}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                  >
                    <line
                      x1={fromPos.x + 60}
                      y1={fromPos.y}
                      x2={toPos.x - 60}
                      y2={toPos.y}
                      stroke="#3b82f6"
                      strokeWidth="2"
                      markerEnd="url(#arrowhead)"
                    />
                    {edge.label && (
                      <text
                        x={(fromPos.x + toPos.x) / 2}
                        y={(fromPos.y + toPos.y) / 2 - 10}
                        className="text-xs fill-slate-600 dark:fill-slate-400"
                        textAnchor="middle"
                      >
                        {edge.label}
                      </text>
                    )}
                  </motion.g>
                );
              })}
              
              {/* 箭头定义 */}
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="10"
                  refX="9"
                  refY="3"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3, 0 6" fill="#3b82f6" />
                </marker>
              </defs>
            </svg>
            
            {/* 节点 */}
            {architecture.nodes.map((node, idx) => {
              const pos = getNodePosition(node.id, idx, architecture.nodes.length);
              const colors = {
                input: 'bg-green-500',
                output: 'bg-blue-500',
                process: 'bg-purple-500',
                router: 'bg-orange-500',
                merge: 'bg-pink-500',
                split: 'bg-yellow-500'
              };
              
              return (
                <motion.div
                  key={node.id}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.4, delay: idx * 0.1 }}
                  className="absolute"
                  style={{
                    left: `${pos.x}px`,
                    top: `${pos.y - 20}px`,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  <div className={`${colors[node.type]} text-white px-4 py-2 rounded-lg shadow-lg text-sm font-medium whitespace-nowrap`}>
                    {node.label}
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* 代码示例 */}
          <div>
            <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
              代码实现
            </h4>
            <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto border border-slate-700">
              <code>{architecture.code}</code>
            </pre>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

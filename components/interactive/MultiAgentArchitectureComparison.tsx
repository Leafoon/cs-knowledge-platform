'use client';

import React, { useState } from 'react';

type ArchitectureType = 'supervisor' | 'hierarchical' | 'collaborative';

interface ArchitectureInfo {
  name: string;
  description: string;
  structure: string;
  advantages: string[];
  disadvantages: string[];
  useCases: string[];
  color: string;
}

export default function MultiAgentArchitectureComparison() {
  const [selected, setSelected] = useState<ArchitectureType>('supervisor');

  const architectures: Record<ArchitectureType, ArchitectureInfo> = {
    supervisor: {
      name: 'Supervisor 模式',
      description: '中心化调度，由一个 Supervisor Agent 负责任务分解和 Worker 调度',
      structure: '星型结构',
      advantages: [
        '统一协调，逻辑清晰',
        '易于实现和调试',
        '适合任务分解明确的场景',
        '性能开销较小'
      ],
      disadvantages: [
        'Supervisor 成为单点故障',
        '扩展性受限于 Supervisor 能力',
        '不适合高度并行的任务',
        'Workers 之间无法直接通信'
      ],
      useCases: [
        '研究助手系统（搜索+分析+写作）',
        '数据处理流水线',
        '简单的客服系统',
        '内容生成工作流'
      ],
      color: 'from-blue-500 to-blue-600'
    },
    hierarchical: {
      name: 'Hierarchical 模式',
      description: '层级管理，模拟企业组织结构，支持多层决策和任务委派',
      structure: '树型结构',
      advantages: [
        '支持大规模 Agent 团队',
        '任务分解更细致',
        '责任划分清晰',
        '易于管理和监控'
      ],
      disadvantages: [
        '通信链路长，延迟高',
        '层级过多导致效率降低',
        '配置和维护复杂',
        '顶层故障影响范围大'
      ],
      useCases: [
        '大型软件项目（规划→开发→测试→部署）',
        '企业流程自动化',
        '复杂决策系统',
        '多部门协作任务'
      ],
      color: 'from-purple-500 to-purple-600'
    },
    collaborative: {
      name: 'Collaborative 模式',
      description: '平等协作，Agents 地位平等，通过协商、投票达成共识',
      structure: '网状结构',
      advantages: [
        '无单点故障',
        '适合创意型任务',
        '多角度评估',
        '容错性强'
      ],
      disadvantages: [
        '决策效率较低',
        '可能出现意见分歧',
        '通信开销大',
        '难以收敛到最优解'
      ],
      useCases: [
        '头脑风暴和创意生成',
        '多专家评审系统',
        '辩论和决策分析',
        '多角色协作游戏'
      ],
      color: 'from-green-500 to-green-600'
    }
  };

  const current = architectures[selected];

  return (
    <div className="my-8 p-6 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700">
      <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        多 Agent 架构模式对比
      </h3>
      
      {/* 架构选择器 */}
      <div className="flex gap-3 mb-6">
        {Object.entries(architectures).map(([key, arch]) => (
          <button
            key={key}
            onClick={() => setSelected(key as ArchitectureType)}
            className={`px-6 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 ${
              selected === key
                ? `bg-gradient-to-r ${arch.color} text-white shadow-lg`
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 shadow-md hover:shadow-lg'
            }`}
          >
            {arch.name}
          </button>
        ))}
      </div>

      {/* 架构可视化 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 mb-6 shadow-lg">
        {selected === 'supervisor' && (
          <svg viewBox="0 0 600 400" className="w-full h-64">
            {/* Supervisor */}
            <circle cx="300" cy="80" r="40" fill="#3b82f6" />
            <text x="300" y="85" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
              Supervisor
            </text>
            
            {/* Workers */}
            {[
              { x: 150, y: 280, label: 'Worker 1' },
              { x: 300, y: 280, label: 'Worker 2' },
              { x: 450, y: 280, label: 'Worker 3' }
            ].map((worker, i) => (
              <g key={i}>
                <line x1="300" y1="120" x2={worker.x} y2="240" stroke="#94a3b8" strokeWidth="2" />
                <circle cx={worker.x} cy={worker.y} r="35" fill="#60a5fa" />
                <text x={worker.x} y={worker.y + 5} textAnchor="middle" fill="white" fontSize="12" fontWeight="600">
                  {worker.label}
                </text>
              </g>
            ))}
          </svg>
        )}

        {selected === 'hierarchical' && (
          <svg viewBox="0 0 700 450" className="w-full h-64">
            {/* Manager */}
            <circle cx="350" cy="60" r="35" fill="#8b5cf6" />
            <text x="350" y="65" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
              Manager
            </text>
            
            {/* Team Leads */}
            {[
              { x: 200, y: 180, label: 'Lead 1' },
              { x: 350, y: 180, label: 'Lead 2' },
              { x: 500, y: 180, label: 'Lead 3' }
            ].map((lead, i) => (
              <g key={i}>
                <line x1="350" y1="95" x2={lead.x} y2="145" stroke="#94a3b8" strokeWidth="2" />
                <circle cx={lead.x} cy={lead.y} r="30" fill="#a78bfa" />
                <text x={lead.x} y={lead.y + 4} textAnchor="middle" fill="white" fontSize="11" fontWeight="600">
                  {lead.label}
                </text>
              </g>
            ))}
            
            {/* Workers */}
            {[
              { lx: 200, wx: 150, wy: 320 },
              { lx: 200, wx: 250, wy: 320 },
              { lx: 350, wx: 300, wy: 320 },
              { lx: 350, wx: 400, wy: 320 },
              { lx: 500, wx: 450, wy: 320 },
              { lx: 500, wx: 550, wy: 320 }
            ].map((w, i) => (
              <g key={i}>
                <line x1={w.lx} y1="210" x2={w.wx} y2="295" stroke="#94a3b8" strokeWidth="1.5" />
                <circle cx={w.wx} cy={w.wy} r="20" fill="#c4b5fd" />
                <text x={w.wx} y={w.wy + 4} textAnchor="middle" fill="white" fontSize="10" fontWeight="600">
                  W{i + 1}
                </text>
              </g>
            ))}
          </svg>
        )}

        {selected === 'collaborative' && (
          <svg viewBox="0 0 600 400" className="w-full h-64">
            {/* Agents */}
            {[
              { x: 300, y: 100, label: 'Agent 1' },
              { x: 450, y: 200, label: 'Agent 2' },
              { x: 350, y: 320, label: 'Agent 3' },
              { x: 150, y: 200, label: 'Agent 4' }
            ].map((agent, i, arr) => (
              <g key={i}>
                {/* 连接线 */}
                {arr.map((other, j) => {
                  if (i < j) {
                    return (
                      <line
                        key={j}
                        x1={agent.x}
                        y1={agent.y}
                        x2={other.x}
                        y2={other.y}
                        stroke="#94a3b8"
                        strokeWidth="2"
                        strokeDasharray="5,5"
                        opacity="0.4"
                      />
                    );
                  }
                  return null;
                })}
                {/* Agent 节点 */}
                <circle cx={agent.x} cy={agent.y} r="35" fill="#10b981" />
                <text x={agent.x} y={agent.y + 5} textAnchor="middle" fill="white" fontSize="12" fontWeight="600">
                  {agent.label}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>

      {/* 详细信息 */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-md">
          <h4 className="font-bold text-lg mb-3 text-green-600 dark:text-green-400">✓ 优势</h4>
          <ul className="space-y-2">
            {current.advantages.map((adv, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                <span className="text-green-500 mt-0.5">▪</span>
                <span>{adv}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-md">
          <h4 className="font-bold text-lg mb-3 text-red-600 dark:text-red-400">✗ 劣势</h4>
          <ul className="space-y-2">
            {current.disadvantages.map((dis, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                <span className="text-red-500 mt-0.5">▪</span>
                <span>{dis}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* 应用场景 */}
      <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h4 className="font-bold text-lg mb-3 text-gray-800 dark:text-gray-100">💡 典型应用场景</h4>
        <div className="grid md:grid-cols-2 gap-3">
          {current.useCases.map((useCase, i) => (
            <div key={i} className="flex items-center gap-3 bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-bold">{i + 1}</span>
              </div>
              <span className="text-sm text-gray-700 dark:text-gray-300">{useCase}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 结构信息 */}
      <div className="mt-4 flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
        <div className="flex items-center gap-2">
          <span className="font-semibold">结构:</span>
          <span className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-full">{current.structure}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold">描述:</span>
          <span>{current.description}</span>
        </div>
      </div>
    </div>
  );
}

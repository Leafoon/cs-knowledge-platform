"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Database, Clock, Network, Search, Plus, Trash2, Eye } from 'lucide-react';

type MemoryType = 'working' | 'episodic' | 'semantic';

type Memory = {
  id: string;
  type: MemoryType;
  content: string;
  timestamp: string;
  importance: number;
  relations?: string[];
};

type KnowledgeNode = {
  id: string;
  label: string;
  type: 'entity' | 'relation';
  x: number;
  y: number;
};

type KnowledgeEdge = {
  from: string;
  to: string;
  label: string;
};

export default function LongTermMemoryArchitecture() {
  const [activeLayer, setActiveLayer] = useState<MemoryType>('working');
  const [workingMemory, setWorkingMemory] = useState<Memory[]>([
    { id: 'w1', type: 'working', content: '用户: 我喜欢Python编程', timestamp: '2024-01-29 14:30:00', importance: 0.8 },
    { id: 'w2', type: 'working', content: '助手: Python是一门优秀的语言', timestamp: '2024-01-29 14:30:05', importance: 0.6 }
  ]);
  
  const [episodicMemory, setEpisodicMemory] = useState<Memory[]>([
    { id: 'e1', type: 'episodic', content: '用户询问过Rust语言学习资料', timestamp: '2024-01-28 10:15:00', importance: 0.7 },
    { id: 'e2', type: 'episodic', content: '用户在ProjectX中遇到并发问题', timestamp: '2024-01-27 16:40:00', importance: 0.85 },
    { id: 'e3', type: 'episodic', content: '用户偏好使用VS Code编辑器', timestamp: '2024-01-26 09:20:00', importance: 0.6 }
  ]);

  const [knowledgeGraph] = useState<{ nodes: KnowledgeNode[], edges: KnowledgeEdge[] }>({
    nodes: [
      { id: 'alice', label: 'Alice', type: 'entity', x: 200, y: 150 },
      { id: 'python', label: 'Python', type: 'entity', x: 350, y: 100 },
      { id: 'projectx', label: 'ProjectX', type: 'entity', x: 350, y: 200 },
      { id: 'rust', label: 'Rust', type: 'entity', x: 500, y: 150 }
    ],
    edges: [
      { from: 'alice', to: 'python', label: 'likes' },
      { from: 'alice', to: 'projectx', label: 'works_on' },
      { from: 'projectx', to: 'python', label: 'uses' },
      { from: 'alice', to: 'rust', label: 'learning' }
    ]
  });

  const [queryInput, setQueryInput] = useState('');
  const [queryResult, setQueryResult] = useState<Memory[]>([]);

  const layers = [
    {
      type: 'working' as MemoryType,
      name: '短期记忆',
      subtitle: 'Working Memory',
      description: '当前会话上下文',
      icon: <Brain className="w-6 h-6" />,
      color: 'bg-blue-500',
      borderColor: 'border-blue-500',
      textColor: 'text-blue-700',
      bgLight: 'bg-blue-50'
    },
    {
      type: 'episodic' as MemoryType,
      name: '中期记忆',
      subtitle: 'Episodic Memory',
      description: '最近会话摘要',
      icon: <Clock className="w-6 h-6" />,
      color: 'bg-green-500',
      borderColor: 'border-green-500',
      textColor: 'text-green-700',
      bgLight: 'bg-green-50'
    },
    {
      type: 'semantic' as MemoryType,
      name: '长期记忆',
      subtitle: 'Semantic Memory',
      description: '结构化知识图谱',
      icon: <Network className="w-6 h-6" />,
      color: 'bg-purple-500',
      borderColor: 'border-purple-500',
      textColor: 'text-purple-700',
      bgLight: 'bg-purple-50'
    }
  ];

  const getCurrentMemories = (): Memory[] => {
    if (activeLayer === 'working') return workingMemory;
    if (activeLayer === 'episodic') return episodicMemory;
    return [];
  };

  const handleQuery = () => {
    if (!queryInput.trim()) return;
    
    // 模拟向量检索（简化版：关键词匹配）
    const allMemories = [...workingMemory, ...episodicMemory];
    const results = allMemories.filter(m => 
      m.content.toLowerCase().includes(queryInput.toLowerCase())
    ).sort((a, b) => b.importance - a.importance);
    
    setQueryResult(results);
  };

  const addMemory = (type: MemoryType) => {
    const newMemory: Memory = {
      id: `${type}-${Date.now()}`,
      type,
      content: '新记忆内容...',
      timestamp: new Date().toISOString(),
      importance: 0.5
    };
    
    if (type === 'working') {
      setWorkingMemory([...workingMemory, newMemory]);
    } else if (type === 'episodic') {
      setEpisodicMemory([...episodicMemory, newMemory]);
    }
  };

  const deleteMemory = (id: string, type: MemoryType) => {
    if (type === 'working') {
      setWorkingMemory(workingMemory.filter(m => m.id !== id));
    } else if (type === 'episodic') {
      setEpisodicMemory(episodicMemory.filter(m => m.id !== id));
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">长期记忆系统架构</h3>
        <p className="text-gray-600">三层记忆：短期（会话）→ 中期（向量）→ 长期（知识图谱）</p>
      </div>

      {/* 记忆层次架构图 */}
      <div className="mb-6 space-y-3">
        {layers.map((layer, index) => (
          <motion.div
            key={layer.type}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <button
              onClick={() => setActiveLayer(layer.type)}
              className={`w-full p-4 rounded-lg border-2 transition-all ${
                activeLayer === layer.type
                  ? `${layer.borderColor} ${layer.bgLight}`
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <div className="flex items-center gap-4">
                <div className={`${layer.color} text-white p-3 rounded-lg`}>
                  {layer.icon}
                </div>
                <div className="flex-1 text-left">
                  <div className="flex items-center gap-2">
                    <h4 className="font-semibold text-lg">{layer.name}</h4>
                    <span className="text-xs px-2 py-1 bg-gray-200 rounded">{layer.subtitle}</span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{layer.description}</p>
                </div>
                <div className={`px-3 py-1 rounded ${layer.color} text-white text-sm font-medium`}>
                  {layer.type === 'working' ? workingMemory.length : 
                   layer.type === 'episodic' ? episodicMemory.length : 
                   knowledgeGraph.nodes.length} 项
                </div>
              </div>
            </button>
            
            {/* 连接箭头 */}
            {index < layers.length - 1 && (
              <div className="flex justify-center">
                <div className="text-2xl text-gray-400">↓</div>
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* 详细视图 */}
      <AnimatePresence mode="wait">
        {activeLayer !== 'semantic' && (
          <motion.div
            key={activeLayer}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="mb-6"
          >
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold">
                  {activeLayer === 'working' ? '当前会话记忆' : '历史会话摘要（向量存储）'}
                </h4>
                <button
                  onClick={() => addMemory(activeLayer)}
                  className="flex items-center gap-1 px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                >
                  <Plus className="w-4 h-4" />
                  添加
                </button>
              </div>

              <div className="space-y-2">
                {getCurrentMemories().map(memory => (
                  <motion.div
                    key={memory.id}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-start gap-3 p-3 bg-white rounded border"
                  >
                    <div className="flex-1">
                      <div className="text-sm">{memory.content}</div>
                      <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {memory.timestamp}
                        </span>
                        <span className="flex items-center gap-1">
                          <Database className="w-3 h-3" />
                          重要性: {(memory.importance * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => deleteMemory(memory.id, memory.type)}
                      className="text-red-600 hover:text-red-800"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {activeLayer === 'semantic' && (
          <motion.div
            key="semantic"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="mb-6"
          >
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-semibold mb-4">知识图谱（NetworkX）</h4>
              
              {/* SVG 图谱 */}
              <div className="bg-white border rounded-lg p-4 h-80 relative overflow-hidden">
                <svg width="100%" height="100%" viewBox="0 0 700 300">
                  {/* 边 */}
                  {knowledgeGraph.edges.map((edge, i) => {
                    const fromNode = knowledgeGraph.nodes.find(n => n.id === edge.from);
                    const toNode = knowledgeGraph.nodes.find(n => n.id === edge.to);
                    if (!fromNode || !toNode) return null;
                    
                    return (
                      <g key={i}>
                        <line
                          x1={fromNode.x}
                          y1={fromNode.y}
                          x2={toNode.x}
                          y2={toNode.y}
                          stroke="#cbd5e1"
                          strokeWidth="2"
                          markerEnd="url(#arrowhead)"
                        />
                        <text
                          x={(fromNode.x + toNode.x) / 2}
                          y={(fromNode.y + toNode.y) / 2 - 10}
                          textAnchor="middle"
                          fontSize="12"
                          fill="#64748b"
                          className="font-mono"
                        >
                          {edge.label}
                        </text>
                      </g>
                    );
                  })}
                  
                  {/* 箭头定义 */}
                  <defs>
                    <marker
                      id="arrowhead"
                      markerWidth="10"
                      markerHeight="10"
                      refX="8"
                      refY="3"
                      orient="auto"
                    >
                      <polygon points="0 0, 10 3, 0 6" fill="#cbd5e1" />
                    </marker>
                  </defs>
                  
                  {/* 节点 */}
                  {knowledgeGraph.nodes.map(node => (
                    <g key={node.id}>
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r="30"
                        fill={node.id === 'alice' ? '#8b5cf6' : '#3b82f6'}
                        stroke="#fff"
                        strokeWidth="3"
                      />
                      <text
                        x={node.x}
                        y={node.y + 5}
                        textAnchor="middle"
                        fontSize="14"
                        fill="white"
                        fontWeight="bold"
                      >
                        {node.label}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>

              {/* 关系列表 */}
              <div className="mt-4 space-y-2">
                <h5 className="text-sm font-semibold">关系三元组</h5>
                {knowledgeGraph.edges.map((edge, i) => (
                  <div key={i} className="text-sm bg-white p-2 rounded border font-mono">
                    <span className="text-purple-600">{edge.from}</span>
                    {' '}
                    <span className="text-gray-500">--[{edge.label}]--&gt;</span>
                    {' '}
                    <span className="text-blue-600">{edge.to}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 查询检索演示 */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-3">向量检索演示</h4>
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
            placeholder="输入查询内容（如：Python）"
            className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleQuery}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            <Search className="w-4 h-4" />
            检索
          </button>
        </div>

        {queryResult.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm text-gray-600">找到 {queryResult.length} 条相关记忆：</div>
            {queryResult.map((result, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="p-3 bg-white border rounded"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="text-sm">{result.content}</div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                      <span>{result.timestamp}</span>
                      <span className={`px-2 py-0.5 rounded ${
                        result.type === 'working' ? 'bg-blue-100 text-blue-700' :
                        'bg-green-100 text-green-700'
                      }`}>
                        {result.type === 'working' ? '短期' : '中期'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs">
                    <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-orange-500"
                        style={{ width: `${result.importance * 100}%` }}
                      ></div>
                    </div>
                    <span>{(result.importance * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* API 代码示例 */}
      <div className="mt-6 bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
        <div className="text-xs font-mono space-y-1">
          <div className="text-green-400"># 向量记忆</div>
          <div>memory_system.<span className="text-yellow-400">save_conversation</span>(</div>
          <div className="ml-4">user_id=<span className="text-orange-400">"user123"</span>,</div>
          <div className="ml-4">conversation=[...],</div>
          <div className="ml-4">importance=<span className="text-blue-400">0.7</span></div>
          <div>)</div>
          <div className="mt-2">results = memory_system.<span className="text-yellow-400">recall_relevant_memories</span>(</div>
          <div className="ml-4">user_id=<span className="text-orange-400">"user123"</span>,</div>
          <div className="ml-4">query=<span className="text-orange-400">"Python"</span>,</div>
          <div className="ml-4">k=<span className="text-blue-400">3</span></div>
          <div>)</div>
          
          <div className="mt-3 text-green-400"># 知识图谱</div>
          <div>kg.<span className="text-yellow-400">add_relation</span>(<span className="text-orange-400">"Alice"</span>, <span className="text-orange-400">"likes"</span>, <span className="text-orange-400">"Python"</span>)</div>
          <div>context = kg.<span className="text-yellow-400">get_entity_context</span>(<span className="text-orange-400">"Alice"</span>, depth=<span className="text-blue-400">2</span>)</div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type RunType = 'chain' | 'llm' | 'tool' | 'retriever' | 'embedding';

type TraceNode = {
  id: string;
  name: string;
  type: RunType;
  duration: number;
  status: 'success' | 'running' | 'error';
  input?: string;
  output?: string;
  children?: TraceNode[];
  expanded?: boolean;
};

const sampleTrace: TraceNode = {
  id: 'root',
  name: 'RetrievalQA Chain',
  type: 'chain',
  duration: 3200,
  status: 'success',
  input: 'What is LangChain?',
  output: 'LangChain is a framework for developing applications powered by language models...',
  expanded: true,
  children: [
    {
      id: 'retriever',
      name: 'VectorStoreRetriever',
      type: 'retriever',
      duration: 1200,
      status: 'success',
      input: 'What is LangChain?',
      output: '3 documents retrieved',
      expanded: false,
      children: [
        {
          id: 'embedding',
          name: 'OpenAIEmbeddings',
          type: 'embedding',
          duration: 500,
          status: 'success',
          input: 'What is LangChain?',
          output: 'Vector [0.123, -0.456, ...]',
        },
        {
          id: 'search',
          name: 'Chroma Search',
          type: 'tool',
          duration: 600,
          status: 'success',
          input: 'similarity search with score',
          output: 'Top 3 documents',
        },
      ],
    },
    {
      id: 'llm',
      name: 'ChatOpenAI',
      type: 'llm',
      duration: 1800,
      status: 'success',
      input: 'Context: ...\nQuestion: What is LangChain?',
      output: 'LangChain is a framework...',
    },
    {
      id: 'parser',
      name: 'StrOutputParser',
      type: 'tool',
      duration: 200,
      status: 'success',
      input: 'AIMessage(content="LangChain is...")',
      output: 'LangChain is a framework...',
    },
  ],
};

export default function TraceTreeVisualizer() {
  const [trace, setTrace] = useState<TraceNode>(sampleTrace);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const toggleNode = (nodeId: string) => {
    const toggleExpanded = (node: TraceNode): TraceNode => {
      if (node.id === nodeId) {
        return { ...node, expanded: !node.expanded };
      }
      if (node.children) {
        return {
          ...node,
          children: node.children.map(toggleExpanded),
        };
      }
      return node;
    };
    setTrace(toggleExpanded(trace));
  };

  const getTypeColor = (type: RunType): string => {
    const colors: Record<RunType, string> = {
      chain: '#8B5CF6',
      llm: '#3B82F6',
      tool: '#10B981',
      retriever: '#F59E0B',
      embedding: '#EF4444',
    };
    return colors[type];
  };

  const getTypeIcon = (type: RunType): string => {
    const icons: Record<RunType, string> = {
      chain: '📦',
      llm: '🤖',
      tool: '🔧',
      retriever: '🔍',
      embedding: '🧮',
    };
    return icons[type];
  };

  const renderNode = (node: TraceNode, level: number = 0) => {
    const hasChildren = node.children && node.children.length > 0;
    const isSelected = selectedNode === node.id;

    return (
      <div key={node.id} style={{ marginLeft: `${level * 24}px` }}>
        {/* Node Header */}
        <motion.div
          layout
          className={`flex items-center gap-2 p-3 mb-2 rounded-lg border-2 cursor-pointer transition-all ${
            isSelected
              ? 'border-blue-500 bg-blue-50'
              : 'border-slate-200 bg-white hover:border-slate-300'
          }`}
          onClick={() => {
            setSelectedNode(node.id);
            if (hasChildren) toggleNode(node.id);
          }}
        >
          {/* Expand/Collapse Icon */}
          {hasChildren && (
            <div className="w-5 h-5 flex items-center justify-center">
              <motion.div
                animate={{ rotate: node.expanded ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                ▶
              </motion.div>
            </div>
          )}
          {!hasChildren && <div className="w-5" />}

          {/* Type Icon */}
          <div className="text-xl">{getTypeIcon(node.type)}</div>

          {/* Name & Duration */}
          <div className="flex-1">
            <div className="font-semibold text-slate-800">{node.name}</div>
            <div className="text-xs text-slate-500">
              {node.type} · {node.duration}ms · {node.status}
            </div>
          </div>

          {/* Status Badge */}
          <div
            className={`px-2 py-1 rounded text-xs font-medium ${
              node.status === 'success'
                ? 'bg-green-100 text-green-700'
                : node.status === 'error'
                ? 'bg-red-100 text-red-700'
                : 'bg-yellow-100 text-yellow-700'
            }`}
          >
            {node.status === 'success' ? '✓' : node.status === 'error' ? '✗' : '⋯'}
          </div>

          {/* Duration Bar */}
          <div className="w-24">
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(100, (node.duration / 2000) * 100)}%`,
                  backgroundColor: getTypeColor(node.type),
                }}
              />
            </div>
          </div>
        </motion.div>

        {/* Children */}
        <AnimatePresence>
          {hasChildren && node.expanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              {node.children!.map((child) => renderNode(child, level + 1))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  const findNode = (nodeId: string, node: TraceNode = trace): TraceNode | null => {
    if (node.id === nodeId) return node;
    if (node.children) {
      for (const child of node.children) {
        const found = findNode(nodeId, child);
        if (found) return found;
      }
    }
    return null;
  };

  const selectedNodeData = selectedNode ? findNode(selectedNode) : null;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Trace Tree Visualizer
      </h3>
      <p className="text-center text-slate-600 mb-6">
        层级结构展示 LangChain 执行链的父子关系与数据流
      </p>

      <div className="grid grid-cols-3 gap-6">
        {/* Tree View */}
        <div className="col-span-2 bg-white rounded-lg p-4 shadow-md">
          <h4 className="font-semibold text-lg mb-4 text-slate-800">Trace Tree</h4>
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {renderNode(trace)}
          </div>
        </div>

        {/* Details Panel */}
        <div className="bg-white rounded-lg p-4 shadow-md">
          <h4 className="font-semibold text-lg mb-4 text-slate-800">Run Details</h4>
          {selectedNodeData ? (
            <div className="space-y-4">
              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                  Name
                </div>
                <div className="text-sm text-slate-800">{selectedNodeData.name}</div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                  Type
                </div>
                <div
                  className="inline-block px-2 py-1 rounded text-xs font-medium text-white"
                  style={{ backgroundColor: getTypeColor(selectedNodeData.type) }}
                >
                  {getTypeIcon(selectedNodeData.type)} {selectedNodeData.type}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                  Duration
                </div>
                <div className="text-sm text-slate-800">{selectedNodeData.duration}ms</div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                  Status
                </div>
                <div className="text-sm text-slate-800">{selectedNodeData.status}</div>
              </div>

              {selectedNodeData.input && (
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                    Input
                  </div>
                  <div className="text-xs text-slate-700 bg-slate-50 p-2 rounded font-mono max-h-32 overflow-y-auto">
                    {selectedNodeData.input}
                  </div>
                </div>
              )}

              {selectedNodeData.output && (
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase mb-1">
                    Output
                  </div>
                  <div className="text-xs text-slate-700 bg-slate-50 p-2 rounded font-mono max-h-32 overflow-y-auto">
                    {selectedNodeData.output}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-500 text-center py-8">
              点击左侧节点查看详细信息
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 p-4 bg-white rounded-lg shadow-md">
        <h4 className="font-semibold text-sm mb-3 text-slate-800">Run Types</h4>
        <div className="grid grid-cols-5 gap-4">
          {(['chain', 'llm', 'tool', 'retriever', 'embedding'] as RunType[]).map((type) => (
            <div key={type} className="flex items-center gap-2">
              <div className="text-xl">{getTypeIcon(type)}</div>
              <div>
                <div className="text-xs font-semibold text-slate-700 capitalize">{type}</div>
                <div
                  className="h-1 rounded-full mt-1"
                  style={{ width: '40px', backgroundColor: getTypeColor(type) }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="text-sm text-slate-700">
          <span className="font-semibold">使用说明：</span>
          点击节点展开/折叠子节点，查看每一步的输入输出和执行时长。通过树形结构理解 LangChain 链的完整执行流程。
        </div>
      </div>
    </div>
  );
}

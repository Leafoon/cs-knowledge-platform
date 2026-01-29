'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Network, Brain, Image as ImageIcon, Volume2, FileText, Link2 } from 'lucide-react'

export default function MultimodalMemoryGraph() {
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [showConnections, setShowConnections] = useState(true)

  const nodes = [
    {
      id: 'text-1',
      type: 'text',
      content: '"如何实现微服务架构？"',
      timestamp: '10:30',
      icon: <FileText className="w-4 h-4" />,
      color: 'blue',
      connections: ['text-2', 'image-1']
    },
    {
      id: 'image-1',
      type: 'image',
      content: 'architecture_diagram.png',
      timestamp: '10:32',
      icon: <ImageIcon className="w-4 h-4" />,
      color: 'purple',
      connections: ['text-1', 'text-3']
    },
    {
      id: 'text-2',
      type: 'text',
      content: '"微服务拆分策略..."',
      timestamp: '10:31',
      icon: <FileText className="w-4 h-4" />,
      color: 'blue',
      connections: ['text-1', 'audio-1']
    },
    {
      id: 'audio-1',
      type: 'audio',
      content: 'voice_question.mp3 → "解释负载均衡"',
      timestamp: '10:35',
      icon: <Volume2 className="w-4 h-4" />,
      color: 'green',
      connections: ['text-2', 'text-3']
    },
    {
      id: 'text-3',
      type: 'text',
      content: '"K8s HPA 配置示例..."',
      timestamp: '10:36',
      icon: <FileText className="w-4 h-4" />,
      color: 'blue',
      connections: ['image-1', 'audio-1', 'image-2']
    },
    {
      id: 'image-2',
      type: 'image',
      content: 'k8s_manifest.png',
      timestamp: '10:38',
      icon: <ImageIcon className="w-4 h-4" />,
      color: 'purple',
      connections: ['text-3']
    }
  ]

  const getNodePosition = (index: number) => {
    const angle = (index / nodes.length) * 2 * Math.PI
    const radius = 140
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius
    }
  }

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border border-indigo-200 dark:border-indigo-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-indigo-500 rounded-lg">
            <Network className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
              跨模态记忆图谱
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              文本、图像、语音统一索引与关联检索
            </p>
          </div>
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showConnections}
            onChange={(e) => setShowConnections(e.target.checked)}
            className="w-4 h-4"
          />
          <span className="text-sm text-slate-700 dark:text-slate-300">显示连接</span>
        </label>
      </div>

      {/* 图谱可视化 */}
      <div className="relative h-96 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
        <svg className="absolute inset-0 w-full h-full">
          {/* 绘制连接线 */}
          {showConnections && nodes.map(node => {
            const startPos = getNodePosition(nodes.indexOf(node))
            return node.connections.map(targetId => {
              const targetNode = nodes.find(n => n.id === targetId)
              if (!targetNode) return null
              
              const targetPos = getNodePosition(nodes.indexOf(targetNode))
              const isHighlighted = selectedNode === node.id || selectedNode === targetId
              
              return (
                <line
                  key={`${node.id}-${targetId}`}
                  x1={startPos.x + 200}
                  y1={startPos.y + 192}
                  x2={targetPos.x + 200}
                  y2={targetPos.y + 192}
                  stroke={isHighlighted ? '#8b5cf6' : '#cbd5e1'}
                  strokeWidth={isHighlighted ? 2 : 1}
                  strokeDasharray={isHighlighted ? '0' : '4'}
                  className="transition-all"
                />
              )
            })
          })}
        </svg>

        {/* 节点 */}
        {nodes.map((node, index) => {
          const pos = getNodePosition(index)
          const isSelected = selectedNode === node.id
          const isConnected = selectedNode && nodes.find(n => n.id === selectedNode)?.connections.includes(node.id)
          
          return (
            <motion.div
              key={node.id}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
              className={`absolute cursor-pointer transition-all ${
                isSelected ? 'z-20 scale-110' : isConnected ? 'z-10' : 'z-0'
              }`}
              style={{
                left: `calc(50% + ${pos.x}px)`,
                top: `calc(50% + ${pos.y}px)`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <div className={`p-4 rounded-lg border-2 transition-all ${
                isSelected
                  ? `bg-${node.color}-100 dark:bg-${node.color}-900/30 border-${node.color}-500`
                  : isConnected
                  ? `bg-${node.color}-50 dark:bg-${node.color}-900/20 border-${node.color}-300`
                  : 'bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600'
              }`}>
                <div className={`p-2 bg-${node.color}-500 rounded-lg inline-block mb-2`}>
                  {node.icon}
                  <span className="ml-1 text-white text-xs font-medium">
                    {node.type.toUpperCase()}
                  </span>
                </div>
                
                <div className="text-xs text-slate-700 dark:text-slate-300 mb-1 max-w-[120px] truncate">
                  {node.content}
                </div>
                <div className="text-xs text-slate-500">{node.timestamp}</div>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* 节点详情 */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          {(() => {
            const node = nodes.find(n => n.id === selectedNode)!
            return (
              <>
                <div className="flex items-center gap-3 mb-3">
                  <div className={`p-2 bg-${node.color}-500 rounded-lg`}>
                    {node.icon}
                  </div>
                  <div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {node.type.charAt(0).toUpperCase() + node.type.slice(1)} 内容
                    </div>
                    <div className="text-xs text-slate-500">{node.timestamp}</div>
                  </div>
                </div>

                <div className="mb-3 p-3 bg-slate-50 dark:bg-slate-900 rounded text-sm text-slate-700 dark:text-slate-300">
                  {node.content}
                </div>

                <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                  <Link2 className="w-4 h-4" />
                  <span>关联节点：{node.connections.length} 个</span>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  {node.connections.map(connId => {
                    const connNode = nodes.find(n => n.id === connId)!
                    return (
                      <button
                        key={connId}
                        onClick={() => setSelectedNode(connId)}
                        className={`px-3 py-1 text-xs rounded-full bg-${connNode.color}-100 dark:bg-${connNode.color}-900/30 text-${connNode.color}-700 dark:text-${connNode.color}-300 hover:bg-${connNode.color}-200 dark:hover:bg-${connNode.color}-900/50 transition-all`}
                      >
                        {connNode.type}: {connNode.content.slice(0, 20)}...
                      </button>
                    )
                  })}
                </div>
              </>
            )
          })()}
        </motion.div>
      )}

      {/* 功能说明 */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <FileText className="w-6 h-6 text-blue-500 mb-2" />
          <div className="font-medium text-slate-800 dark:text-slate-200 mb-1">文本记忆</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">
            向量化存储，语义检索
          </div>
        </div>

        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <ImageIcon className="w-6 h-6 text-purple-500 mb-2" />
          <div className="font-medium text-slate-800 dark:text-slate-200 mb-1">图像记忆</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">
            CLIP 嵌入，图文互检
          </div>
        </div>

        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <Volume2 className="w-6 h-6 text-green-500 mb-2" />
          <div className="font-medium text-slate-800 dark:text-slate-200 mb-1">语音记忆</div>
          <div className="text-xs text-slate-600 dark:text-slate-400">
            Whisper 转文本存储
          </div>
        </div>
      </div>
    </div>
  )
}

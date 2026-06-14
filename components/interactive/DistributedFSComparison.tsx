'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HardDrive, Server, Database, Cloud, ChevronDown, ChevronUp, Layers, Shield, Zap, Clock } from 'lucide-react'

interface DFS {
  id: string
  name: string
  fullName: string
  icon: React.ElementType
  color: string
  bgColor: string
  borderColor: string
  year: string
  architecture: string
  consistency: string
  cacheStrategy: string
  protocol: string
  stateful: string
  scalability: string
  useCase: string
  strengths: string[]
  weaknesses: string[]
}

const fileSystems: DFS[] = [
  {
    id: 'nfs', name: 'NFS', fullName: 'Network File System', icon: HardDrive,
    color: 'text-blue-600 dark:text-blue-400', bgColor: 'bg-blue-50 dark:bg-blue-900/20', borderColor: 'border-blue-300 dark:border-blue-700',
    year: '1984', architecture: 'Client-Server（无状态服务器）',
    consistency: '弱一致性（缓存超时检查）',
    cacheStrategy: '块级缓存（内存，3-30 秒超时）',
    protocol: '无状态 RPC',
    stateful: '无状态（Stateless）',
    scalability: '好（服务器无状态，易扩展）',
    useCase: '局域网文件共享，Unix/Linux 环境',
    strengths: ['简单、易于实现', '服务器崩溃恢复容易', '跨平台兼容性好', '无状态设计，易扩展'],
    weaknesses: ['缓存一致性弱', '性能受网络延迟影响', '不支持离线访问', '安全模型较弱'],
  },
  {
    id: 'afs', name: 'AFS', fullName: 'Andrew File System', icon: Server,
    color: 'text-green-600 dark:text-green-400', bgColor: 'bg-green-50 dark:bg-green-900/20', borderColor: 'border-green-300 dark:border-green-700',
    year: '1986', architecture: 'Client-Server（有状态服务器）',
    consistency: '回调保证一致性（Callback）',
    cacheStrategy: '全文件缓存（本地磁盘）',
    protocol: '有状态协议 + 回调机制',
    stateful: '有状态（Stateful）',
    scalability: '中（服务器维护回调状态）',
    useCase: '广域网文件系统，大学/企业环境',
    strengths: ['全文件缓存，减少网络访问', '回调机制保证一致性', '支持离线访问', '适合广域网部署'],
    weaknesses: ['服务器维护回调状态开销大', '首次访问延迟高', '不适合小文件频繁更新', '扩展性受限'],
  },
  {
    id: 'gfs', name: 'GFS', fullName: 'Google File System', icon: Database,
    color: 'text-purple-600 dark:text-purple-400', bgColor: 'bg-purple-50 dark:bg-purple-900/20', borderColor: 'border-purple-300 dark:border-purple-700',
    year: '2003', architecture: 'Master-ChunkServer（单 Master）',
    consistency: '松弛一致性（Record Append 语义）',
    cacheStrategy: '无客户端缓存（流式读取）',
    protocol: '自定义 RPC',
    stateful: '有状态（Master 维护元数据）',
    scalability: '高（数据节点水平扩展）',
    useCase: '大规模数据处理（MapReduce）',
    strengths: ['为大文件优化，64MB Chunk', '高吞吐量，顺序读写', '自动副本管理', '追加操作高效'],
    weaknesses: ['单 Master 是瓶颈和单点故障', '不适合小文件', '一致性较弱', '非通用文件系统'],
  },
  {
    id: 'hdfs', name: 'HDFS', fullName: 'Hadoop Distributed File System', icon: Cloud,
    color: 'text-orange-600 dark:text-orange-400', bgColor: 'bg-orange-50 dark:bg-orange-900/20', borderColor: 'border-orange-300 dark:border-orange-700',
    year: '2006', architecture: 'NameNode-DataNode（HA 支持）',
    consistency: '一次写入多次读取（WORM）',
    cacheStrategy: '无客户端缓存',
    protocol: '自定义 RPC + TCP',
    stateful: '有状态（NameNode 维护元数据）',
    scalability: '高（数据节点水平扩展）',
    useCase: 'Hadoop 生态，大数据批处理',
    strengths: ['NameNode HA 避免单点故障', '128MB 大块设计', '副本策略灵活', 'Hadoop 生态集成'],
    weaknesses: ['NameNode 内存限制', '不适合低延迟访问', '小文件问题', '元数据操作较慢'],
  },
]

const comparisonDimensions = [
  { key: 'year', label: '诞生年份', icon: Clock },
  { key: 'architecture', label: '架构', icon: Layers },
  { key: 'consistency', label: '一致性保证', icon: Shield },
  { key: 'cacheStrategy', label: '缓存策略', icon: HardDrive },
  { key: 'stateful', label: '状态模型', icon: Database },
  { key: 'scalability', label: '可扩展性', icon: Zap },
  { key: 'useCase', label: '适用场景', icon: Server },
] as const

export default function DistributedFSComparison() {
  const [selected, setSelected] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards')

  const selectedFS = fileSystems.find(f => f.id === selected)

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-4">
        <HardDrive className="w-6 h-6 text-blue-500" />
        <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">分布式文件系统对比</h3>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        NFS / AFS / GFS / HDFS 四种分布式文件系统的架构、一致性、性能全面对比。
      </p>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setViewMode('cards')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${viewMode === 'cards' ? 'bg-blue-500 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
          卡片视图
        </button>
        <button onClick={() => setViewMode('table')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${viewMode === 'table' ? 'bg-blue-500 text-white' : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300'}`}>
          表格视图
        </button>
      </div>

      {viewMode === 'cards' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          {fileSystems.map(fs => {
            const Icon = fs.icon
            const isSelected = selected === fs.id
            return (
              <motion.div key={fs.id} onClick={() => setSelected(isSelected ? null : fs.id)}
                className={`p-4 rounded-xl border-2 cursor-pointer transition-colors ${isSelected ? `${fs.borderColor} ${fs.bgColor}` : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'}`}
                whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`p-2 rounded-lg ${fs.bgColor}`}><Icon className={`w-5 h-5 ${fs.color}`} /></div>
                  <div>
                    <div className={`font-bold ${fs.color}`}>{fs.name}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">{fs.fullName} ({fs.year})</div>
                  </div>
                  {isSelected ? <ChevronUp className="w-4 h-4 text-slate-400 ml-auto" /> : <ChevronDown className="w-4 h-4 text-slate-400 ml-auto" />}
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs mt-3">
                  <div><span className="text-slate-500 dark:text-slate-400">架构：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.architecture.split('（')[0]}</span></div>
                  <div><span className="text-slate-500 dark:text-slate-400">一致性：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.consistency.split('（')[0]}</span></div>
                  <div><span className="text-slate-500 dark:text-slate-400">缓存：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.cacheStrategy.split('（')[0]}</span></div>
                  <div><span className="text-slate-500 dark:text-slate-400">扩展性：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.scalability.split('（')[0]}</span></div>
                </div>

                <AnimatePresence>
                  {isSelected && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                      <div className="space-y-2 text-xs">
                        <div><span className="font-medium text-slate-600 dark:text-slate-300">协议：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.protocol}</span></div>
                        <div><span className="font-medium text-slate-600 dark:text-slate-300">状态：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.stateful}</span></div>
                        <div><span className="font-medium text-slate-600 dark:text-slate-300">场景：</span><span className="text-slate-700 dark:text-slate-300 ml-1">{fs.useCase}</span></div>
                        <div>
                          <span className="font-medium text-green-600 dark:text-green-400">优势：</span>
                          <ul className="mt-1 space-y-0.5">
                            {fs.strengths.map((s, i) => <li key={i} className="text-slate-600 dark:text-slate-400 ml-3">• {s}</li>)}
                          </ul>
                        </div>
                        <div>
                          <span className="font-medium text-red-600 dark:text-red-400">劣势：</span>
                          <ul className="mt-1 space-y-0.5">
                            {fs.weaknesses.map((w, i) => <li key={i} className="text-slate-600 dark:text-slate-400 ml-3">• {w}</li>)}
                          </ul>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </div>
      ) : (
        <div className="overflow-x-auto mb-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-slate-200 dark:border-slate-700">
                <th className="py-2 px-3 text-left text-slate-600 dark:text-slate-400 font-medium">维度</th>
                {fileSystems.map(fs => (
                  <th key={fs.id} className={`py-2 px-3 text-left font-bold ${fs.color}`}>{fs.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonDimensions.map((dim, i) => {
                const Icon = dim.icon
                return (
                  <tr key={dim.key} className={`border-b border-slate-100 dark:border-slate-800 ${i % 2 === 0 ? 'bg-slate-50 dark:bg-slate-800/50' : ''}`}>
                    <td className="py-2 px-3 font-medium text-slate-700 dark:text-slate-300 whitespace-nowrap">
                      <div className="flex items-center gap-1.5"><Icon className="w-3.5 h-3.5 text-slate-400" />{dim.label}</div>
                    </td>
                    {fileSystems.map(fs => (
                      <td key={fs.id} className="py-2 px-3 text-slate-600 dark:text-slate-400 text-xs">{(fs as any)[dim.key]}</td>
                    ))}
                  </tr>
                )
              })}
              <tr className="border-b border-slate-100 dark:border-slate-800">
                <td className="py-2 px-3 font-medium text-green-600 dark:text-green-400 whitespace-nowrap">优势</td>
                {fileSystems.map(fs => (
                  <td key={fs.id} className="py-2 px-3 text-xs text-slate-600 dark:text-slate-400">
                    <ul className="space-y-0.5">{fs.strengths.map((s, i) => <li key={i}>• {s}</li>)}</ul>
                  </td>
                ))}
              </tr>
              <tr>
                <td className="py-2 px-3 font-medium text-red-600 dark:text-red-400 whitespace-nowrap">劣势</td>
                {fileSystems.map(fs => (
                  <td key={fs.id} className="py-2 px-3 text-xs text-slate-600 dark:text-slate-400">
                    <ul className="space-y-0.5">{fs.weaknesses.map((w, i) => <li key={i}>• {w}</li>)}</ul>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">设计演进总结</h4>
        <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400 flex-wrap">
          <span className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium">NFS</span>
          <span className="text-slate-400">→</span>
          <span className="px-2 py-1 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 font-medium">AFS</span>
          <span className="text-slate-400">→</span>
          <span className="px-2 py-1 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 font-medium">GFS</span>
          <span className="text-slate-400">→</span>
          <span className="px-2 py-1 rounded bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 font-medium">HDFS</span>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
          从通用局域网文件共享（NFS）→ 广域网一致性保证（AFS）→ 大数据专用（GFS/HDFS），
          每一代都在特定场景下优化了架构和一致性模型。
        </p>
      </div>
    </div>
  )
}

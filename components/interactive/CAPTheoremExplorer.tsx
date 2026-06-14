'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Zap, Network, Info, ArrowRight, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'

type Property = 'C' | 'A' | 'P'

interface SystemExample {
  name: string
  type: 'CP' | 'AP' | 'CA'
  description: string
  useCase: string
  tradeoff: string
}

const systemExamples: Record<string, SystemExample[]> = {
  CP: [
    { name: 'ZooKeeper', type: 'CP', description: '分布式协调服务', useCase: '配置管理、Leader 选举', tradeoff: '网络分区时可能拒绝服务' },
    { name: 'HBase', type: 'CP', description: '列式分布式数据库', useCase: '大数据随机读写', tradeoff: '分区时 Region 不可用' },
    { name: 'etcd', type: 'CP', description: '键值存储', useCase: 'Kubernetes 配置存储', tradeoff: '少数节点故障时可能不可用' },
  ],
  AP: [
    { name: 'Cassandra', type: 'AP', description: '宽列分布式数据库', useCase: '高写入吞吐量场景', tradeoff: '可能读到旧数据' },
    { name: 'DynamoDB', type: 'AP', description: 'Amazon 键值数据库', useCase: '购物车、Session 管理', tradeoff: '最终一致性' },
    { name: 'DNS', type: 'AP', description: '域名解析系统', useCase: '互联网基础设施', tradeoff: '记录更新有延迟' },
  ],
  CA: [
    { name: 'PostgreSQL', type: 'CA', description: '传统关系型数据库', useCase: '单机或局域网，事务处理', tradeoff: '不能跨网络分区部署' },
    { name: 'MySQL', type: 'CA', description: '关系型数据库', useCase: 'Web 应用', tradeoff: '单机部署，无分区容忍' },
  ],
}

export default function CAPTheoremExplorer() {
  const [selectedProps, setSelectedProps] = useState<Set<Property>>(new Set())
  const [hoveredProp, setHoveredProp] = useState<Property | null>(null)

  const properties = {
    C: { name: '一致性', nameEn: 'Consistency', icon: Shield, color: 'blue', description: '所有节点在同一时间看到相同的数据' },
    A: { name: '可用性', nameEn: 'Availability', icon: Zap, color: 'green', description: '每个请求都能在有限时间内获得响应' },
    P: { name: '分区容忍', nameEn: 'Partition Tolerance', icon: Network, color: 'purple', description: '系统在网络分区时仍能继续运行' },
  }

  const toggleProp = (p: Property) => {
    const next = new Set(selectedProps)
    if (next.has(p)) next.delete(p)
    else if (next.size < 2) next.add(p)
    setSelectedProps(next)
  }

  const getSystemType = (): string | null => {
    const s = [...selectedProps].sort().join('')
    if (s === 'AC' || s === 'CA') return 'CA'
    if (s === 'CP') return 'CP'
    if (s === 'AP') return 'AP'
    return null
  }

  const systemType = getSystemType()
  const examples = systemType ? systemExamples[systemType] || [] : []

  const trianglePoints = [
    { x: 200, y: 30, prop: 'C' as Property, label: 'C 一致性' },
    { x: 40, y: 280, prop: 'A' as Property, label: 'A 可用性' },
    { x: 360, y: 280, prop: 'P' as Property, label: 'P 分区容忍' },
  ]

  const propColors: Record<Property, string> = { C: '#3b82f6', A: '#22c55e', P: '#a855f7' }
  const propBg: Record<Property, string> = { C: '#dbeafe', A: '#dcfce7', P: '#f3e8ff' }

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-6">
        <Network className="w-6 h-6 text-purple-500" />
        <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">CAP 定理探索器</h3>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-400 mb-6">
        在分布式系统中，网络分区（P）不可避免。选择任意两个属性查看对应的系统类型和示例。
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <svg viewBox="0 0 400 320" className="w-full">
            <polygon points="200,30 40,280 360,280" fill="none" stroke="#e2e8f0" strokeWidth="2" />

            {selectedProps.size === 2 && (
              <motion.polygon
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.15 }}
                points={
                  [...selectedProps].map(p => {
                    const pt = trianglePoints.find(t => t.prop === p)!
                    return `${pt.x},${pt.y}`
                  }).join(' ') + (selectedProps.size === 2
                    ? ` ${trianglePoints.find(t => !selectedProps.has(t.prop))!.x},${trianglePoints.find(t => !selectedProps.has(t.prop))!.y}`
                    : '')
                }
                fill="#6366f1"
                strokeWidth="0"
              />
            )}

            {trianglePoints.map(pt => {
              const isSelected = selectedProps.has(pt.prop)
              const isHovered = hoveredProp === pt.prop
              const color = propColors[pt.prop]
              return (
                <g key={pt.prop} className="cursor-pointer" onClick={() => toggleProp(pt.prop)}
                   onMouseEnter={() => setHoveredProp(pt.prop)} onMouseLeave={() => setHoveredProp(null)}>
                  <motion.circle cx={pt.x} cy={pt.y} r={isSelected ? 28 : isHovered ? 24 : 20}
                    fill={isSelected ? color : isHovered ? propBg[pt.prop] : '#f8fafc'}
                    stroke={color} strokeWidth={isSelected ? 3 : 2}
                    animate={{ scale: isSelected ? 1.1 : 1 }} transition={{ duration: 0.2 }} />
                  <text x={pt.x} y={pt.y + 1} textAnchor="middle" dominantBaseline="middle"
                    fontSize={isSelected ? 18 : 14} fontWeight="bold" fill={isSelected ? 'white' : color}>
                    {pt.prop}
                  </text>
                  <text x={pt.x} y={pt.y + (pt.prop === 'C' ? -35 : 40)} textAnchor="middle"
                    fontSize="12" fill="#64748b" className="dark:fill-slate-400">
                    {pt.label}
                  </text>
                </g>
              )
            })}

            {selectedProps.size === 2 && (() => {
              const pts = [...selectedProps].map(p => trianglePoints.find(t => t.prop === p)!)
              return pts.map((p, i) => (
                <motion.line key={i} x1={p.x} y1={p.y} x2={pts[1 - i].x} y2={pts[1 - i].y}
                  stroke="#6366f1" strokeWidth="3" strokeDasharray="8 4" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} />
              ))
            })()}
          </svg>

          <div className="grid grid-cols-3 gap-2 mt-2">
            {(Object.keys(properties) as Property[]).map(p => {
              const prop = properties[p]
              const Icon = prop.icon
              const isSelected = selectedProps.has(p)
              return (
                <motion.button key={p} onClick={() => toggleProp(p)}
                  className={`p-3 rounded-lg border-2 transition-colors ${isSelected ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30' : 'border-slate-200 dark:border-slate-700 hover:border-slate-300'}`}
                  whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  <Icon className={`w-5 h-5 mx-auto mb-1 ${isSelected ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'}`} />
                  <div className={`text-xs font-medium ${isSelected ? 'text-indigo-700 dark:text-indigo-300' : 'text-slate-500'}`}>{prop.name}</div>
                </motion.button>
              )
            })}
          </div>
        </div>

        <div>
          <AnimatePresence mode="wait">
            {systemType ? (
              <motion.div key={systemType} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">{systemType}</span>
                  <span className="text-sm text-slate-500 dark:text-slate-400">系统类型</span>
                </div>

                <div className="mb-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-amber-800 dark:text-amber-300">
                      {systemType === 'CA' ? 'CA 系统只能在单机或局域网环境部署，无法处理网络分区。' : systemType === 'CP' ? 'CP 系统在网络分区时可能拒绝服务以保证一致性。' : 'AP 系统在网络分区时继续服务，但可能返回旧数据。'}
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  {examples.map((ex, i) => (
                    <motion.div key={ex.name} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                      className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
                      <div className="font-medium text-slate-800 dark:text-slate-200">{ex.name}</div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{ex.description}</div>
                      <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">场景：{ex.useCase}</div>
                      <div className="mt-1 text-xs text-orange-600 dark:text-orange-400">权衡：{ex.tradeoff}</div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            ) : (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center h-full text-center py-12">
                <Info className="w-10 h-10 text-slate-300 dark:text-slate-600 mb-3" />
                <p className="text-slate-400 dark:text-slate-500 text-sm">选择两个属性查看对应的系统类型</p>
                <div className="flex gap-1 mt-2 text-xs text-slate-400 dark:text-slate-500">
                  <span>CP</span> <ArrowRight className="w-3 h-3" /> <span>一致性优先</span>
                  <span className="mx-1">|</span>
                  <span>AP</span> <ArrowRight className="w-3 h-3" /> <span>可用性优先</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {selectedProps.size > 0 && (
            <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              onClick={() => setSelectedProps(new Set())}
              className="mt-4 text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 underline">
              重置选择
            </motion.button>
          )}
        </div>
      </div>

      <div className="mt-6 p-4 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">核心理解</h4>
        <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
          <li className="flex items-start gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" /> 分布式系统中，网络分区（P）不可避免</li>
          <li className="flex items-start gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" /> 实际选择是在 C（一致性）和 A（可用性）之间权衡</li>
          <li className="flex items-start gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-green-500 mt-0.5 flex-shrink-0" /> 一致性是连续光谱，不是简单的二元选择</li>
          <li className="flex items-start gap-1.5"><XCircle className="w-3.5 h-3.5 text-red-500 mt-0.5 flex-shrink-0" /> CA 系统在分布式环境中无法真正存在</li>
        </ul>
      </div>
    </div>
  )
}

'use client'

import React, { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, User, Clock, FileText, AlertTriangle, CheckCircle, Activity, BarChart3, Search } from 'lucide-react'

interface AuditEvent {
  id: string
  timestamp: string
  userId: string
  userRole: string
  eventType: 'chain_start' | 'chain_end' | 'tool_execution' | 'chain_error' | 'data_access' | 'permission_denied'
  resource: string
  status: 'success' | 'failure' | 'blocked'
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
  details: string
  ipAddress?: string
  metadata?: Record<string, any>
}

const EVENT_CONFIG = {
  chain_start: {
    icon: <Activity className="w-4 h-4" />,
    label: '链执行开始',
    color: 'blue'
  },
  chain_end: {
    icon: <CheckCircle className="w-4 h-4" />,
    label: '链执行完成',
    color: 'green'
  },
  tool_execution: {
    icon: <AlertTriangle className="w-4 h-4" />,
    label: '工具调用',
    color: 'orange'
  },
  chain_error: {
    icon: <AlertTriangle className="w-4 h-4" />,
    label: '执行错误',
    color: 'red'
  },
  data_access: {
    icon: <FileText className="w-4 h-4" />,
    label: '数据访问',
    color: 'purple'
  },
  permission_denied: {
    icon: <Shield className="w-4 h-4" />,
    label: '权限拒绝',
    color: 'red'
  }
}

const RISK_LEVEL_CONFIG = {
  low: { label: '低', color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' },
  medium: { label: '中', color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' },
  high: { label: '高', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300' },
  critical: { label: '严重', color: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' }
}

// 模拟审计日志数据
const generateMockLogs = (): AuditEvent[] => {
  const users = [
    { id: 'admin_001', role: 'admin' },
    { id: 'user_123', role: 'user' },
    { id: 'guest_456', role: 'guest' },
    { id: 'internal_789', role: 'internal' }
  ]

  const resources = [
    'CustomerServiceChain',
    'DatabaseWriteTool',
    'SensitiveDataRetriever',
    'AnalyticsChain',
    'AdminDashboard'
  ]

  const events: AuditEvent[] = []
  const now = new Date()

  for (let i = 0; i < 50; i++) {
    const user = users[Math.floor(Math.random() * users.length)]
    const resource = resources[Math.floor(Math.random() * resources.length)]
    const timestamp = new Date(now.getTime() - Math.random() * 3600000 * 24).toISOString()
    
    let eventType: AuditEvent['eventType']
    let status: AuditEvent['status']
    let riskLevel: AuditEvent['riskLevel']
    let details: string

    // 根据用户角色和资源生成不同的事件
    if (user.role === 'guest' && resource.includes('Sensitive')) {
      eventType = 'permission_denied'
      status = 'blocked'
      riskLevel = 'high'
      details = `Guest user attempted to access ${resource}`
    } else if (resource === 'DatabaseWriteTool') {
      eventType = 'tool_execution'
      status = 'success'
      riskLevel = user.role === 'admin' ? 'medium' : 'high'
      details = `${user.role} executed database write operation`
    } else if (Math.random() > 0.9) {
      eventType = 'chain_error'
      status = 'failure'
      riskLevel = 'medium'
      details = 'Chain execution failed due to timeout'
    } else {
      eventType = Math.random() > 0.5 ? 'chain_start' : 'chain_end'
      status = 'success'
      riskLevel = 'low'
      details = `${resource} ${eventType === 'chain_start' ? 'started' : 'completed'} successfully`
    }

    events.push({
      id: `evt_${i.toString().padStart(3, '0')}`,
      timestamp,
      userId: user.id,
      userRole: user.role,
      eventType,
      resource,
      status,
      riskLevel,
      details,
      ipAddress: `192.168.1.${Math.floor(Math.random() * 255)}`,
      metadata: {
        duration: Math.random() * 5000,
        tokenCount: Math.floor(Math.random() * 1000)
      }
    })
  }

  return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
}

export default function SecurityAuditDashboard() {
  const [logs] = useState<AuditEvent[]>(generateMockLogs())
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null)
  const [filterStatus, setFilterStatus] = useState<'all' | 'success' | 'failure' | 'blocked'>('all')
  const [filterRisk, setFilterRisk] = useState<'all' | 'low' | 'medium' | 'high' | 'critical'>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h')

  // 过滤日志
  const filteredLogs = useMemo(() => {
    let filtered = logs

    if (filterStatus !== 'all') {
      filtered = filtered.filter(log => log.status === filterStatus)
    }

    if (filterRisk !== 'all') {
      filtered = filtered.filter(log => log.riskLevel === filterRisk)
    }

    if (searchTerm) {
      const term = searchTerm.toLowerCase()
      filtered = filtered.filter(log =>
        log.userId.toLowerCase().includes(term) ||
        log.resource.toLowerCase().includes(term) ||
        log.details.toLowerCase().includes(term)
      )
    }

    // 时间范围过滤
    const now = new Date()
    const rangeMs = {
      '1h': 3600000,
      '24h': 86400000,
      '7d': 604800000,
      '30d': 2592000000
    }[timeRange]

    filtered = filtered.filter(log =>
      now.getTime() - new Date(log.timestamp).getTime() < rangeMs
    )

    return filtered
  }, [logs, filterStatus, filterRisk, searchTerm, timeRange])

  // 统计数据
  const stats = useMemo(() => {
    return {
      total: filteredLogs.length,
      success: filteredLogs.filter(l => l.status === 'success').length,
      failure: filteredLogs.filter(l => l.status === 'failure').length,
      blocked: filteredLogs.filter(l => l.status === 'blocked').length,
      critical: filteredLogs.filter(l => l.riskLevel === 'critical').length,
      high: filteredLogs.filter(l => l.riskLevel === 'high').length,
      byUser: Object.entries(
        filteredLogs.reduce((acc, log) => {
          acc[log.userId] = (acc[log.userId] || 0) + 1
          return acc
        }, {} as Record<string, number>)
      ).sort((a, b) => b[1] - a[1]).slice(0, 5),
      byResource: Object.entries(
        filteredLogs.reduce((acc, log) => {
          acc[log.resource] = (acc[log.resource] || 0) + 1
          return acc
        }, {} as Record<string, number>)
      ).sort((a, b) => b[1] - a[1]).slice(0, 5)
    }
  }, [filteredLogs])

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()

    if (diff < 60000) return '刚刚'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`
    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-indigo-500 rounded-lg">
          <Shield className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            安全审计仪表盘
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            实时监控和审计 LangChain 应用的所有安全相关事件
          </p>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-600 dark:text-slate-400">总事件</span>
            <Activity className="w-4 h-4 text-blue-500" />
          </div>
          <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            {stats.total}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-600 dark:text-slate-400">成功</span>
            <CheckCircle className="w-4 h-4 text-green-500" />
          </div>
          <div className="text-2xl font-bold text-green-600">
            {stats.success}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-600 dark:text-slate-400">拦截</span>
            <Shield className="w-4 h-4 text-orange-500" />
          </div>
          <div className="text-2xl font-bold text-orange-600">
            {stats.blocked}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-600 dark:text-slate-400">高危事件</span>
            <AlertTriangle className="w-4 h-4 text-red-500" />
          </div>
          <div className="text-2xl font-bold text-red-600">
            {stats.critical + stats.high}
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 主日志列表 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 过滤器 */}
          <div className="flex flex-wrap gap-2 p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2 flex-1 min-w-[200px]">
              <Search className="w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="搜索用户、资源或详情..."
                className="flex-1 px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-sm"
              />
            </div>

            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-sm"
            >
              <option value="1h">最近1小时</option>
              <option value="24h">最近24小时</option>
              <option value="7d">最近7天</option>
              <option value="30d">最近30天</option>
            </select>

            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as any)}
              className="px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-sm"
            >
              <option value="all">全部状态</option>
              <option value="success">成功</option>
              <option value="failure">失败</option>
              <option value="blocked">拦截</option>
            </select>

            <select
              value={filterRisk}
              onChange={(e) => setFilterRisk(e.target.value as any)}
              className="px-3 py-1.5 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-900 text-sm"
            >
              <option value="all">全部风险</option>
              <option value="low">低</option>
              <option value="medium">中</option>
              <option value="high">高</option>
              <option value="critical">严重</option>
            </select>
          </div>

          {/* 日志列表 */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            <AnimatePresence mode="popLayout">
              {filteredLogs.slice(0, 20).map((log, index) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.02 }}
                  onClick={() => setSelectedEvent(log)}
                  className={`p-3 bg-white dark:bg-slate-800 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                    selectedEvent?.id === log.id
                      ? 'border-indigo-500 ring-2 ring-indigo-200 dark:ring-indigo-800'
                      : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-1.5 rounded ${
                      log.status === 'success' ? 'bg-green-100 dark:bg-green-900/30' :
                      log.status === 'blocked' ? 'bg-red-100 dark:bg-red-900/30' :
                      'bg-orange-100 dark:bg-orange-900/30'
                    }`}>
                      {EVENT_CONFIG[log.eventType].icon}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="font-medium text-sm text-slate-800 dark:text-slate-200 truncate">
                          {EVENT_CONFIG[log.eventType].label}
                        </span>
                        <div className="flex items-center gap-2 shrink-0">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            RISK_LEVEL_CONFIG[log.riskLevel].color
                          }`}>
                            {RISK_LEVEL_CONFIG[log.riskLevel].label}
                          </span>
                          <span className="text-xs text-slate-500 dark:text-slate-400">
                            {formatTimestamp(log.timestamp)}
                          </span>
                        </div>
                      </div>

                      <div className="text-xs text-slate-600 dark:text-slate-400 mb-1">
                        <User className="w-3 h-3 inline mr-1" />
                        {log.userId} ({log.userRole}) → {log.resource}
                      </div>

                      <div className="text-xs text-slate-500 dark:text-slate-500 truncate">
                        {log.details}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {filteredLogs.length === 0 && (
              <div className="p-12 text-center text-slate-400 dark:text-slate-500">
                没有匹配的审计日志
              </div>
            )}
          </div>
        </div>

        {/* 右侧详情和统计 */}
        <div className="space-y-4">
          {/* 事件详情 */}
          {selectedEvent ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
            >
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                事件详情
              </h4>

              <div className="space-y-3 text-sm">
                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">事件ID</div>
                  <div className="font-mono text-slate-700 dark:text-slate-300">{selectedEvent.id}</div>
                </div>

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">时间戳</div>
                  <div className="flex items-center gap-1 text-slate-700 dark:text-slate-300">
                    <Clock className="w-3 h-3" />
                    {new Date(selectedEvent.timestamp).toLocaleString('zh-CN')}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">用户</div>
                  <div className="flex items-center gap-1 text-slate-700 dark:text-slate-300">
                    <User className="w-3 h-3" />
                    {selectedEvent.userId} <span className="text-xs text-slate-500">({selectedEvent.userRole})</span>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">资源</div>
                  <div className="text-slate-700 dark:text-slate-300">{selectedEvent.resource}</div>
                </div>

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">状态</div>
                  <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                    selectedEvent.status === 'success' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                    selectedEvent.status === 'blocked' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' :
                    'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300'
                  }`}>
                    {selectedEvent.status}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">风险级别</div>
                  <div className={`inline-block px-2 py-1 rounded text-xs font-medium ${
                    RISK_LEVEL_CONFIG[selectedEvent.riskLevel].color
                  }`}>
                    {RISK_LEVEL_CONFIG[selectedEvent.riskLevel].label}
                  </div>
                </div>

                {selectedEvent.ipAddress && (
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">IP 地址</div>
                    <div className="font-mono text-slate-700 dark:text-slate-300">{selectedEvent.ipAddress}</div>
                  </div>
                )}

                <div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">详情</div>
                  <div className="p-2 bg-slate-50 dark:bg-slate-900 rounded text-slate-700 dark:text-slate-300">
                    {selectedEvent.details}
                  </div>
                </div>

                {selectedEvent.metadata && (
                  <div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">元数据</div>
                    <pre className="p-2 bg-slate-50 dark:bg-slate-900 rounded text-xs overflow-auto">
                      {JSON.stringify(selectedEvent.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <div className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 text-center text-slate-400 dark:text-slate-500">
              选择一个事件查看详情
            </div>
          )}

          {/* Top 用户 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              活跃用户 Top 5
            </h4>
            <div className="space-y-2">
              {stats.byUser.map(([userId, count], idx) => (
                <div key={userId} className="flex items-center justify-between text-sm">
                  <span className="text-slate-600 dark:text-slate-400 truncate">{userId}</span>
                  <span className="font-medium text-slate-800 dark:text-slate-200">{count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top 资源 */}
          <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">
            <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              热门资源 Top 5
            </h4>
            <div className="space-y-2">
              {stats.byResource.map(([resource, count], idx) => (
                <div key={resource} className="flex items-center justify-between text-sm">
                  <span className="text-slate-600 dark:text-slate-400 truncate">{resource}</span>
                  <span className="font-medium text-slate-800 dark:text-slate-200">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

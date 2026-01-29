'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, TrendingUp, Zap, Clock, AlertCircle, CheckCircle, BarChart3, PieChart } from 'lucide-react'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RePieChart, Pie, Cell } from 'recharts'

export default function MetricsDashboard() {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('1h')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState(new Date())
  
  // 模拟实时数据更新
  useEffect(() => {
    if (!autoRefresh) return
    
    const interval = setInterval(() => {
      setLastUpdate(new Date())
    }, 3000)
    
    return () => clearInterval(interval)
  }, [autoRefresh])
  
  // 延迟数据 (P50, P95, P99)
  const latencyData = [
    { time: '10:00', p50: 0.45, p95: 1.2, p99: 2.8 },
    { time: '10:15', p50: 0.52, p95: 1.5, p99: 3.2 },
    { time: '10:30', p50: 0.48, p95: 1.3, p99: 2.9 },
    { time: '10:45', p50: 0.61, p95: 1.8, p99: 4.1 },
    { time: '11:00', p50: 0.55, p95: 1.4, p99: 3.5 },
    { time: '11:15', p50: 0.49, p95: 1.2, p99: 2.7 },
    { time: '11:30', p50: 0.53, p95: 1.6, p99: 3.8 },
  ]
  
  // 请求量数据
  const requestData = [
    { time: '10:00', success: 145, error: 2 },
    { time: '10:15', success: 168, error: 3 },
    { time: '10:30', success: 152, error: 1 },
    { time: '10:45', success: 189, error: 5 },
    { time: '11:00', success: 176, error: 2 },
    { time: '11:15', success: 163, error: 1 },
    { time: '11:30', success: 194, error: 4 },
  ]
  
  // Token 消耗分布
  const tokenData = [
    { name: 'GPT-4', value: 42300, color: '#8b5cf6' },
    { name: 'GPT-3.5', value: 18500, color: '#3b82f6' },
    { name: 'Claude-3', value: 15200, color: '#10b981' },
    { name: 'Llama-2', value: 8100, color: '#f59e0b' },
  ]
  
  // 关键指标
  const metrics = {
    totalRequests: 1247,
    successRate: 98.4,
    avgLatency: 0.52,
    p95Latency: 1.4,
    errorRate: 1.6,
    activeConnections: 23,
    tokensPerMin: 14200,
    costPerHour: 4.35,
  }
  
  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl shadow-lg">
      {/* 标题栏 */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-8 h-8 text-indigo-600" />
            <h3 className="text-2xl font-bold text-gray-800">LangServe 监控仪表盘</h3>
          </div>
          <p className="text-sm text-gray-600">
            上次更新: {lastUpdate.toLocaleTimeString()} 
            {autoRefresh && <span className="ml-2 text-green-600">● 实时刷新</span>}
          </p>
        </div>
        
        <div className="flex gap-3">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            {autoRefresh ? '⏸ 暂停刷新' : '▶ 开始刷新'}
          </motion.button>
        </div>
      </div>

      {/* 时间范围选择 */}
      <div className="flex gap-2 mb-6">
        {(['1h', '6h', '24h'] as const).map((range) => (
          <button
            key={range}
            onClick={() => setTimeRange(range)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              timeRange === range
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            {range === '1h' ? '最近 1 小时' : range === '6h' ? '最近 6 小时' : '最近 24 小时'}
          </button>
        ))}
      </div>

      {/* 关键指标卡片 */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          icon={Activity}
          label="总请求数"
          value={metrics.totalRequests.toLocaleString()}
          trend="+12.3%"
          trendUp={true}
          color="blue"
        />
        <MetricCard
          icon={CheckCircle}
          label="成功率"
          value={`${metrics.successRate}%`}
          trend="+0.8%"
          trendUp={true}
          color="green"
        />
        <MetricCard
          icon={Clock}
          label="平均延迟"
          value={`${metrics.avgLatency}s`}
          trend="-0.05s"
          trendUp={true}
          color="purple"
        />
        <MetricCard
          icon={AlertCircle}
          label="错误率"
          value={`${metrics.errorRate}%`}
          trend="-0.3%"
          trendUp={true}
          color="red"
        />
      </div>

      {/* 图表区域 */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* 延迟分布图 */}
        <div className="bg-white rounded-lg p-6 shadow">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-indigo-600" />
            响应延迟 (秒)
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
              />
              <Line type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={2} name="P50" />
              <Line type="monotone" dataKey="p95" stroke="#f59e0b" strokeWidth={2} name="P95" />
              <Line type="monotone" dataKey="p99" stroke="#ef4444" strokeWidth={2} name="P99" />
            </LineChart>
          </ResponsiveContainer>
          
          <div className="flex justify-center gap-4 mt-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-xs text-gray-600">P50 (中位数)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              <span className="text-xs text-gray-600">P95</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-xs text-gray-600">P99</span>
            </div>
          </div>
        </div>

        {/* 请求量趋势图 */}
        <div className="bg-white rounded-lg p-6 shadow">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-green-600" />
            请求量趋势
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={requestData}>
              <defs>
                <linearGradient id="colorSuccess" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="colorError" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
              />
              <Area type="monotone" dataKey="success" stroke="#10b981" fillOpacity={1} fill="url(#colorSuccess)" name="成功" />
              <Area type="monotone" dataKey="error" stroke="#ef4444" fillOpacity={1} fill="url(#colorError)" name="错误" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Token 消耗 & 成本分析 */}
      <div className="grid grid-cols-2 gap-6">
        {/* Token 分布饼图 */}
        <div className="bg-white rounded-lg p-6 shadow">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-purple-600" />
            Token 消耗分布
          </h4>
          
          <div className="flex items-center justify-between">
            <ResponsiveContainer width="50%" height={180}>
              <RePieChart>
                <Pie
                  data={tokenData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={70}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {tokenData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </RePieChart>
            </ResponsiveContainer>
            
            <div className="space-y-2">
              {tokenData.map((item) => (
                <div key={item.name} className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: item.color }} />
                    <span className="text-sm text-gray-700">{item.name}</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-800">
                    {item.value.toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 实时指标 */}
        <div className="bg-white rounded-lg p-6 shadow">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-600" />
            实时指标
          </h4>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm text-gray-700">活跃连接</span>
              <span className="text-lg font-bold text-gray-800">{metrics.activeConnections}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm text-gray-700">Tokens/分钟</span>
              <span className="text-lg font-bold text-gray-800">{metrics.tokensPerMin.toLocaleString()}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm text-gray-700">成本/小时</span>
              <span className="text-lg font-bold text-green-600">${metrics.costPerHour}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
              <span className="text-sm text-blue-800 font-medium">P95 延迟</span>
              <span className={`text-lg font-bold ${
                metrics.p95Latency < 1 ? 'text-green-600' :
                metrics.p95Latency < 2 ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {metrics.p95Latency}s
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 告警提示 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="mt-6 p-4 bg-green-50 border-l-4 border-green-500 rounded-r-lg"
      >
        <div className="flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
          <div>
            <h5 className="font-semibold text-green-900 mb-1">系统运行正常</h5>
            <p className="text-sm text-green-800">
              所有指标处于健康范围。错误率 &lt; 2%，P95 延迟 &lt; 2s，成功率 &gt; 98%。
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

// 指标卡片组件
function MetricCard({ 
  icon: Icon, 
  label, 
  value, 
  trend, 
  trendUp, 
  color 
}: { 
  icon: React.ElementType
  label: string
  value: string
  trend: string
  trendUp: boolean
  color: 'blue' | 'green' | 'purple' | 'red'
}) {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200 text-blue-600',
    green: 'bg-green-50 border-green-200 text-green-600',
    purple: 'bg-purple-50 border-purple-200 text-purple-600',
    red: 'bg-red-50 border-red-200 text-red-600',
  }
  
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-white rounded-lg p-4 shadow border-l-4"
      style={{ borderLeftColor: color === 'blue' ? '#3b82f6' : color === 'green' ? '#10b981' : color === 'purple' ? '#8b5cf6' : '#ef4444' }}
    >
      <div className="flex items-center justify-between mb-2">
        <Icon className={`w-6 h-6 ${colorClasses[color]}`} />
        <span className={`text-xs font-semibold ${trendUp ? 'text-green-600' : 'text-red-600'}`}>
          {trend}
        </span>
      </div>
      <h5 className="text-sm text-gray-600 mb-1">{label}</h5>
      <p className="text-2xl font-bold text-gray-800">{value}</p>
    </motion.div>
  )
}

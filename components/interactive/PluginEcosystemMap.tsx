'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, Package, Download, Star, TrendingUp, Users } from 'lucide-react'

interface Plugin {
  id: string
  name: string
  category: string
  description: string
  downloads: string
  rating: number
  author: string
  tags: string[]
}

export default function PluginEcosystemMap() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null)

  const categories = [
    { id: 'all', name: '全部', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'rag', name: 'RAG 增强', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'observability', name: '可观测性', icon: <Users className="w-4 h-4" /> },
    { id: 'memory', name: '记忆管理', icon: <Package className="w-4 h-4" /> },
    { id: 'tools', name: '工具集成', icon: <Download className="w-4 h-4" /> }
  ]

  const plugins: Plugin[] = [
    {
      id: 'rag-pro',
      name: 'LangChain RAG Pro',
      category: 'rag',
      description: '混合检索 + 重排序 + 语义缓存',
      downloads: '150K',
      rating: 4.8,
      author: 'langchain-ai',
      tags: ['Hybrid Search', 'Reranker', 'Cache']
    },
    {
      id: 'obs-datadog',
      name: 'Datadog Observability',
      category: 'observability',
      description: '自动埋点、指标上报、告警集成',
      downloads: '89K',
      rating: 4.6,
      author: 'datadog',
      tags: ['Tracing', 'Metrics', 'Alerts']
    },
    {
      id: 'redis-vector-memory',
      name: 'Redis Vector Memory',
      category: 'memory',
      description: 'Redis 向量存储 + 会话管理',
      downloads: '120K',
      rating: 4.7,
      author: 'redis',
      tags: ['Vector DB', 'Session', 'Cache']
    },
    {
      id: 'rag-advanced',
      name: 'Advanced RAG Toolkit',
      category: 'rag',
      description: 'HyDE、多查询、父文档检索',
      downloads: '75K',
      rating: 4.5,
      author: 'community',
      tags: ['HyDE', 'Multi-Query', 'Parent Doc']
    },
    {
      id: 'prometheus-metrics',
      name: 'Prometheus Metrics',
      category: 'observability',
      description: '导出 Prometheus 格式指标',
      downloads: '95K',
      rating: 4.7,
      author: 'prometheus',
      tags: ['Metrics', 'Monitoring']
    },
    {
      id: 'graph-memory',
      name: 'Knowledge Graph Memory',
      category: 'memory',
      description: 'Neo4j 知识图谱记忆',
      downloads: '45K',
      rating: 4.4,
      author: 'neo4j',
      tags: ['Graph', 'Knowledge', 'Reasoning']
    },
    {
      id: 'browser-tools',
      name: 'Browser Automation Tools',
      category: 'tools',
      description: 'Playwright 浏览器自动化',
      downloads: '110K',
      rating: 4.6,
      author: 'microsoft',
      tags: ['Browser', 'Automation', 'Scraping']
    },
    {
      id: 'code-executor',
      name: 'Safe Code Executor',
      category: 'tools',
      description: '沙箱代码执行环境',
      downloads: '68K',
      rating: 4.3,
      author: 'community',
      tags: ['Sandbox', 'Code', 'Security']
    }
  ]

  const filteredPlugins = selectedCategory === 'all'
    ? plugins
    : plugins.filter(p => p.category === selectedCategory)

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border border-blue-200 dark:border-blue-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-blue-500 rounded-lg">
          <Package className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            LangChain 插件生态（预测）
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            未来插件市场：一键安装、社区贡献、版本管理
          </p>
        </div>
      </div>

      {/* 分类导航 */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map(cat => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              selectedCategory === cat.id
                ? 'bg-blue-500 text-white'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-blue-50 dark:hover:bg-blue-900/20'
            }`}
          >
            {cat.icon}
            {cat.name}
          </button>
        ))}
      </div>

      {/* 插件列表 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {filteredPlugins.map((plugin, idx) => (
          <motion.div
            key={plugin.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            onClick={() => setSelectedPlugin(plugin)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedPlugin?.id === plugin.id
                ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-500'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:shadow-lg'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                  {plugin.name}
                </div>
                <div className="text-xs text-slate-500">by {plugin.author}</div>
              </div>
              <div className="flex items-center gap-1 text-yellow-500">
                <Star className="w-4 h-4 fill-current" />
                <span className="text-sm font-medium">{plugin.rating}</span>
              </div>
            </div>

            <div className="text-sm text-slate-600 dark:text-slate-400 mb-3">
              {plugin.description}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Download className="w-3 h-3" />
                {plugin.downloads} 下载
              </div>
              <div className="flex flex-wrap gap-1">
                {plugin.tags.slice(0, 2).map(tag => (
                  <span
                    key={tag}
                    className="px-2 py-0.5 text-xs bg-slate-100 dark:bg-slate-700 rounded-full text-slate-600 dark:text-slate-400"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* 插件详情 */}
      {selectedPlugin && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="font-bold text-xl text-slate-800 dark:text-slate-200 mb-1">
                {selectedPlugin.name}
              </div>
              <div className="text-sm text-slate-500">
                作者：{selectedPlugin.author} | 下载量：{selectedPlugin.downloads}
              </div>
            </div>
            <button className="px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-all">
              <Download className="w-4 h-4 inline mr-2" />
              安装插件
            </button>
          </div>

          <div className="mb-4 p-3 bg-slate-50 dark:bg-slate-900 rounded text-sm text-slate-700 dark:text-slate-300">
            {selectedPlugin.description}
          </div>

          <div className="mb-4">
            <div className="font-medium text-slate-700 dark:text-slate-300 mb-2">标签</div>
            <div className="flex flex-wrap gap-2">
              {selectedPlugin.tags.map(tag => (
                <span
                  key={tag}
                  className="px-3 py-1 text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-900 rounded-lg font-mono text-sm">
            <div className="text-xs text-slate-500 mb-2"># 安装示例</div>
            <div className="text-slate-800 dark:text-slate-200">
              pip install langchain-plugin-{selectedPlugin.id}
            </div>
            <div className="text-slate-800 dark:text-slate-200 mt-2">
              from langchain.plugins import install_plugin
            </div>
            <div className="text-slate-800 dark:text-slate-200">
              plugin = install_plugin("{selectedPlugin.id}", version="latest")
            </div>
          </div>
        </motion.div>
      )}

      {/* 统计信息 */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
          <div className="flex items-center gap-3 mb-2">
            <Package className="w-6 h-6 text-blue-500" />
            <div className="font-bold text-2xl text-slate-800 dark:text-slate-200">
              {plugins.length}
            </div>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">可用插件</div>
        </div>

        <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
          <div className="flex items-center gap-3 mb-2">
            <Download className="w-6 h-6 text-green-500" />
            <div className="font-bold text-2xl text-slate-800 dark:text-slate-200">
              952K
            </div>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">总下载量</div>
        </div>

        <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
          <div className="flex items-center gap-3 mb-2">
            <Users className="w-6 h-6 text-purple-500" />
            <div className="font-bold text-2xl text-slate-800 dark:text-slate-200">
              1.2K
            </div>
          </div>
          <div className="text-sm text-slate-600 dark:text-slate-400">贡献者</div>
        </div>
      </div>
    </div>
  )
}

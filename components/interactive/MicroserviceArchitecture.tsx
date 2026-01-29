'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Server, Database, MessageSquare, Brain, Users, Zap, ArrowRight } from 'lucide-react'

export default function MicroserviceArchitecture() {
  const [selectedService, setSelectedService] = useState<string | null>(null)

  const services = [
    {
      id: 'gateway',
      name: 'API Gateway',
      icon: <Zap className="w-6 h-6" />,
      color: 'blue',
      description: '统一入口：认证、限流、路由',
      instances: 2,
      cpu: '20%',
      memory: '512MB'
    },
    {
      id: 'chat',
      name: 'Chat Service',
      icon: <MessageSquare className="w-6 h-6" />,
      color: 'green',
      description: '对话处理服务',
      instances: 5,
      cpu: '45%',
      memory: '1GB'
    },
    {
      id: 'rag',
      name: 'RAG Service',
      icon: <Database className="w-6 h-6" />,
      color: 'purple',
      description: '向量检索服务',
      instances: 8,
      cpu: '70%',
      memory: '2GB'
    },
    {
      id: 'agent',
      name: 'Agent Service',
      icon: <Brain className="w-6 h-6" />,
      color: 'orange',
      description: 'Agent 编排与工具调用',
      instances: 3,
      cpu: '35%',
      memory: '1.5GB'
    },
    {
      id: 'memory',
      name: 'Memory Service',
      icon: <Users className="w-6 h-6" />,
      color: 'pink',
      description: '对话记忆管理',
      instances: 4,
      cpu: '25%',
      memory: '800MB'
    }
  ]

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-900/20 rounded-xl border border-slate-200 dark:border-slate-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 bg-blue-500 rounded-lg">
          <Server className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
            微服务架构演示
          </h3>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            点击服务查看详细信息
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {services.map((service, idx) => (
          <motion.div
            key={service.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            onClick={() => setSelectedService(service.id)}
            className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
              selectedService === service.id
                ? `border-${service.color}-500 bg-${service.color}-50 dark:bg-${service.color}-900/20`
                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:shadow-lg'
            }`}
          >
            <div className={`p-3 bg-${service.color}-100 dark:bg-${service.color}-900/30 rounded-lg inline-block mb-3`}>
              {service.icon}
            </div>
            <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-2">{service.name}</h4>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">{service.description}</p>
            
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">实例数：</span>
                <span className="font-medium">{service.instances}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">CPU：</span>
                <span className="font-medium">{service.cpu}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">内存：</span>
                <span className="font-medium">{service.memory}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {selectedService && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-6 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700"
        >
          <h4 className="font-bold text-lg mb-4 text-slate-800 dark:text-slate-200">
            {services.find(s => s.id === selectedService)?.name} - 架构优势
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="font-medium text-green-700 dark:text-green-300 mb-2">✅ 独立扩容</div>
              <div className="text-sm text-green-600 dark:text-green-400">
                根据负载独立调整副本数，无需扩展整个系统
              </div>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="font-medium text-blue-700 dark:text-blue-300 mb-2">🛡️ 故障隔离</div>
              <div className="text-sm text-blue-600 dark:text-blue-400">
                服务崩溃不影响其他服务，降低故障影响范围
              </div>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="font-medium text-purple-700 dark:text-purple-300 mb-2">🚀 独立部署</div>
              <div className="text-sm text-purple-600 dark:text-purple-400">
                各服务独立发布、回滚，加快迭代速度
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

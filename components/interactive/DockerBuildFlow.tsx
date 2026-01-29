'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Box, FileCode, Layers, CheckCircle, ArrowRight, Play, Pause } from 'lucide-react'

type BuildStage = 'source' | 'dependencies' | 'build' | 'runtime' | 'final'

interface StageInfo {
  id: BuildStage
  name: string
  description: string
  files: string[]
  size: string
  color: string
}

export default function DockerBuildFlow() {
  const [currentStage, setCurrentStage] = useState<BuildStage>('source')
  const [isAnimating, setIsAnimating] = useState(false)
  const [completedStages, setCompletedStages] = useState<BuildStage[]>([])
  
  const stages: StageInfo[] = [
    {
      id: 'source',
      name: '源代码',
      description: '应用源代码和配置文件',
      files: ['main.py', 'chains/', 'requirements.txt', 'config.yaml'],
      size: '2.5 MB',
      color: 'blue',
    },
    {
      id: 'dependencies',
      name: '安装依赖',
      description: '下载并安装 Python 包',
      files: ['langchain', 'fastapi', 'uvicorn', 'pydantic', 'openai'],
      size: '450 MB',
      color: 'yellow',
    },
    {
      id: 'build',
      name: '构建阶段',
      description: '编译和打包应用（多阶段构建）',
      files: ['__pycache__/', 'compiled modules', 'wheels/'],
      size: '480 MB',
      color: 'orange',
    },
    {
      id: 'runtime',
      name: '运行时环境',
      description: '仅复制运行所需文件',
      files: ['python3.11-slim', 'app/', 'dependencies/'],
      size: '180 MB',
      color: 'green',
    },
    {
      id: 'final',
      name: '最终镜像',
      description: '优化后的生产镜像',
      files: ['app + deps + runtime'],
      size: '185 MB',
      color: 'purple',
    },
  ]
  
  const colorClasses = {
    blue: { bg: 'bg-blue-500', text: 'text-blue-700', light: 'bg-blue-50' },
    yellow: { bg: 'bg-yellow-500', text: 'text-yellow-700', light: 'bg-yellow-50' },
    orange: { bg: 'bg-orange-500', text: 'text-orange-700', light: 'bg-orange-50' },
    green: { bg: 'bg-green-500', text: 'text-green-700', light: 'bg-green-50' },
    purple: { bg: 'bg-purple-500', text: 'text-purple-700', light: 'bg-purple-50' },
  }
  
  const runAnimation = async () => {
    setIsAnimating(true)
    setCompletedStages([])
    
    for (let i = 0; i < stages.length; i++) {
      setCurrentStage(stages[i].id)
      await new Promise(resolve => setTimeout(resolve, 1500))
      setCompletedStages(prev => [...prev, stages[i].id])
    }
    
    setIsAnimating(false)
  }
  
  const dockerfileCode = `# 多阶段构建 Dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY main.py chains/ config/ ./
ENV PATH=/root/.local/bin:$PATH
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]`
  
  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-gray-50 to-indigo-50 rounded-xl shadow-lg">
      {/* 标题 */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Box className="w-8 h-8 text-indigo-600" />
          <h3 className="text-2xl font-bold text-gray-800">Docker 多阶段构建流程</h3>
        </div>
        <p className="text-gray-600">观察镜像从 480MB 优化到 185MB 的过程</p>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center mb-8">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={runAnimation}
          disabled={isAnimating}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
            isAnimating
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
        >
          {isAnimating ? (
            <>
              <Pause className="w-5 h-5" />
              构建中...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              开始构建
            </>
          )}
        </motion.button>
      </div>

      {/* 构建阶段可视化 */}
      <div className="relative mb-8">
        {/* 连接线 */}
        <div className="absolute top-16 left-0 right-0 h-1 bg-gray-200 z-0" />
        
        <div className="relative z-10 grid grid-cols-5 gap-3">
          {stages.map((stage, idx) => {
            const colors = colorClasses[stage.color as keyof typeof colorClasses]
            const isActive = currentStage === stage.id
            const isCompleted = completedStages.includes(stage.id)
            
            return (
              <div key={stage.id} className="relative">
                {/* 阶段卡片 */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    isActive
                      ? `${colors.light} border-${stage.color}-500 shadow-lg scale-105`
                      : isCompleted
                      ? `${colors.light} border-${stage.color}-400`
                      : 'bg-white border-gray-200'
                  }`}
                >
                  {/* 图标 */}
                  <div className="flex items-center justify-center mb-3 relative">
                    <div className={`w-12 h-12 rounded-full ${colors.bg} flex items-center justify-center`}>
                      {stage.id === 'source' && <FileCode className="w-6 h-6 text-white" />}
                      {stage.id === 'dependencies' && <Layers className="w-6 h-6 text-white" />}
                      {stage.id === 'build' && <Box className="w-6 h-6 text-white" />}
                      {stage.id === 'runtime' && <Layers className="w-6 h-6 text-white" />}
                      {stage.id === 'final' && <Box className="w-6 h-6 text-white" />}
                    </div>
                    
                    <AnimatePresence>
                      {isCompleted && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          exit={{ scale: 0 }}
                          className="absolute -top-1 -right-1 bg-green-500 rounded-full p-1"
                        >
                          <CheckCircle className="w-4 h-4 text-white" />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  
                  {/* 名称 */}
                  <h4 className={`font-semibold text-center mb-2 text-sm ${
                    isActive ? colors.text : 'text-gray-700'
                  }`}>
                    {stage.name}
                  </h4>
                  
                  {/* 描述 */}
                  <p className="text-xs text-gray-600 text-center mb-3 leading-relaxed">
                    {stage.description}
                  </p>
                  
                  {/* 大小标签 */}
                  <div className={`text-xs font-semibold text-center px-2 py-1 rounded ${
                    isActive ? `${colors.bg} text-white` : 'bg-gray-100 text-gray-600'
                  }`}>
                    {stage.size}
                  </div>
                  
                  {/* 加载动画 */}
                  <AnimatePresence>
                    {isActive && isAnimating && (
                      <motion.div
                        initial={{ width: '0%' }}
                        animate={{ width: '100%' }}
                        transition={{ duration: 1.5 }}
                        className={`absolute bottom-0 left-0 h-1 ${colors.bg} rounded-b-lg`}
                      />
                    )}
                  </AnimatePresence>
                </motion.div>

                {/* 箭头 */}
                {idx < stages.length - 1 && (
                  <div className="absolute top-16 -right-4 z-20">
                    <ArrowRight className={`w-6 h-6 ${
                      isCompleted ? 'text-green-500' : 'text-gray-300'
                    } transition-colors`} />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* 文件列表 */}
      <div className="bg-white rounded-lg p-6 shadow-inner mb-6">
        <h4 className="text-lg font-semibold text-gray-800 mb-4">当前阶段文件</h4>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h5 className="text-sm font-medium text-gray-700 mb-2">包含的文件/模块</h5>
            <div className="space-y-1">
              {stages.find(s => s.id === currentStage)?.files.map((file, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center gap-2 text-sm text-gray-600"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                  {file}
                </motion.div>
              ))}
            </div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h5 className="text-sm font-medium text-gray-700 mb-3">大小变化</h5>
            <div className="space-y-2">
              {stages.slice(0, stages.findIndex(s => s.id === currentStage) + 1).map((stage, idx) => (
                <div key={stage.id} className="flex items-center justify-between">
                  <span className="text-xs text-gray-600">{stage.name}</span>
                  <span className={`text-sm font-semibold ${
                    stage.id === 'final' ? 'text-green-600' : 'text-gray-700'
                  }`}>
                    {stage.size}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Dockerfile 代码 */}
      <div className="bg-gray-900 text-gray-100 rounded-lg p-6 overflow-hidden">
        <div className="flex items-center gap-2 mb-4">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <span className="text-sm text-gray-400 ml-2">Dockerfile</span>
        </div>

        <pre className="text-sm overflow-x-auto">
          <code>{dockerfileCode}</code>
        </pre>
      </div>

      {/* 优化提示 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-6 p-4 bg-green-50 border-l-4 border-green-500 rounded-r-lg"
      >
        <div className="flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
          <div>
            <h5 className="font-semibold text-green-900 mb-2">镜像优化效果</h5>
            <div className="text-sm text-green-800 space-y-1">
              <p>• 单阶段构建: <span className="line-through">480 MB</span></p>
              <p>• 多阶段构建: <span className="font-semibold">185 MB</span> (减少 61%)</p>
              <p>• 启动时间: 从 15s 降低到 3s</p>
              <p>• 攻击面减小: 移除编译工具和开发依赖</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

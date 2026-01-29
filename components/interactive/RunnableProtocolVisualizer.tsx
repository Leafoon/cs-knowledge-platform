'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Zap, Layers, RotateCw } from 'lucide-react';

interface RunnableMethod {
  name: string;
  description: string;
  sync: boolean;
  streaming: boolean;
  batch: boolean;
  icon: React.ReactNode;
  example: string;
  output: string;
}

const methods: RunnableMethod[] = [
  {
    name: 'invoke(input)',
    description: '同步调用，阻塞直到结果返回',
    sync: true,
    streaming: false,
    batch: false,
    icon: <Play className="w-5 h-5" />,
    example: 'result = model.invoke([message])',
    output: 'AIMessage(content="Hello!")'
  },
  {
    name: 'ainvoke(input)',
    description: '异步调用，非阻塞执行',
    sync: false,
    streaming: false,
    batch: false,
    icon: <Zap className="w-5 h-5" />,
    example: 'result = await model.ainvoke([message])',
    output: 'AIMessage(content="Hello!")'
  },
  {
    name: 'stream(input)',
    description: '同步流式输出，逐块返回',
    sync: true,
    streaming: true,
    batch: false,
    icon: <RotateCw className="w-5 h-5" />,
    example: 'for chunk in model.stream([message]):\n    print(chunk.content)',
    output: 'H...e...l...l...o...!'
  },
  {
    name: 'astream(input)',
    description: '异步流式输出',
    sync: false,
    streaming: true,
    batch: false,
    icon: <Zap className="w-5 h-5" />,
    example: 'async for chunk in model.astream([message]):\n    print(chunk.content)',
    output: 'H...e...l...l...o...!'
  },
  {
    name: 'batch(inputs)',
    description: '批量处理多个输入',
    sync: true,
    streaming: false,
    batch: true,
    icon: <Layers className="w-5 h-5" />,
    example: 'results = model.batch([msg1, msg2, msg3])',
    output: '[AIMessage(...), AIMessage(...), AIMessage(...)]'
  },
  {
    name: 'abatch(inputs)',
    description: '异步批量处理',
    sync: false,
    streaming: false,
    batch: true,
    icon: <Layers className="w-5 h-5" />,
    example: 'results = await model.abatch([msg1, msg2, msg3])',
    output: '[AIMessage(...), AIMessage(...), AIMessage(...)]'
  }
];

export default function RunnableProtocolVisualizer() {
  const [selectedMethod, setSelectedMethod] = useState<RunnableMethod>(methods[0]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStep, setExecutionStep] = useState(0);

  const executeDemo = async () => {
    setIsExecuting(true);
    setExecutionStep(0);

    const steps = selectedMethod.streaming ? 6 : 3;
    
    for (let i = 0; i <= steps; i++) {
      setExecutionStep(i);
      await new Promise(resolve => setTimeout(resolve, 600));
    }

    setIsExecuting(false);
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-blue-900/20 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          Runnable 协议可视化
        </h3>
        <p className="text-slate-400">
          探索 LangChain 统一接口的六种调用方式
        </p>
      </div>

      {/* 方法选择器 */}
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        {methods.map((method) => (
          <motion.button
            key={method.name}
            onClick={() => setSelectedMethod(method)}
            className={`
              p-4 rounded-xl border-2 transition-all text-left
              ${selectedMethod.name === method.name
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
              }
            `}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center gap-3 mb-2">
              <div className={`
                p-2 rounded-lg
                ${selectedMethod.name === method.name
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-700 text-slate-400'
                }
              `}>
                {method.icon}
              </div>
              <span className="font-mono text-sm text-white">
                {method.name}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {method.description}
            </p>
            <div className="flex gap-2 mt-3">
              {method.sync && (
                <span className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400">
                  同步
                </span>
              )}
              {!method.sync && (
                <span className="px-2 py-1 text-xs rounded bg-purple-500/20 text-purple-400">
                  异步
                </span>
              )}
              {method.streaming && (
                <span className="px-2 py-1 text-xs rounded bg-orange-500/20 text-orange-400">
                  流式
                </span>
              )}
              {method.batch && (
                <span className="px-2 py-1 text-xs rounded bg-blue-500/20 text-blue-400">
                  批量
                </span>
              )}
            </div>
          </motion.button>
        ))}
      </div>

      {/* 执行演示区 */}
      <div className="bg-slate-800 rounded-xl p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-lg font-semibold text-white">执行演示</h4>
          <motion.button
            onClick={executeDemo}
            disabled={isExecuting}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 
                     text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Play className="w-4 h-4" />
            {isExecuting ? '执行中...' : '运行示例'}
          </motion.button>
        </div>

        {/* 代码示例 */}
        <div className="bg-slate-900 rounded-lg p-4 mb-4 font-mono text-sm">
          <pre className="text-green-400 whitespace-pre-wrap">
            {selectedMethod.example}
          </pre>
        </div>

        {/* 执行流程可视化 */}
        <div className="space-y-3">
          {/* 步骤 1: 输入 */}
          <motion.div
            className={`
              p-4 rounded-lg border-2 transition-all
              ${executionStep >= 1 ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700'}
            `}
            initial={{ opacity: 0.5 }}
            animate={{ opacity: executionStep >= 1 ? 1 : 0.5 }}
          >
            <div className="flex items-center gap-3">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center font-bold
                ${executionStep >= 1 ? 'bg-blue-500 text-white' : 'bg-slate-700 text-slate-400'}
              `}>
                1
              </div>
              <div className="flex-1">
                <div className="text-white font-medium">接收输入</div>
                <div className="text-slate-400 text-sm">
                  {selectedMethod.batch ? 'inputs = [msg1, msg2, msg3]' : 'input = [HumanMessage(content="Hello")]'}
                </div>
              </div>
              {executionStep === 1 && (
                <motion.div
                  className="w-2 h-2 bg-blue-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                />
              )}
            </div>
          </motion.div>

          {/* 步骤 2: 处理 */}
          <motion.div
            className={`
              p-4 rounded-lg border-2 transition-all
              ${executionStep >= 2 ? 'border-purple-500 bg-purple-500/10' : 'border-slate-700'}
            `}
            initial={{ opacity: 0.5 }}
            animate={{ opacity: executionStep >= 2 ? 1 : 0.5 }}
          >
            <div className="flex items-center gap-3">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center font-bold
                ${executionStep >= 2 ? 'bg-purple-500 text-white' : 'bg-slate-700 text-slate-400'}
              `}>
                2
              </div>
              <div className="flex-1">
                <div className="text-white font-medium">
                  {selectedMethod.streaming ? '流式处理' : 'LLM 处理'}
                </div>
                <div className="text-slate-400 text-sm">
                  {selectedMethod.sync ? '同步等待' : '异步执行'}
                  {selectedMethod.batch && ' (并行处理多个输入)'}
                </div>
              </div>
              {executionStep === 2 && (
                <motion.div
                  className="w-2 h-2 bg-purple-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                />
              )}
            </div>
          </motion.div>

          {/* 步骤 3: 输出 (流式额外步骤) */}
          {selectedMethod.streaming && executionStep >= 3 && (
            <AnimatePresence>
              {[3, 4, 5].map((step) => (
                executionStep >= step && (
                  <motion.div
                    key={step}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                    className="p-4 rounded-lg border-2 border-orange-500 bg-orange-500/10 ml-12"
                  >
                    <div className="flex items-center gap-3">
                      <div className="text-orange-400 font-mono text-sm">
                        Chunk {step - 2}: "{selectedMethod.output.charAt((step - 3) * 2)}"
                      </div>
                      {executionStep === step && (
                        <motion.div
                          className="w-2 h-2 bg-orange-500 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{ repeat: Infinity, duration: 0.8 }}
                        />
                      )}
                    </div>
                  </motion.div>
                )
              ))}
            </AnimatePresence>
          )}

          {/* 步骤 3/6: 返回结果 */}
          <motion.div
            className={`
              p-4 rounded-lg border-2 transition-all
              ${executionStep >= 3 || (selectedMethod.streaming && executionStep >= 6) 
                ? 'border-green-500 bg-green-500/10' 
                : 'border-slate-700'}
            `}
            initial={{ opacity: 0.5 }}
            animate={{ 
              opacity: executionStep >= 3 || (selectedMethod.streaming && executionStep >= 6) ? 1 : 0.5 
            }}
          >
            <div className="flex items-center gap-3">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center font-bold
                ${executionStep >= 3 || (selectedMethod.streaming && executionStep >= 6)
                  ? 'bg-green-500 text-white' 
                  : 'bg-slate-700 text-slate-400'}
              `}>
                {selectedMethod.streaming ? '6' : '3'}
              </div>
              <div className="flex-1">
                <div className="text-white font-medium">返回结果</div>
                <div className="text-slate-400 text-sm font-mono">
                  {selectedMethod.output}
                </div>
              </div>
              {(executionStep === 3 || (selectedMethod.streaming && executionStep === 6)) && (
                <motion.div
                  className="w-2 h-2 bg-green-500 rounded-full"
                  animate={{ scale: [1, 1.5, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                />
              )}
            </div>
          </motion.div>
        </div>
      </div>

      {/* 对比表格 */}
      <div className="bg-slate-800 rounded-xl p-6">
        <h4 className="text-lg font-semibold text-white mb-4">方法选择指南</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-slate-300">方法</th>
                <th className="text-left py-3 px-4 text-slate-300">使用场景</th>
                <th className="text-left py-3 px-4 text-slate-300">优势</th>
                <th className="text-left py-3 px-4 text-slate-300">劣势</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 font-mono text-green-400">invoke()</td>
                <td className="py-3 px-4 text-slate-300">简单脚本、单次调用</td>
                <td className="py-3 px-4 text-slate-400">代码简洁</td>
                <td className="py-3 px-4 text-slate-400">阻塞主线程</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 font-mono text-purple-400">ainvoke()</td>
                <td className="py-3 px-4 text-slate-300">Web 后端、并发场景</td>
                <td className="py-3 px-4 text-slate-400">高效并发</td>
                <td className="py-3 px-4 text-slate-400">需异步上下文</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 font-mono text-orange-400">stream()</td>
                <td className="py-3 px-4 text-slate-300">聊天界面、实时反馈</td>
                <td className="py-3 px-4 text-slate-400">用户体验好</td>
                <td className="py-3 px-4 text-slate-400">处理复杂</td>
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 font-mono text-blue-400">batch()</td>
                <td className="py-3 px-4 text-slate-300">数据处理、批量任务</td>
                <td className="py-3 px-4 text-slate-400">节省请求次数</td>
                <td className="py-3 px-4 text-slate-400">内存占用大</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

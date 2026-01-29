'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw } from 'lucide-react';

interface ChainNode {
  id: string;
  name: string;
  type: 'prompt' | 'model' | 'parser' | 'function';
  description: string;
  input: string;
  output: string;
}

const chainNodes: ChainNode[] = [
  {
    id: 'input',
    name: 'Input',
    type: 'prompt',
    description: '接收原始输入',
    input: '{ text: "Hello", language: "French" }',
    output: '{ text: "Hello", language: "French" }'
  },
  {
    id: 'prompt',
    name: 'ChatPromptTemplate',
    type: 'prompt',
    description: '生成提示文本',
    input: '{ text: "Hello", language: "French" }',
    output: '[HumanMessage(content="Translate to French: Hello")]'
  },
  {
    id: 'model',
    name: 'ChatOpenAI',
    type: 'model',
    description: 'LLM 处理',
    input: '[HumanMessage(...)]',
    output: 'AIMessage(content="Bonjour")'
  },
  {
    id: 'parser',
    name: 'StrOutputParser',
    type: 'parser',
    description: '提取文本内容',
    input: 'AIMessage(content="Bonjour")',
    output: '"Bonjour"'
  }
];

const getNodeColor = (type: string) => {
  switch (type) {
    case 'prompt':
      return 'from-blue-500 to-cyan-500';
    case 'model':
      return 'from-purple-500 to-pink-500';
    case 'parser':
      return 'from-green-500 to-emerald-500';
    case 'function':
      return 'from-orange-500 to-red-500';
    default:
      return 'from-slate-500 to-slate-600';
  }
};

export default function ChainGraphVisualizer() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedNode, setSelectedNode] = useState<ChainNode | null>(null);

  React.useEffect(() => {
    if (!isPlaying) return;

    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= chainNodes.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1500);

    return () => clearInterval(timer);
  }, [isPlaying]);

  const togglePlay = () => {
    if (currentStep >= chainNodes.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-blue-900/20 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          链执行流程可视化
        </h3>
        <p className="text-slate-400">
          查看数据在链中的流动过程
        </p>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-4 mb-8">
        <motion.button
          onClick={togglePlay}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg 
                   font-medium transition-colors flex items-center gap-2"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isPlaying ? (
            <>
              <Pause className="w-5 h-5" />
              暂停
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              {currentStep >= chainNodes.length - 1 ? '重新播放' : '播放'}
            </>
          )}
        </motion.button>

        <motion.button
          onClick={reset}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg 
                   font-medium transition-colors flex items-center gap-2"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw className="w-5 h-5" />
          重置
        </motion.button>
      </div>

      {/* 链可视化 */}
      <div className="mb-8">
        <div className="flex items-center justify-center gap-6">
          {chainNodes.map((node, index) => (
            <React.Fragment key={node.id}>
              {/* 节点 */}
              <motion.div
                className="relative"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ 
                  scale: 1, 
                  opacity: 1,
                  y: currentStep === index ? -10 : 0
                }}
                transition={{ delay: index * 0.1 }}
              >
                <motion.button
                  onClick={() => setSelectedNode(node)}
                  className={`
                    w-32 h-32 rounded-2xl cursor-pointer
                    bg-gradient-to-br ${getNodeColor(node.type)}
                    shadow-lg hover:shadow-2xl transition-all
                    ${currentStep === index ? 'ring-4 ring-white scale-110' : ''}
                    ${currentStep > index ? 'opacity-60' : ''}
                  `}
                  whileHover={{ scale: currentStep !== index ? 1.05 : 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="p-4 text-center h-full flex flex-col justify-center">
                    <div className="text-white font-bold text-sm mb-2">
                      {node.name}
                    </div>
                    <div className="text-white/80 text-xs">
                      {node.type}
                    </div>
                  </div>

                  {/* 活动指示器 */}
                  {currentStep === index && (
                    <motion.div
                      className="absolute inset-0 rounded-2xl bg-white"
                      initial={{ opacity: 0.3, scale: 1 }}
                      animate={{ opacity: 0, scale: 1.2 }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                    />
                  )}

                  {/* 完成标记 */}
                  {currentStep > index && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 
                               rounded-full flex items-center justify-center"
                    >
                      <span className="text-white text-lg">✓</span>
                    </motion.div>
                  )}
                </motion.button>
              </motion.div>

              {/* 箭头 */}
              {index < chainNodes.length - 1 && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: currentStep > index ? 1 : 0.3 }}
                  className="h-1 w-12 bg-gradient-to-r from-white/30 to-white/10 rounded-full"
                  style={{ transformOrigin: 'left' }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: currentStep > index ? 1 : 0 }}
                    transition={{ duration: 0.5 }}
                  />
                </motion.div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 当前步骤详情 */}
      <div className="bg-slate-800 rounded-xl p-6 mb-6">
        <h4 className="text-lg font-semibold text-white mb-4">
          当前步骤: {chainNodes[currentStep]?.name}
        </h4>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* 输入 */}
          <div>
            <div className="text-sm font-medium text-slate-400 mb-2">输入</div>
            <div className="bg-slate-900 rounded-lg p-4">
              <pre className="text-blue-400 font-mono text-sm whitespace-pre-wrap">
                {chainNodes[currentStep]?.input}
              </pre>
            </div>
          </div>

          {/* 输出 */}
          <div>
            <div className="text-sm font-medium text-slate-400 mb-2">输出</div>
            <div className="bg-slate-900 rounded-lg p-4">
              <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap">
                {chainNodes[currentStep]?.output}
              </pre>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
          <div className="text-blue-300 text-sm">
            {chainNodes[currentStep]?.description}
          </div>
        </div>
      </div>

      {/* 节点详情弹窗 */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedNode(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-slate-800 rounded-2xl p-6 max-w-lg w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className={`
                    text-2xl font-bold bg-gradient-to-r ${getNodeColor(selectedNode.type)}
                    bg-clip-text text-transparent
                  `}>
                    {selectedNode.name}
                  </h3>
                  <p className="text-slate-400 mt-1">{selectedNode.description}</p>
                </div>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-slate-400 hover:text-white text-2xl"
                >
                  ×
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="text-sm font-semibold text-slate-300 mb-2">输入类型</div>
                  <div className="bg-slate-900 rounded-lg p-3 font-mono text-sm text-blue-400">
                    {selectedNode.input}
                  </div>
                </div>

                <div>
                  <div className="text-sm font-semibold text-slate-300 mb-2">输出类型</div>
                  <div className="bg-slate-900 rounded-lg p-3 font-mono text-sm text-green-400">
                    {selectedNode.output}
                  </div>
                </div>

                <div>
                  <div className="text-sm font-semibold text-slate-300 mb-2">节点类型</div>
                  <span className={`
                    px-3 py-1 rounded-full text-sm font-medium text-white
                    bg-gradient-to-r ${getNodeColor(selectedNode.type)}
                  `}>
                    {selectedNode.type}
                  </span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 代码示例 */}
      <div className="bg-slate-900 rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-3">对应代码</div>
        <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义链
prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 组合
chain = prompt | model | parser

# 执行
result = chain.invoke({"text": "Hello", "language": "French"})
# 输出: "Bonjour"`}
        </pre>
      </div>
    </div>
  );
}

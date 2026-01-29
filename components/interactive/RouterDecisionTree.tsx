"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Intent = 'technical' | 'billing' | 'general';

const routes = {
  technical: {
    name: '技术支持',
    description: '处理技术问题、Bug 报告、功能咨询',
    color: 'blue',
    examples: ['登录失败', 'API 报错', '功能如何使用']
  },
  billing: {
    name: '账单查询',
    description: '处理付款、退款、发票相关问题',
    color: 'green',
    examples: ['重复扣费', '申请退款', '下载发票']
  },
  general: {
    name: '通用客服',
    description: '处理一般性咨询和其他问题',
    color: 'purple',
    examples: ['产品介绍', '使用指南', '合作咨询']
  }
};

const testMessages = [
  { text: "我登录时显示密码错误，但密码肯定是对的", intent: 'technical' as Intent },
  { text: "上个月被扣了两次费用，能退一次吗？", intent: 'billing' as Intent },
  { text: "你们公司的产品主要做什么的？", intent: 'general' as Intent },
  { text: "API 返回 500 错误，怎么解决？", intent: 'technical' as Intent },
  { text: "需要开具增值税发票", intent: 'billing' as Intent }
];

export default function RouterDecisionTree() {
  const [selectedMessage, setSelectedMessage] = useState(0);
  const [showRouting, setShowRouting] = useState(false);
  const [routedIntent, setRoutedIntent] = useState<Intent | null>(null);

  const currentMessage = testMessages[selectedMessage];

  const handleRoute = () => {
    setShowRouting(true);
    setTimeout(() => {
      setRoutedIntent(currentMessage.intent);
    }, 1000);
  };

  const handleReset = () => {
    setShowRouting(false);
    setRoutedIntent(null);
  };

  const handleNext = () => {
    handleReset();
    setSelectedMessage((prev) => (prev + 1) % testMessages.length);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
      <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
        路由决策树可视化
      </h3>
      
      <p className="text-slate-600 dark:text-slate-400 mb-6">
        演示智能路由链如何根据用户消息内容，动态选择合适的处理链进行响应
      </p>

      {/* 测试消息选择器 */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
          选择测试消息：
        </label>
        <div className="grid gap-2">
          {testMessages.map((msg, idx) => (
            <button
              key={idx}
              onClick={() => {
                setSelectedMessage(idx);
                handleReset();
              }}
              className={`text-left p-3 rounded-lg border-2 transition-all ${
                selectedMessage === idx
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-slate-200 dark:border-slate-700 hover:border-blue-300'
              }`}
            >
              <div className="font-medium text-slate-900 dark:text-white">
                {msg.text}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                预期路由: {routes[msg.intent].name}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 路由流程可视化 */}
      <div className="bg-slate-50 dark:bg-slate-800/50 rounded-lg p-6 mb-6 min-h-[500px]">
        <div className="flex flex-col items-center">
          {/* 输入消息 */}
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-full max-w-md bg-white dark:bg-slate-900 rounded-lg p-4 border-2 border-blue-500 shadow-lg mb-8"
          >
            <div className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-2">
              用户消息
            </div>
            <div className="text-slate-900 dark:text-white">
              "{currentMessage.text}"
            </div>
          </motion.div>

          {/* 路由器节点 */}
          <AnimatePresence mode="wait">
            {!showRouting ? (
              <motion.button
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                onClick={handleRoute}
                className="w-32 h-32 rounded-full bg-gradient-to-br from-orange-500 to-orange-600 text-white font-bold text-lg shadow-xl hover:shadow-2xl hover:scale-110 transition-all mb-8"
              >
                开始路由
              </motion.button>
            ) : (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1, rotate: routedIntent ? 360 : 0 }}
                transition={{ duration: 1 }}
                className="w-32 h-32 rounded-full bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center text-white font-bold shadow-xl mb-8"
              >
                {routedIntent ? (
                  <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin" />
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* 路由目标 */}
          <div className="grid grid-cols-3 gap-4 w-full max-w-4xl">
            {(Object.keys(routes) as Intent[]).map((intent) => {
              const route = routes[intent];
              const isSelected = routedIntent === intent;
              const colors = {
                blue: 'from-blue-500 to-blue-600',
                green: 'from-green-500 to-green-600',
                purple: 'from-purple-500 to-purple-600'
              };

              return (
                <motion.div
                  key={intent}
                  initial={{ y: 50, opacity: 0 }}
                  animate={{ 
                    y: 0, 
                    opacity: 1,
                    scale: isSelected ? 1.1 : 1,
                    zIndex: isSelected ? 10 : 1
                  }}
                  transition={{ delay: 0.2 }}
                  className={`relative p-4 rounded-lg border-2 ${
                    isSelected
                      ? 'border-orange-500 shadow-2xl'
                      : 'border-slate-300 dark:border-slate-600'
                  } ${isSelected ? 'bg-gradient-to-br ' + colors[route.color as keyof typeof colors] + ' text-white' : 'bg-white dark:bg-slate-800'}`}
                >
                  {isSelected && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute -top-3 -right-3 w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center text-white"
                    >
                      ✓
                    </motion.div>
                  )}

                  <h4 className={`font-bold text-lg mb-2 ${isSelected ? 'text-white' : 'text-slate-900 dark:text-white'}`}>
                    {route.name}
                  </h4>
                  <p className={`text-sm mb-3 ${isSelected ? 'text-white/90' : 'text-slate-600 dark:text-slate-400'}`}>
                    {route.description}
                  </p>
                  
                  <div className={`text-xs ${isSelected ? 'text-white/80' : 'text-slate-500 dark:text-slate-500'}`}>
                    <div className="font-semibold mb-1">示例：</div>
                    <ul className="space-y-1">
                      {route.examples.map((ex, idx) => (
                        <li key={idx}>• {ex}</li>
                      ))}
                    </ul>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* 路由逻辑说明 */}
          {routedIntent && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 max-w-2xl"
            >
              <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                路由决策依据：
              </h5>
              <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                <div>✓ 语义分析：识别关键词和上下文</div>
                <div>✓ 意图分类：使用 LLM 或规则引擎判断</div>
                <div>✓ 动态分发：将请求路由到最合适的专业链</div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-between">
        <button
          onClick={handleReset}
          disabled={!showRouting}
          className="px-6 py-2 rounded-lg font-medium text-sm bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          重置
        </button>
        
        <button
          onClick={handleNext}
          className="px-6 py-2 rounded-lg font-medium text-sm bg-blue-500 text-white hover:bg-blue-600 transition-colors"
        >
          下一个消息 →
        </button>
      </div>

      {/* 代码示例 */}
      <div className="mt-6">
        <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
          路由链代码实现：
        </h4>
        <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-lg p-4 text-sm font-mono overflow-x-auto border border-slate-700">
          <code>{`router_chain = RunnableBranch(
    (
        lambda x: "技术" in x["message"] or "错误" in x["message"],
        technical_chain
    ),
    (
        lambda x: "付款" in x["message"] or "退款" in x["message"],
        billing_chain
    ),
    general_chain  # 默认链
)

result = router_chain.invoke({"message": "..."})`}</code>
        </pre>
      </div>
    </div>
  );
}

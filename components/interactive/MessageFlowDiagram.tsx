'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Bot, Settings, ArrowRight } from 'lucide-react';

interface Message {
  type: 'system' | 'human' | 'ai';
  content: string;
  timestamp: number;
}

export default function MessageFlowDiagram() {
  const [messages, setMessages] = useState<Message[]>([
    { type: 'system', content: 'You are a helpful assistant.', timestamp: Date.now() }
  ]);
  const [userInput, setUserInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const addMessage = async () => {
    if (!userInput.trim()) return;

    // 添加用户消息
    const humanMsg: Message = {
      type: 'human',
      content: userInput,
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, humanMsg]);
    setUserInput('');
    setIsProcessing(true);

    // 模拟 AI 响应
    await new Promise(resolve => setTimeout(resolve, 1500));

    const aiMsg: Message = {
      type: 'ai',
      content: `回复: ${userInput}`,
      timestamp: Date.now()
    };
    setMessages(prev => [...prev, aiMsg]);
    setIsProcessing(false);
  };

  const clearMessages = () => {
    setMessages([messages[0]]);  // 保留系统消息
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'system':
        return <Settings className="w-5 h-5" />;
      case 'human':
        return <User className="w-5 h-5" />;
      case 'ai':
        return <Bot className="w-5 h-5" />;
    }
  };

  const getMessageColor = (type: string) => {
    switch (type) {
      case 'system':
        return 'from-yellow-500 to-orange-500';
      case 'human':
        return 'from-blue-500 to-cyan-500';
      case 'ai':
        return 'from-purple-500 to-pink-500';
    }
  };

  const getMessageBg = (type: string) => {
    switch (type) {
      case 'system':
        return 'bg-yellow-500/10 border-yellow-500/30';
      case 'human':
        return 'bg-blue-500/10 border-blue-500/30';
      case 'ai':
        return 'bg-purple-500/10 border-purple-500/30';
    }
  };

  return (
    <div className="w-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl p-8 shadow-2xl">
      <div className="text-center mb-8">
        <h3 className="text-2xl font-bold text-white mb-2">
          消息流动可视化
        </h3>
        <p className="text-slate-400">
          理解 SystemMessage、HumanMessage、AIMessage 在对话中的流动
        </p>
      </div>

      {/* 消息类型说明 */}
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4 border border-yellow-500/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-yellow-500 to-orange-500">
              <Settings className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-white">SystemMessage</span>
          </div>
          <p className="text-sm text-slate-400">
            系统指令，定义 AI 的角色和行为规则
          </p>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-blue-500/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500">
              <User className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-white">HumanMessage</span>
          </div>
          <p className="text-sm text-slate-400">
            用户输入，代表人类的问题或指令
          </p>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-purple-500/30">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-white">AIMessage</span>
          </div>
          <p className="text-sm text-slate-400">
            AI 回复，模型生成的响应内容
          </p>
        </div>
      </div>

      {/* 消息列表 */}
      <div className="bg-slate-800 rounded-xl p-6 mb-6 min-h-[300px] max-h-[400px] overflow-y-auto">
        <div className="space-y-4">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-4 rounded-lg border-2 ${getMessageBg(msg.type)}`}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  p-2 rounded-lg bg-gradient-to-br ${getMessageColor(msg.type)}
                  flex-shrink-0
                `}>
                  {getMessageIcon(msg.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-white capitalize">
                      {msg.type}Message
                    </span>
                    <span className="text-xs text-slate-500">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-slate-300 break-words">
                    {msg.content}
                  </p>
                </div>
              </div>

              {/* 连接箭头 */}
              {index < messages.length - 1 && (
                <div className="flex justify-center my-2">
                  <ArrowRight className="w-5 h-5 text-slate-600" />
                </div>
              )}
            </motion.div>
          ))}

          {isProcessing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="p-4 rounded-lg border-2 bg-purple-500/10 border-purple-500/30"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white">AI 正在思考</span>
                  <motion.div
                    className="flex gap-1"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                  >
                    <div className="w-2 h-2 bg-purple-500 rounded-full" />
                    <div className="w-2 h-2 bg-purple-500 rounded-full" />
                    <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  </motion.div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* 输入区域 */}
      <div className="flex gap-3">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addMessage()}
          placeholder="输入消息..."
          className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg 
                   text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
          disabled={isProcessing}
        />
        <button
          onClick={addMessage}
          disabled={isProcessing || !userInput.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 
                   text-white rounded-lg font-medium transition-colors"
        >
          发送
        </button>
        <button
          onClick={clearMessages}
          className="px-4 py-3 bg-slate-700 hover:bg-slate-600 
                   text-white rounded-lg font-medium transition-colors"
        >
          清空
        </button>
      </div>

      {/* 代码示例 */}
      <div className="mt-6 bg-slate-900 rounded-lg p-4">
        <div className="text-sm text-slate-400 mb-2">消息列表构建示例：</div>
        <pre className="text-green-400 font-mono text-sm overflow-x-auto">
{`messages = [
    SystemMessage(content="${messages[0]?.content}"),
${messages.slice(1).map(msg => 
  `    ${msg.type === 'human' ? 'HumanMessage' : 'AIMessage'}(content="${msg.content}")`
).join(',\n')}
]`}
        </pre>
      </div>
    </div>
  );
}

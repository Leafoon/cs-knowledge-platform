"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Database, Zap, TrendingUp, Clock } from 'lucide-react';

type MemoryStage = 'buffer' | 'window' | 'summary' | 'vector' | 'entity';

interface ConversationTurn {
  id: number;
  human: string;
  ai: string;
  timestamp: number;
}

const MEMORY_TYPES = [
  {
    id: 'buffer' as MemoryStage,
    name: 'Buffer Memory',
    icon: '💾',
    description: '保存完整对话历史',
    color: 'blue',
    pros: ['实现简单', '完整上下文', '适合短对话'],
    cons: ['Token 消耗大', '不适合长对话'],
  },
  {
    id: 'window' as MemoryStage,
    name: 'Window Memory',
    icon: '🪟',
    description: '滑动窗口，保留最近 N 轮',
    color: 'green',
    pros: ['Token 可控', '保留最新信息', '性能稳定'],
    cons: ['丢失早期信息', '需要调参'],
  },
  {
    id: 'summary' as MemoryStage,
    name: 'Summary Memory',
    icon: '📝',
    description: '压缩历史为摘要',
    color: 'purple',
    pros: ['极大节省 Token', '适合长对话', '保留核心信息'],
    cons: ['可能丢失细节', '摘要需要 LLM 调用'],
  },
  {
    id: 'vector' as MemoryStage,
    name: 'Vector Memory',
    icon: '🔍',
    description: '语义检索相关历史',
    color: 'orange',
    pros: ['智能检索', '适合复杂对话', '语义匹配'],
    cons: ['需要向量库', '检索延迟', '成本较高'],
  },
  {
    id: 'entity' as MemoryStage,
    name: 'Entity Memory',
    icon: '🏷️',
    description: '跟踪实体信息',
    color: 'pink',
    pros: ['个性化强', '实体跟踪', '结构化信息'],
    cons: ['实现复杂', '依赖实体提取'],
  }
];

const SAMPLE_CONVERSATION: ConversationTurn[] = [
  { id: 1, human: "Hi, my name is Alice", ai: "Hello Alice! How can I help you?", timestamp: 0 },
  { id: 2, human: "I'm a software engineer", ai: "Great! What kind of development do you do?", timestamp: 5 },
  { id: 3, human: "I work on machine learning", ai: "ML is such an exciting field!", timestamp: 10 },
  { id: 4, human: "I use Python mostly", ai: "Python is perfect for ML work.", timestamp: 15 },
  { id: 5, human: "Do you remember my name?", ai: "Yes, your name is Alice.", timestamp: 20 },
  { id: 6, human: "What's my job?", ai: "You're a software engineer working on machine learning.", timestamp: 25 },
];

export default function MemoryEvolutionTimeline() {
  const [selectedType, setSelectedType] = useState<MemoryStage>('buffer');
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const memoryType = MEMORY_TYPES.find(m => m.id === selectedType)!;

  const playTimeline = () => {
    setIsPlaying(true);
    setCurrentTurn(0);

    let turn = 0;
    const interval = setInterval(() => {
      if (turn >= SAMPLE_CONVERSATION.length) {
        clearInterval(interval);
        setIsPlaying(false);
        return;
      }
      setCurrentTurn(turn + 1);
      turn++;
    }, 1500);
  };

  const getMemoryContent = () => {
    const visibleTurns = SAMPLE_CONVERSATION.slice(0, currentTurn);
    
    switch (selectedType) {
      case 'buffer':
        return visibleTurns.map(t => `Human: ${t.human}\nAI: ${t.ai}`).join('\n\n');
      
      case 'window':
        const windowSize = 2;
        const windowTurns = visibleTurns.slice(-windowSize);
        return windowTurns.map(t => `Human: ${t.human}\nAI: ${t.ai}`).join('\n\n');
      
      case 'summary':
        if (visibleTurns.length === 0) return '';
        return "Summary: Alice is a software engineer working on machine learning using Python.";
      
      case 'vector':
        if (currentTurn === 5) {
          return "Retrieved relevant: Turn 1 (name=Alice)";
        } else if (currentTurn === 6) {
          return "Retrieved relevant: Turn 2 (job=engineer), Turn 3 (field=ML)";
        }
        return visibleTurns.slice(-1).map(t => `Human: ${t.human}\nAI: ${t.ai}`).join('\n\n');
      
      case 'entity':
        const entities: Record<string, string> = {};
        visibleTurns.forEach(t => {
          if (t.id === 1) entities['name'] = 'Alice';
          if (t.id === 2) entities['job'] = 'Software Engineer';
          if (t.id === 3) entities['field'] = 'Machine Learning';
          if (t.id === 4) entities['language'] = 'Python';
        });
        return Object.entries(entities).map(([k, v]) => `${k}: ${v}`).join('\n');
      
      default:
        return '';
    }
  };

  const getTokenCount = () => {
    const content = getMemoryContent();
    return Math.ceil(content.split(' ').length * 1.3); // 简化估算
  };

  const getColorClasses = (color: string) => {
    const colors = {
      blue: 'bg-blue-500/10 text-blue-700 border-blue-200',
      green: 'bg-green-500/10 text-green-700 border-green-200',
      purple: 'bg-purple-500/10 text-purple-700 border-purple-200',
      orange: 'bg-orange-500/10 text-orange-700 border-orange-200',
      pink: 'bg-pink-500/10 text-pink-700 border-pink-200'
    };
    return colors[color as keyof typeof colors];
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl border border-slate-200">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-slate-800 mb-2">Memory Evolution Timeline</h3>
        <p className="text-slate-600">对比不同记忆类型随对话演进的存储策略</p>
      </div>

      {/* Memory Type Selection */}
      <div className="mb-6 grid grid-cols-2 md:grid-cols-5 gap-3">
        {MEMORY_TYPES.map(type => (
          <button
            key={type.id}
            onClick={() => { setSelectedType(type.id); setCurrentTurn(0); }}
            disabled={isPlaying}
            className={`p-4 rounded-lg border-2 transition-all text-left ${
              selectedType === type.id
                ? `${getColorClasses(type.color)} border-current shadow-lg`
                : 'bg-white border-slate-200 hover:border-slate-300'
            } disabled:opacity-50`}
          >
            <div className="text-2xl mb-2">{type.icon}</div>
            <div className="font-semibold text-sm">{type.name}</div>
            <div className="text-xs opacity-70 mt-1">{type.description}</div>
          </button>
        ))}
      </div>

      {/* Details */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-slate-200">
        <h4 className="font-semibold text-slate-800 mb-3">{memoryType.name} 特性</h4>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <div className="text-sm font-medium text-green-700 mb-2">✓ 优点：</div>
            <ul className="text-sm text-slate-600 space-y-1">
              {memoryType.pros.map((pro, idx) => (
                <li key={idx}>• {pro}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-sm font-medium text-red-700 mb-2">✗ 缺点：</div>
            <ul className="text-sm text-slate-600 space-y-1">
              {memoryType.cons.map((con, idx) => (
                <li key={idx}>• {con}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Timeline Controls */}
      <div className="mb-6 flex items-center justify-between p-4 bg-white rounded-lg border border-slate-200">
        <div className="flex items-center gap-4">
          <button
            onClick={playTimeline}
            disabled={isPlaying}
            className="px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:bg-slate-300 transition-colors"
          >
            {isPlaying ? '播放中...' : '开始对话'}
          </button>
          <button
            onClick={() => { setCurrentTurn(0); setIsPlaying(false); }}
            className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors"
          >
            重置
          </button>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <Clock className="w-4 h-4" />
          <span>对话轮次: {currentTurn}/{SAMPLE_CONVERSATION.length}</span>
        </div>
      </div>

      {/* Conversation Display */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Left: Conversation */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-4">
            <MessageSquare className="w-5 h-5 text-slate-600" />
            <h4 className="font-semibold text-slate-800">对话历史</h4>
          </div>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {SAMPLE_CONVERSATION.slice(0, currentTurn).map((turn, idx) => (
                <motion.div
                  key={turn.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-2"
                >
                  <div className="flex gap-2">
                    <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-sm font-bold flex-shrink-0">
                      👤
                    </div>
                    <div className="flex-1 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-sm text-slate-800">{turn.human}</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center text-sm font-bold flex-shrink-0">
                      🤖
                    </div>
                    <div className="flex-1 p-3 bg-purple-50 rounded-lg border border-purple-200">
                      <p className="text-sm text-slate-800">{turn.ai}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Right: Memory Content */}
        <div className="bg-white rounded-lg border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-4">
            <Database className="w-5 h-5 text-slate-600" />
            <h4 className="font-semibold text-slate-800">记忆内容</h4>
            <span className="ml-auto text-xs text-slate-500">
              ~{getTokenCount()} tokens
            </span>
          </div>
          
          <div className={`p-4 rounded-lg border-2 min-h-[300px] ${getColorClasses(memoryType.color)}`}>
            <pre className="text-xs font-mono whitespace-pre-wrap">
              {getMemoryContent() || '(空)'}
            </pre>
          </div>

          {selectedType === 'window' && currentTurn > 0 && (
            <div className="mt-3 p-2 bg-yellow-50 rounded border border-yellow-200 text-xs text-yellow-800">
              💡 窗口大小: 2，只保留最近 2 轮对话
            </div>
          )}
          {selectedType === 'summary' && currentTurn > 2 && (
            <div className="mt-3 p-2 bg-purple-50 rounded border border-purple-200 text-xs text-purple-800">
              💡 自动压缩为摘要，节省约 70% Token
            </div>
          )}
          {selectedType === 'vector' && currentTurn >= 5 && (
            <div className="mt-3 p-2 bg-orange-50 rounded border border-orange-200 text-xs text-orange-800">
              💡 基于语义相似度检索相关历史
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-indigo-600">{currentTurn}</div>
          <div className="text-sm text-slate-600 mt-1">已处理轮次</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-purple-600">{getTokenCount()}</div>
          <div className="text-sm text-slate-600 mt-1">当前 Token 数</div>
        </div>
        <div className="bg-white rounded-lg border border-slate-200 p-4 text-center">
          <div className="text-3xl font-bold text-green-600">
            {selectedType === 'summary' ? '30%' : selectedType === 'window' ? '50%' : '100%'}
          </div>
          <div className="text-sm text-slate-600 mt-1">Token 使用率</div>
        </div>
      </div>
    </div>
  );
}

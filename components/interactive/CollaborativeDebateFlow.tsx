"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type Agent = {
  id: string;
  name: string;
  role: string;
  color: string;
  position: { x: number; y: number };
};

type Message = {
  from: string;
  to: string;
  content: string;
  round: number;
};

const agents: Agent[] = [
  { id: 'optimist', name: 'Optimist', role: '乐观派', color: '#10B981', position: { x: 20, y: 50 } },
  { id: 'pessimist', name: 'Pessimist', role: '悲观派', color: '#EF4444', position: { x: 50, y: 20 } },
  { id: 'analyst', name: 'Analyst', role: '分析师', color: '#3B82F6', position: { x: 80, y: 50 } },
  { id: 'moderator', name: 'Moderator', role: '主持人', color: '#8B5CF6', position: { x: 50, y: 80 } },
];

const debateScenario = {
  topic: "AI 是否会在 2030 年取代大部分程序员工作？",
  rounds: [
    {
      round: 1,
      messages: [
        { from: 'optimist', to: 'all', content: 'AI 会提升效率，但不会完全取代。程序员将专注于创意和架构。' },
        { from: 'pessimist', to: 'all', content: '历史表明自动化总会减少就业。初级开发岗位将大量消失。' },
        { from: 'analyst', to: 'all', content: '数据显示：AI 辅助编码已提升 40% 效率，但复杂系统设计仍需人类。' },
      ],
    },
    {
      round: 2,
      messages: [
        { from: 'moderator', to: 'all', content: '请聚焦：哪些具体岗位最容易受影响？' },
        { from: 'optimist', to: 'pessimist', content: '重复性任务会被替代，但新岗位会出现（AI 训练师、提示工程师）。' },
        { from: 'pessimist', to: 'optimist', content: '新岗位数量远少于被替代的岗位，存在结构性失业风险。' },
        { from: 'analyst', to: 'all', content: '案例：GitHub Copilot 用户调查显示 88% 开发者认为提升了生产力，但无人因此失业。' },
      ],
    },
    {
      round: 3,
      messages: [
        { from: 'moderator', to: 'all', content: '请提出具体建议。' },
        { from: 'optimist', to: 'all', content: '建议：学习 AI 工具，专注领域知识和软技能。' },
        { from: 'pessimist', to: 'all', content: '建议：政府需制定再培训计划，企业应负责任地引入自动化。' },
        { from: 'analyst', to: 'all', content: '综合建议：个人持续学习 + 行业标准制定 + 社会安全网建设。' },
      ],
    },
  ],
};

export default function CollaborativeDebateFlow() {
  const [currentRound, setCurrentRound] = useState(0);
  const [displayedMessages, setDisplayedMessages] = useState<Message[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);

  const startDebate = () => {
    setIsPlaying(true);
    setDisplayedMessages([]);
    setCurrentRound(0);
    playRound(0);
  };

  const playRound = (roundIndex: number) => {
    if (roundIndex >= debateScenario.rounds.length) {
      setIsPlaying(false);
      return;
    }

    const round = debateScenario.rounds[roundIndex];
    let messageIndex = 0;

    const interval = setInterval(() => {
      if (messageIndex >= round.messages.length) {
        clearInterval(interval);
        setTimeout(() => {
          setCurrentRound(roundIndex + 1);
          playRound(roundIndex + 1);
        }, 2000);
        return;
      }

      const msg = round.messages[messageIndex];
      setDisplayedMessages(prev => [...prev, { ...msg, round: round.round }]);
      messageIndex++;
    }, 1500);
  };

  const reset = () => {
    setIsPlaying(false);
    setDisplayedMessages([]);
    setCurrentRound(0);
  };

  const getAgentColor = (agentId: string) => {
    return agents.find(a => a.id === agentId)?.color || '#6B7280';
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-center mb-2 text-slate-800">
        Collaborative Debate Flow
      </h3>
      <p className="text-center text-slate-600 mb-6">
        协作式辩论：多 Agent 平等交流，通过观点碰撞达成共识
      </p>

      {/* 辩题 */}
      <div className="bg-white rounded-lg p-4 mb-6 shadow-md border-l-4 border-purple-500">
        <div className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-1">
          辩题
        </div>
        <div className="text-lg font-medium text-slate-800">
          {debateScenario.topic}
        </div>
      </div>

      {/* Agent 网络图 */}
      <div className="relative bg-white rounded-lg p-8 mb-6 shadow-md" style={{ height: '300px' }}>
        <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: 'none' }}>
          {/* 连接线 */}
          {agents.map((agent, i) =>
            agents.slice(i + 1).map((otherAgent, j) => {
              const hasMessage = displayedMessages.some(
                msg =>
                  (msg.from === agent.id && msg.to === otherAgent.id) ||
                  (msg.from === otherAgent.id && msg.to === agent.id) ||
                  msg.to === 'all'
              );
              return (
                <line
                  key={`${agent.id}-${otherAgent.id}`}
                  x1={`${agent.position.x}%`}
                  y1={`${agent.position.y}%`}
                  x2={`${otherAgent.position.x}%`}
                  y2={`${otherAgent.position.y}%`}
                  stroke={hasMessage ? '#8B5CF6' : '#E5E7EB'}
                  strokeWidth={hasMessage ? 2 : 1}
                  strokeDasharray={hasMessage ? '0' : '4'}
                />
              );
            })
          )}
        </svg>

        {/* Agents */}
        {agents.map(agent => {
          const isActive = displayedMessages[displayedMessages.length - 1]?.from === agent.id;
          return (
            <motion.div
              key={agent.id}
              className="absolute"
              style={{
                left: `${agent.position.x}%`,
                top: `${agent.position.y}%`,
                transform: 'translate(-50%, -50%)',
              }}
              animate={{
                scale: isActive ? 1.2 : 1,
              }}
            >
              <div
                className="w-20 h-20 rounded-full flex flex-col items-center justify-center text-white shadow-lg border-4 border-white"
                style={{ backgroundColor: agent.color }}
              >
                <div className="text-xs font-bold">{agent.name}</div>
                <div className="text-xs opacity-80">{agent.role}</div>
              </div>
              {isActive && (
                <motion.div
                  className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-400 rounded-full border-2 border-white"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                />
              )}
            </motion.div>
          );
        })}
      </div>

      {/* 消息历史 */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6" style={{ maxHeight: '400px', overflowY: 'auto' }}>
        <h4 className="font-semibold text-lg mb-4 text-slate-800 sticky top-0 bg-white pb-2">
          辩论记录
        </h4>
        <div className="space-y-4">
          <AnimatePresence>
            {displayedMessages.map((msg, index) => {
              const agent = agents.find(a => a.id === msg.from);
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex gap-3"
                >
                  <div
                    className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-bold"
                    style={{ backgroundColor: agent?.color }}
                  >
                    {agent?.name[0]}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-sm text-slate-800">{agent?.name}</span>
                      <span className="text-xs text-slate-500">
                        Round {msg.round}
                        {msg.to !== 'all' && ` → ${agents.find(a => a.id === msg.to)?.name}`}
                      </span>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 text-sm text-slate-700">
                      {msg.content}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
          {displayedMessages.length === 0 && (
            <div className="text-center text-slate-400 py-8">
              点击"开始辩论"查看 Agents 如何协作讨论
            </div>
          )}
        </div>
      </div>

      {/* 进度指示 */}
      <div className="flex items-center justify-center gap-2 mb-6">
        {debateScenario.rounds.map((_, index) => (
          <div
            key={index}
            className={`h-2 rounded-full transition-all ${
              index < currentRound
                ? 'w-8 bg-purple-600'
                : index === currentRound
                ? 'w-12 bg-purple-400'
                : 'w-8 bg-slate-200'
            }`}
          />
        ))}
      </div>

      {/* 控制按钮 */}
      <div className="flex justify-center gap-4">
        <button
          onClick={startDebate}
          disabled={isPlaying}
          className="px-6 py-2 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
        >
          {isPlaying ? '辩论进行中...' : '开始辩论'}
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 bg-slate-600 text-white rounded-lg font-medium hover:bg-slate-700 transition-colors"
        >
          重置
        </button>
      </div>

      {/* 说明 */}
      <div className="mt-6 p-4 bg-purple-50 rounded-lg border border-purple-200">
        <div className="text-sm text-slate-700">
          <span className="font-semibold">协作特点：</span>
          无中心节点，Agents 平等交流；主持人引导但不控制；通过多轮对话逐步收敛到共识。
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Lightbulb, Puzzle, Settings2, Network, Zap, Trophy, RotateCcw, CheckCircle, XCircle } from "lucide-react";

interface Principle {
  id: string;
  title: string;
  icon: React.ReactNode;
  color: string;
  summary: string;
  definition: string;
  examples: { os: string; detail: string }[];
  quiz: { scenario: string; correct: boolean; explanation: string }[];
}

const PRINCIPLES: Principle[] = [
  {
    id: "kiss",
    title: "简单性 (KISS)",
    icon: <Lightbulb className="w-5 h-5" />,
    color: "text-amber-500",
    summary: "系统设计应追求尽可能简单，但不能过度简化",
    definition: "KISS 原则要求设计者在满足需求的前提下选择最简单的方案。简单的系统更容易理解、维护和调试。",
    examples: [
      { os: "Unix/Linux", detail: "read()/write() 统一接口覆盖所有设备" },
      { os: "xv6", detail: "约 100 行核心调度代码" },
    ],
    quiz: [
      { scenario: "为一个新的块设备设计驱动接口，你会选择：", correct: true, explanation: "复用已有的 read/write/ioctl 接口比设计全新接口更简单" },
      { scenario: "为每个设备类型设计完全不同的 API", correct: false, explanation: "这违反了 KISS——增加了学习和维护成本" },
    ],
  },
  {
    id: "mechanism",
    title: "机制与策略分离",
    icon: <Settings2 className="w-5 h-5" />,
    color: "text-blue-500",
    summary: "提供机制（如何做），让用户决定策略（做什么）",
    definition: "机制是可重用的构建块，策略是使用这些构建块的决策。分离两者使系统更灵活。",
    examples: [
      { os: "Linux 调度器", detail: "CFS 提供公平调度机制，nice 值控制优先级策略" },
      { os: "页面置换", detail: "LRU 框架是机制，具体阈值是策略" },
    ],
    quiz: [
      { scenario: "操作系统提供 mmap() 系统调用", correct: true, explanation: "mmap 是机制（内存映射），具体映射什么是策略" },
      { scenario: "内核硬编码只允许 3 个进程", correct: false, explanation: "硬编码策略限制了灵活性" },
    ],
  },
  {
    id: "modularity",
    title: "模块化",
    icon: <Puzzle className="w-5 h-5" />,
    color: "text-emerald-500",
    summary: "系统应分解为独立、可替换的模块",
    definition: "模块化降低了复杂度，允许独立开发和测试，支持可插拔的组件替换。",
    examples: [
      { os: "Linux 内核", detail: "可加载内核模块 (LKM) 动态插入/卸载" },
      { os: "微内核", detail: "文件系统、驱动运行在用户态" },
    ],
    quiz: [
      { scenario: "将文件系统实现为可插拔模块", correct: true, explanation: "允许在不重编译内核的情况下替换文件系统" },
      { scenario: "将所有功能写在一个巨大函数中", correct: false, explanation: "违反模块化，难以维护和测试" },
    ],
  },
  {
    id: "e2e",
    title: "端到端论证",
    icon: <Network className="w-5 h-5" />,
    color: "text-purple-500",
    summary: "某些功能在系统高层实现比低层更有效",
    definition: "如果一个功能必须由端点检查才能保证正确性，那么在低层实现它可能是冗余的。",
    examples: [
      { os: "TCP/IP", detail: "端点重传 vs 链路层重传" },
      { os: "NFS", detail: "无状态服务器，一致性由客户端保证" },
    ],
    quiz: [
      { scenario: "在应用层实现加密", correct: true, explanation: "端到端加密比链路层加密更安全" },
      { scenario: "在每层网络协议都加密数据", correct: false, explanation: "过度实现——如果端点必须加密，中间层加密是冗余的" },
    ],
  },
  {
    id: "endian",
    title: "小即是美",
    icon: <Zap className="w-5 h-5" />,
    color: "text-rose-500",
    summary: "简洁的设计比复杂的设计更可靠",
    definition: "代码量越少，bug 越少，攻击面越小。在满足需求的前提下，选择最精简的实现。",
    examples: [
      { os: "Plan 9", detail: "比 Unix 更彻底的抽象，但核心代码量极小" },
      { os: "seL4", detail: "经过形式化验证的微内核，仅约 1 万行代码" },
    ],
    quiz: [
      { scenario: "用 100 行代码实现调度器", correct: true, explanation: "更少的代码意味着更少的 bug 和攻击面" },
      { scenario: "为了覆盖所有边界情况写 5000 行调度器", correct: false, explanation: "过度设计——应该问是否真的需要这些情况" },
    ],
  },
];

type Mode = "learn" | "quiz";

export default function DesignPrinciplesMap() {
  const [mode, setMode] = useState<Mode>("learn");
  const [selected, setSelected] = useState<string | null>(null);
  const [quizIdx, setQuizIdx] = useState(0);
  const [answered, setAnswered] = useState<boolean | null>(null);
  const [score, setScore] = useState(0);
  const [total, setTotal] = useState(0);

  const active = PRINCIPLES.find((p) => p.id === selected);
  const currentPrinciple = PRINCIPLES[quizIdx % PRINCIPLES.length];

  const handleQuizAnswer = useCallback((correct: boolean) => {
    setAnswered(correct);
    setTotal((t) => t + 1);
    if (correct) setScore((s) => s + 1);
    setTimeout(() => {
      setQuizIdx((i) => i + 1);
      setAnswered(null);
    }, 1500);
  }, []);

  const resetQuiz = () => {
    setQuizIdx(0);
    setScore(0);
    setTotal(0);
    setAnswered(null);
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-indigo-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        操作系统设计原则
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-4">
        掌握 5 大核心设计原则及其在真实 OS 中的应用
      </p>

      <div className="flex gap-3 mb-6 justify-center">
        <button onClick={() => { setMode("learn"); setSelected(null); }} className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${mode === "learn" ? "bg-indigo-500 text-white" : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300"}`}>
          学习模式
        </button>
        <button onClick={() => { setMode("quiz"); resetQuiz(); }} className={`px-4 py-2 rounded-lg text-sm font-bold transition-colors ${mode === "quiz" ? "bg-indigo-500 text-white" : "bg-slate-200 text-slate-600 dark:bg-gray-700 dark:text-gray-300"}`}>
          测验模式
        </button>
      </div>

      {mode === "learn" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-2">
            {PRINCIPLES.map((p) => (
              <button key={p.id} onClick={() => setSelected(selected === p.id ? null : p.id)}
                className={`w-full text-left p-3 rounded-lg border-2 transition-all ${selected === p.id ? "border-indigo-400 bg-indigo-50 dark:bg-indigo-950/30 shadow-md" : "border-transparent bg-white dark:bg-gray-800 hover:border-slate-300 dark:hover:border-gray-600"}`}>
                <div className="flex items-center gap-2">
                  <span className={p.color}>{p.icon}</span>
                  <span className="text-sm font-bold text-slate-700 dark:text-gray-200">{p.title}</span>
                </div>
                <p className="text-xs text-slate-500 dark:text-gray-400 mt-1">{p.summary}</p>
              </button>
            ))}
          </div>

          <div className="lg:col-span-2">
            <AnimatePresence mode="wait">
              {active ? (
                <motion.div key={active.id} initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }}
                  className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-slate-200 dark:border-gray-700 shadow-md">
                  <h3 className={`text-lg font-bold mb-3 ${active.color}`}>{active.title}</h3>
                  <p className="text-sm text-slate-600 dark:text-gray-300 mb-4">{active.definition}</p>
                  <h4 className="text-xs font-bold text-slate-500 dark:text-gray-400 uppercase mb-2">真实 OS 案例</h4>
                  <div className="space-y-2 mb-4">
                    {active.examples.map((ex, i) => (
                      <div key={i} className="flex items-start gap-2 bg-slate-50 dark:bg-gray-750 rounded p-2">
                        <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">{ex.os}</span>
                        <span className="text-xs text-slate-600 dark:text-gray-300">{ex.detail}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 border border-slate-200 dark:border-gray-700 text-center">
                  <Lightbulb className="w-10 h-10 text-slate-300 mx-auto mb-3" />
                  <p className="text-sm text-slate-500 dark:text-gray-400">点击左侧原则查看详细信息</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      ) : (
        /* Quiz Mode */
        <div className="max-w-2xl mx-auto">
          <div className="flex justify-between items-center mb-4">
            <span className="text-sm font-bold text-slate-700 dark:text-gray-200">
              问题 {total + (answered !== null ? 0 : 1)}
            </span>
            <span className="text-sm text-slate-500 dark:text-gray-400">
              得分: {score}/{total}
            </span>
          </div>

          <motion.div key={quizIdx} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-slate-200 dark:border-gray-700 mb-4">
            <div className="flex items-center gap-2 mb-3">
              <span className={currentPrinciple.color}>{currentPrinciple.icon}</span>
              <span className="text-sm font-bold text-slate-700 dark:text-gray-200">{currentPrinciple.title}</span>
            </div>
            <p className="text-sm text-slate-600 dark:text-gray-300 mb-4">{currentPrinciple.quiz[0].scenario}</p>

            <div className="flex gap-3">
              <button onClick={() => handleQuizAnswer(currentPrinciple.quiz[0].correct)} disabled={answered !== null}
                className="flex-1 px-4 py-3 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg font-bold text-sm hover:bg-emerald-200 dark:hover:bg-emerald-900/50 disabled:opacity-50 transition-colors">
                <CheckCircle className="w-4 h-4 inline mr-1" /> 正确做法
              </button>
              <button onClick={() => handleQuizAnswer(!currentPrinciple.quiz[0].correct)} disabled={answered !== null}
                className="flex-1 px-4 py-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg font-bold text-sm hover:bg-red-200 dark:hover:bg-red-900/50 disabled:opacity-50 transition-colors">
                <XCircle className="w-4 h-4 inline mr-1" /> 错误做法
              </button>
            </div>
          </motion.div>

          <AnimatePresence>
            {answered !== null && (
              <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className={`p-4 rounded-lg text-sm ${answered ? "bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700" : "bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-700"}`}>
                {currentPrinciple.quiz[0].explanation}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-4 flex justify-center">
            <button onClick={resetQuiz} className="flex items-center gap-2 px-4 py-2 bg-slate-200 dark:bg-gray-700 text-slate-600 dark:text-gray-300 rounded-lg text-sm">
              <RotateCcw className="w-4 h-4" /> 重新开始
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

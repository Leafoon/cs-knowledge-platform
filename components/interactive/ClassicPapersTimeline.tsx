"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, ExternalLink, ChevronRight, Calendar, Award } from "lucide-react";

interface Paper {
  year: number;
  title: string;
  authors: string;
  venue: string;
  impact: string;
  keyIdea: string;
  osRelevance: string;
  color: string;
}

const PAPERS: Paper[] = [
  { year: 1969, title: "On the Structure of Switching Theory", authors: "Karp & Miller", venue: "ACM SIGOPS", impact: "奠定进程抽象理论基础", keyIdea: "有限状态机描述系统行为", osRelevance: "进程状态机模型", color: "bg-slate-400" },
  { year: 1970, title: "Virtual Memory Implementation", authors: "Denning", venue: "ACM SOSP", impact: "提出工作集模型", keyIdea: "工作集理论解释程序局部性", osRelevance: "页面置换算法的理论基础", color: "bg-blue-400" },
  { year: 1974, title: "The UNIX Time-Sharing System", authors: "Ritchie & Thompson", venue: "CACM", impact: "奠定现代操作系统设计范式", keyIdea: "一切皆文件、管道、shell", osRelevance: "所有类 Unix 系统的设计基础", color: "bg-purple-400" },
  { year: 1978, title: "Time Clocks and Ordering of Events", authors: "Lamport", venue: "CACM", impact: "提出逻辑时钟", keyIdea: "happens-before 关系", osRelevance: "分布式系统时序保证", color: "bg-amber-400" },
  { year: 1990, title: "The Duality of Memory and Communication", authors: "Ousterhout", venue: "ACM SOSP", impact: "线程辩论", keyIdea: "线程 vs 事件驱动的利弊", osRelevance: "现代并发模型选择", color: "bg-rose-400" },
  { year: 1994, title: "Lottery Scheduling", authors: "Waldspurger & Weihl", venue: "OSDI", impact: "提出概率调度", keyIdea: "随机化实现灵活的资源分配", osRelevance: "公平份额调度", color: "bg-emerald-400" },
  { year: 1996, title: "Why Threads Are a Bad Idea", authors: "Ousterhout", venue: "USENIX", impact: "重新审视线程模型", keyIdea: "线程的复杂性不值得", osRelevance: "事件驱动架构", color: "bg-pink-400" },
  { year: 2000, title: "Systems Software Research Is Irrelevant", authors: "Pike", venue: "USENIX", impact: "呼吁关注系统软件", keyIdea: "系统软件是创新的基础", osRelevance: "OS 研究价值论证", color: "bg-cyan-400" },
  { year: 2003, title: "The Google File System", authors: "Ghemawat et al.", venue: "SOSP", impact: "分布式文件系统里程碑", keyIdea: "Master-Chunk 架构", osRelevance: "GFS → HDFS → 现代分布式存储", color: "bg-orange-400" },
  { year: 2009, title: "seL4: Formal Verification", authors: "Klein et al.", venue: "SOSP", impact: "首个经过形式化验证的 OS 内核", keyIdea: "数学证明内核正确性", osRelevance: "高可靠性系统", color: "bg-violet-400" },
  { year: 2016, title: "Scalable Locks for Giant Multicores", authors: "David et al.", venue: "ACM TOPC", impact: "可扩展锁设计", keyIdea: "CLH/MCS 锁在多核下的优势", osRelevance: "现代多核内核锁优化", color: "bg-teal-400" },
];

export default function ClassicPapersTimeline() {
  const [selected, setSelected] = useState<number | null>(null);
  const [compare, setCompare] = useState<number | null>(null);

  const selectedPaper = PAPERS.find((p) => p.year === selected);
  const comparePaper = PAPERS.find((p) => p.year === compare);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-amber-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        经典论文时间线
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        操作系统领域最具影响力的论文 — 点击查看详情
      </p>

      <div className="relative mb-6">
        {/* Timeline line */}
        <div className="absolute top-1/2 left-0 right-0 h-1 bg-slate-200 dark:bg-gray-700 -translate-y-1/2" />

        {/* Year markers */}
        <div className="relative flex justify-between px-2">
          {PAPERS.map((p) => {
            const isActive = selected === p.year;
            const isComparing = compare === p.year;
            return (
              <motion.button key={p.year} onClick={() => setSelected(isActive ? null : p.year)}
                onDoubleClick={() => setCompare(compare === p.year ? null : p.year)}
                className="relative z-10 flex flex-col items-center"
                whileHover={{ scale: 1.1 }}>
                <span className="text-[10px] text-slate-400 dark:text-gray-500 mb-1">{p.year}</span>
                <div className={`w-4 h-4 rounded-full ${p.color} border-2 transition-all ${
                  isActive ? "border-slate-800 dark:border-white scale-150 shadow-lg" :
                  isComparing ? "border-slate-400 scale-125" :
                  "border-white dark:border-gray-600"
                }`} />
                <span className="text-[9px] text-slate-500 dark:text-gray-400 mt-1 max-w-[60px] text-center leading-tight hidden sm:block">
                  {p.title.slice(0, 20)}...
                </span>
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="text-[10px] text-slate-400 text-center mb-4">单击查看 | 双击对比</div>

      <AnimatePresence mode="wait">
        {selectedPaper && (
          <motion.div key={selectedPaper.year} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-slate-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-3">
              <span className={`w-8 h-8 rounded-full ${selectedPaper.color} flex items-center justify-center text-white text-xs font-bold`}>
                {selectedPaper.year.toString().slice(-2)}
              </span>
              <div>
                <h3 className="text-base font-bold text-slate-800 dark:text-gray-100">{selectedPaper.title}</h3>
                <p className="text-xs text-slate-500 dark:text-gray-400">{selectedPaper.authors} · {selectedPaper.venue}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 dark:bg-gray-750 rounded p-3">
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400">核心思想</span>
                <p className="text-xs text-slate-700 dark:text-gray-200 mt-1">{selectedPaper.keyIdea}</p>
              </div>
              <div className="bg-slate-50 dark:bg-gray-750 rounded p-3">
                <span className="text-xs font-bold text-slate-500 dark:text-gray-400">对 OS 的影响</span>
                <p className="text-xs text-slate-700 dark:text-gray-200 mt-1">{selectedPaper.osRelevance}</p>
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <Award className="w-3 h-3 text-amber-500" />
              <span className="text-xs text-amber-600 dark:text-amber-400">{selectedPaper.impact}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {comparePaper && selectedPaper && selectedPaper.year !== comparePaper.year && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="mt-3 bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
          <h4 className="text-sm font-bold text-slate-600 dark:text-gray-300 mb-2">
            对比: {selectedPaper.title} vs {comparePaper.title}
          </h4>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-50 dark:bg-gray-750 rounded p-2">
              <span className="font-bold text-slate-600">{selectedPaper.year}: {selectedPaper.keyIdea}</span>
            </div>
            <div className="bg-slate-50 dark:bg-gray-750 rounded p-2">
              <span className="font-bold text-slate-600">{comparePaper.year}: {comparePaper.keyIdea}</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

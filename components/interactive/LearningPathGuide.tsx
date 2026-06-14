"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GraduationCap, BookOpen, Code2, Trophy, CheckCircle2, Circle, ChevronDown } from "lucide-react";

interface Skill {
  id: string;
  name: string;
  required: string[];
  resources: string[];
}

interface Stage {
  id: string;
  level: string;
  title: string;
  icon: React.ReactNode;
  color: string;
  skills: Skill[];
}

const STAGES: Stage[] = [
  {
    id: "foundations", level: "初级", title: "基础概念", icon: <BookOpen className="w-5 h-5" />,
    color: "bg-blue-500",
    skills: [
      { id: "process", name: "进程与线程", required: [], resources: ["OSTEP Ch2-5", "xv6 labs"] },
      { id: "memory", name: "虚拟内存", required: [], resources: ["OSTEP Ch13-20"] },
      { id: "io", name: "I/O 与文件系统", required: [], resources: ["OSTEP Ch36-40"] },
      { id: "scheduling", name: "CPU 调度", required: [], resources: ["OSTEP Ch6-12"] },
    ],
  },
  {
    id: "concurrency", level: "中级", title: "并发与同步", icon: <Code2 className="w-5 h-5" />,
    color: "bg-amber-500",
    skills: [
      { id: "locks", name: "锁与互斥", required: ["process"], resources: ["OSTEP Ch26-28"] },
      { id: "sync", name: "信号量与条件变量", required: ["locks"], resources: ["OSTEP Ch30-33"] },
      { id: "deadlock", name: "死锁", required: ["locks"], resources: ["OSTEP Ch34-35"] },
    ],
  },
  {
    id: "implementation", level: "高级", title: "内核实现", icon: <GraduationCap className="w-5 h-5" />,
    color: "bg-purple-500",
    skills: [
      { id: "xv6-trap", name: "xv6 陷阱处理", required: ["process"], resources: ["MIT 6.S081 Lab 4"] },
      { id: "xv6-fs", name: "xv6 文件系统", required: ["io"], resources: ["MIT 6.S081 Lab 8-10"] },
      { id: "xv6-mem", name: "xv6 内存管理", required: ["memory"], resources: ["MIT 6.S081 Lab 3"] },
    ],
  },
  {
    id: "expert", level: "专家", title: "前沿研究", icon: <Trophy className="w-5 h-5" />,
    color: "bg-rose-500",
    skills: [
      { id: "multicore", name: "多核可扩展性", required: ["locks", "sync"], resources: ["Linux kernel source", "SOSP/OSDI papers"] },
      { id: "virtualization", name: "虚拟化技术", required: ["memory", "process"], resources: ["KVM source", "Docker internals"] },
      { id: "distributed", name: "分布式系统", required: ["sync", "deadlock"], resources: ["Raft paper", "GFS paper"] },
    ],
  },
];

export default function LearningPathGuide() {
  const [completed, setCompleted] = useState<Set<string>>(new Set());
  const [expandedStage, setExpandedStage] = useState<string>("foundations");

  const isSkillUnlocked = (skill: Skill) =>
    skill.required.length === 0 || skill.required.every((r) => completed.has(r));

  const totalSkills = STAGES.reduce((acc, s) => acc + s.skills.length, 0);
  const completedCount = completed.size;

  const toggleSkill = (id: string) => {
    setCompleted((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-violet-50 rounded-xl shadow-lg dark:from-gray-900 dark:to-gray-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-gray-100 mb-2 text-center">
        操作系统学习路径
      </h2>
      <p className="text-sm text-slate-500 dark:text-gray-400 text-center mb-6">
        从入门到专家 — 勾选已掌握的技能，解锁下一阶段
      </p>

      {/* Progress bar */}
      <div className="mb-6 bg-white dark:bg-gray-800 rounded-xl p-4 border border-slate-200 dark:border-gray-700">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-bold text-slate-700 dark:text-gray-200">总进度</span>
          <span className="text-sm text-slate-500 dark:text-gray-400">{completedCount}/{totalSkills}</span>
        </div>
        <div className="w-full h-3 bg-slate-100 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
            animate={{ width: `${(completedCount / totalSkills) * 100}%` }} transition={{ duration: 0.5 }} />
        </div>
      </div>

      {/* Stages */}
      <div className="space-y-3">
        {STAGES.map((stage) => {
          const stageCompleted = stage.skills.every((s) => completed.has(s.id));
          const stageProgress = stage.skills.filter((s) => completed.has(s.id)).length;
          const isExpanded = expandedStage === stage.id;

          return (
            <div key={stage.id} className="bg-white dark:bg-gray-800 rounded-xl border border-slate-200 dark:border-gray-700 overflow-hidden">
              <button onClick={() => setExpandedStage(isExpanded ? "" : stage.id)}
                className="w-full p-4 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-gray-750 transition-colors">
                <div className={`w-10 h-10 rounded-lg ${stage.color} flex items-center justify-center text-white`}>
                  {stage.icon}
                </div>
                <div className="flex-1 text-left">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-800 dark:text-gray-100">{stage.title}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-100 dark:bg-gray-700 text-slate-500">{stage.level}</span>
                    {stageCompleted && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                  </div>
                  <span className="text-xs text-slate-400">{stageProgress}/{stage.skills.length} 技能</span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }}
                    className="border-t border-slate-100 dark:border-gray-700">
                    <div className="p-4 space-y-2">
                      {stage.skills.map((skill) => {
                        const unlocked = isSkillUnlocked(skill);
                        const done = completed.has(skill.id);
                        return (
                          <div key={skill.id} className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                            done ? "bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800" :
                            unlocked ? "bg-slate-50 dark:bg-gray-750 border border-slate-200 dark:border-gray-600" :
                            "bg-slate-50 dark:bg-gray-750 border border-slate-100 dark:border-gray-700 opacity-50"
                          }`}>
                            <button onClick={() => unlocked && toggleSkill(skill.id)} className="shrink-0">
                              {done ? <CheckCircle2 className="w-5 h-5 text-emerald-500" /> : <Circle className="w-5 h-5 text-slate-300 dark:text-gray-600" />}
                            </button>
                            <div className="flex-1">
                              <span className={`text-sm font-bold ${done ? "text-emerald-700 dark:text-emerald-300" : "text-slate-700 dark:text-gray-200"}`}>
                                {skill.name}
                              </span>
                              {!unlocked && (
                                <span className="ml-2 text-[10px] text-slate-400">
                                  需要: {skill.required.map((r) => STAGES.flatMap((s) => s.skills).find((sk) => sk.id === r)?.name).join(", ")}
                                </span>
                              )}
                            </div>
                            <div className="text-[10px] text-slate-400 text-right">
                              {skill.resources.join(" | ")}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {completedCount === totalSkills && (
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
          className="mt-6 text-center p-6 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30 rounded-xl border border-amber-200 dark:border-amber-800">
          <Trophy className="w-10 h-10 text-amber-500 mx-auto mb-2" />
          <p className="text-lg font-bold text-amber-700 dark:text-amber-300">恭喜！所有技能已掌握！</p>
        </motion.div>
      )}
    </div>
  );
}

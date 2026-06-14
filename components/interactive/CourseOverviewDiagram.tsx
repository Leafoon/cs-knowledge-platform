"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, BookOpen, Cpu, Binary, Calculator, CircuitBoard, Layers } from "lucide-react";

const modules = [
  {
    id: "intro",
    title: "计算机系统概述",
    icon: BookOpen,
    color: "#667eea",
    chapters: [
      { id: "0-1", title: "计算机发展历程" },
      { id: "0-2", title: "计算机系统层次结构" },
      { id: "0-3", title: "计算机性能指标" },
    ],
  },
  {
    id: "data",
    title: "数据的表示和运算",
    icon: Binary,
    color: "#f59e0b",
    chapters: [
      { id: "2-1", title: "数制与编码" },
      { id: "2-2", title: "定点数的表示和运算" },
      { id: "2-3", title: "浮点数的表示和运算" },
      { id: "2-4", title: "算术逻辑单元ALU" },
    ],
  },
  {
    id: "memory",
    title: "存储器层次结构",
    icon: Layers,
    color: "#10b981",
    chapters: [
      { id: "3-1", title: "存储器概述" },
      { id: "3-2", title: "主存储器" },
      { id: "3-3", title: "高速缓冲存储器" },
      { id: "3-4", title: "虚拟存储器" },
    ],
  },
  {
    id: "inst",
    title: "指令系统",
    icon: CircuitBoard,
    color: "#ef4444",
    chapters: [
      { id: "4-1", title: "指令格式" },
      { id: "4-2", title: "寻址方式" },
      { id: "4-3", title: "CISC与RISC" },
    ],
  },
  {
    id: "cpu",
    title: "中央处理器",
    icon: Cpu,
    color: "#8b5cf6",
    chapters: [
      { id: "5-1", title: "CPU的功能和基本结构" },
      { id: "5-2", title: "指令执行过程" },
      { id: "5-3", title: "数据通路的功能和基本结构" },
      { id: "5-4", title: "控制器的功能和工作原理" },
      { id: "5-5", title: "指令流水线" },
    ],
  },
  {
    id: "bus",
    title: "总线与I/O",
    icon: Calculator,
    color: "#ec4899",
    chapters: [
      { id: "6-1", title: "总线" },
      { id: "6-2", title: "I/O系统" },
      { id: "6-3", title: "I/O方式" },
    ],
  },
];

export function CourseOverviewDiagram() {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <h3 className="text-xl font-semibold mb-4 text-text-primary">
        课程知识体系总览
      </h3>
      <p className="text-sm text-text-secondary mb-6">
        点击模块查看子章节，了解计算机组成原理的完整知识架构
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {modules.map((mod) => {
          const Icon = mod.icon;
          const isOpen = expanded === mod.id;
          return (
            <motion.div
              key={mod.id}
              layout
              className="rounded-lg border border-border-subtle bg-bg-secondary overflow-hidden cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => setExpanded(isOpen ? null : mod.id)}
            >
              <div className="flex items-center gap-3 p-4">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: mod.color + "20", color: mod.color }}
                >
                  <Icon size={20} />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-sm text-text-primary">{mod.title}</h4>
                  <p className="text-xs text-text-secondary">{mod.chapters.length} 个知识点</p>
                </div>
                <motion.div
                  animate={{ rotate: isOpen ? 90 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronRight size={16} className="text-text-secondary" />
                </motion.div>
              </div>
              <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 space-y-2">
                      {mod.chapters.map((ch, i) => (
                        <motion.div
                          key={ch.id}
                          initial={{ x: -10, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{ delay: i * 0.05 }}
                          className="flex items-center gap-2 text-sm text-text-secondary"
                        >
                          <span
                            className="w-6 h-6 rounded flex items-center justify-center text-xs font-mono font-bold"
                            style={{ backgroundColor: mod.color + "15", color: mod.color }}
                          >
                            {ch.id}
                          </span>
                          <span>{ch.title}</span>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

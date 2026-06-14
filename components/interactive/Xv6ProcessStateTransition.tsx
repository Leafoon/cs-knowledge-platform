"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MousePointerClick, ArrowRight, RotateCcw } from "lucide-react";

interface StateInfo {
  id: string;
  desc: string;
  code: string;
}

interface Transition {
  from: string;
  to: string;
  label: string;
  func: string;
  detail: string;
}

const stateDetails: StateInfo[] = [
  { id: "UNUSED", desc: "PCB 槽位空闲，可被分配", code: "p->state == UNUSED" },
  { id: "USED", desc: "正在分配，初始化内核栈和页表", code: "p->state = USED" },
  { id: "SLEEPING", desc: "等待事件（I/O/信号），让出 CPU", code: "p->state = SLEEPING" },
  { id: "RUNNABLE", desc: "可运行，等待调度器选中", code: "p->state = RUNNABLE" },
  { id: "RUNNING", desc: "正在 CPU 上执行用户/内核代码", code: "p->state = RUNNING" },
  { id: "ZOMBIE", desc: "已退出，等待父进程 wait() 回收", code: "p->state = ZOMBIE" },
];

const states = [
  { id: "UNUSED", color: "#6b7280", x: 60, y: 200 },
  { id: "USED", color: "#8b5cf6", x: 220, y: 80 },
  { id: "SLEEPING", color: "#3b82f6", x: 400, y: 80 },
  { id: "RUNNABLE", color: "#10b981", x: 400, y: 320 },
  { id: "RUNNING", color: "#f59e0b", x: 220, y: 320 },
  { id: "ZOMBIE", color: "#ef4444", x: 60, y: 320 },
];

const transitions: Transition[] = [
  { from: "UNUSED", to: "USED", label: "allocproc()", func: "allocproc()", detail: "分配 PCB 槽位，初始化内核栈和 trapframe" },
  { from: "USED", to: "RUNNABLE", label: "forkret()", func: "forkret()", detail: "进程初始化完成，设为可调度状态" },
  { from: "RUNNABLE", to: "RUNNING", label: "scheduler()", func: "scheduler()", detail: "调度器选中该进程，调用 swtch() 切换上下文" },
  { from: "RUNNING", to: "RUNNABLE", label: "yield()", func: "yield()", detail: "时间片用完，定时器中断触发 yield()" },
  { from: "RUNNING", to: "SLEEPING", label: "sleep()", func: "sleep(chan, lock)", detail: "等待 I/O 或其他事件，释放锁并让出 CPU" },
  { from: "SLEEPING", to: "RUNNABLE", label: "wakeup()", func: "wakeup(chan)", detail: "等待的事件发生，遍历进程表唤醒匹配的进程" },
  { from: "RUNNING", to: "ZOMBIE", label: "exit()", func: "exit(status)", detail: "进程主动退出，关闭文件，转交子进程给 init" },
  { from: "ZOMBIE", to: "UNUSED", label: "wait()", func: "wait()", detail: "父进程回收子进程资源，调用 freeproc()" },
];

const stateMap = new Map(states.map((s) => [s.id, s]));

export default function Xv6ProcessStateTransition() {
  const [selected, setSelected] = useState<Transition | null>(null);
  const [activeTransition, setActiveTransition] = useState<string | null>(null);
  const [hoveredState, setHoveredState] = useState<string | null>(null);

  const handleTransitionClick = (t: Transition) => {
    setSelected(t);
    setActiveTransition(`${t.from}-${t.to}`);
  };

  const handleStateClick = (stateId: string) => {
    setHoveredState(hoveredState === stateId ? null : stateId);
    setSelected(null);
    setActiveTransition(null);
  };

  const reset = () => {
    setSelected(null);
    setActiveTransition(null);
    setHoveredState(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white dark:bg-slate-900 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <MousePointerClick className="w-5 h-5 text-amber-500" />
          xv6 进程状态转换图
        </h3>
        <button
          onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
        >
          <RotateCcw className="w-3 h-3" /> 重置
        </button>
      </div>

      <svg viewBox="0 0 560 420" className="w-full h-auto">
        <defs>
          <marker id="arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#f59e0b" />
          </marker>
        </defs>

        {transitions.map((t) => {
          const from = stateMap.get(t.from)!;
          const to = stateMap.get(t.to)!;
          const isActive = activeTransition === `${t.from}-${t.to}`;
          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const len = Math.sqrt(dx * dx + dy * dy);
          const offX = (dx / len) * 45;
          const offY = (dy / len) * 45;
          const mx = (from.x + to.x) / 2;
          const my = (from.y + to.y) / 2;
          const perpX = -(dy / len) * 20;
          const perpY = (dx / len) * 20;

          return (
            <g key={`${t.from}-${t.to}`} onClick={() => handleTransitionClick(t)} className="cursor-pointer">
              <line
                x1={from.x + offX}
                y1={from.y + offY}
                x2={to.x - offX}
                y2={to.y - offY}
                stroke={isActive ? "#f59e0b" : "#cbd5e1"}
                strokeWidth={isActive ? 3 : 1.5}
                markerEnd={isActive ? "url(#arrow-active)" : "url(#arrow)"}
                className="dark:stroke-slate-600 transition-all"
              />
              <rect
                x={mx + perpX - 32}
                y={my + perpY - 10}
                width={64}
                height={20}
                rx={4}
                fill={isActive ? "#fef3c7" : "#f8fafc"}
                stroke={isActive ? "#f59e0b" : "#e2e8f0"}
                strokeWidth={1}
                className="dark:fill-slate-800 dark:stroke-slate-600"
              />
              <text
                x={mx + perpX}
                y={my + perpY + 4}
                textAnchor="middle"
                fontSize={9}
                fill={isActive ? "#b45309" : "#64748b"}
                fontWeight={isActive ? 700 : 400}
                className="dark:fill-slate-300 select-none"
              >
                {t.label}
              </text>
            </g>
          );
        })}

        {states.map((s) => (
          <g key={s.id} onClick={() => handleStateClick(s.id)} className="cursor-pointer">
            <circle cx={s.x} cy={s.y} r={40} fill={s.color} opacity={hoveredState === s.id ? 0.3 : 0.15} />
            <circle cx={s.x} cy={s.y} r={40} fill="none" stroke={s.color} strokeWidth={hoveredState === s.id ? 3 : 2} />
            <text x={s.x} y={s.y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill={s.color} className="select-none">
              {s.id}
            </text>
          </g>
        ))}

        <text x={280} y={410} textAnchor="middle" fontSize={11} fill="#94a3b8" className="select-none">
          点击箭头查看状态转换详情
        </text>
      </svg>

      <AnimatePresence mode="wait">
        {selected && (
          <motion.div
            key={`${selected.from}-${selected.to}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-xs font-mono font-bold text-white" style={{ backgroundColor: stateMap.get(selected.from)?.color }}>
                {selected.from}
              </span>
              <ArrowRight className="w-4 h-4 text-amber-500" />
              <span className="px-2 py-0.5 rounded text-xs font-mono font-bold text-white" style={{ backgroundColor: stateMap.get(selected.to)?.color }}>
                {selected.to}
              </span>
            </div>
            <p className="text-sm font-mono text-amber-700 dark:text-amber-300 mb-1">
              触发函数：<code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">{selected.func}</code>
            </p>
            <p className="text-sm text-slate-600 dark:text-slate-300">{selected.detail}</p>
          </motion.div>
        )}
        {hoveredState && !selected && (
          <motion.div
            key={`state-${hoveredState}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 p-4 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-xs font-mono font-bold text-white" style={{ backgroundColor: stateMap.get(hoveredState)?.color }}>
                {hoveredState}
              </span>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-300 mb-1">
              {stateDetails.find((s) => s.id === hoveredState)?.desc}
            </p>
            <p className="text-xs font-mono text-slate-500 dark:text-slate-400">
              代码：<code className="bg-slate-100 dark:bg-slate-800 px-1 rounded">{stateDetails.find((s) => s.id === hoveredState)?.code}</code>
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-4 grid grid-cols-3 gap-2">
        {stateDetails.map((s) => (
          <button
            key={s.id}
            onClick={() => handleStateClick(s.id)}
            className={`p-2 rounded-lg border text-left transition-colors ${
              hoveredState === s.id
                ? "border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20"
                : "border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50"
            }`}
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: stateMap.get(s.id)?.color }} />
              <span className="text-xs font-mono font-bold text-slate-700 dark:text-slate-200">{s.id}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-tight">{s.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Layers } from "lucide-react";

const priorities = [
  { level: 3, name: "紧急故障", color: "red", desc: "电源故障、硬件错误" },
  { level: 2, name: "磁盘I/O", color: "orange", desc: "磁盘数据传输完成" },
  { level: 1, name: "网络I/O", color: "yellow", desc: "网络数据包到达" },
  { level: 0, name: "键盘输入", color: "blue", desc: "键盘按键事件" },
];

export function NestedInterruptHandler() {
  const [mask, setMask] = useState(0b1111);
  const [activeInterrupts, setActiveInterrupts] = useState<number[]>([]);
  const [execStack, setExecStack] = useState<{ level: number; phase: "entry" | "service" | "exit" }[]>([]);

  const toggleMask = (bit: number) => {
    setMask((m) => m ^ (1 << bit));
  };

  const triggerInterrupt = (level: number) => {
    const effectiveMask = mask;
    if (!(effectiveMask & (1 << level))) return;
    setActiveInterrupts((prev) => [...prev, level].sort((a, b) => b - a));
    setExecStack((prev) => [...prev, { level, phase: "entry" }]);
  };

  const stepExecution = () => {
    setExecStack((prev) => {
      if (prev.length === 0) return prev;
      const top = prev[prev.length - 1];
      if (top.phase === "entry") return [...prev.slice(0, -1), { ...top, phase: "service" }];
      if (top.phase === "service") return [...prev.slice(0, -1), { ...top, phase: "exit" }];
      return prev.slice(0, -1);
    });
    if (execStack.length > 0 && execStack[execStack.length - 1].phase === "exit") {
      setActiveInterrupts((prev) => prev.slice(1));
    }
  };

  const currentExecuting = execStack.length > 0 ? execStack[execStack.length - 1] : null;

  return (
    <div className="my-8 border border-border-subtle rounded-lg p-6 bg-bg-elevated">
      <div className="flex items-center gap-2 mb-4">
        <Layers className="w-5 h-5 text-red-400" />
        <h3 className="text-lg font-semibold">嵌套中断处理</h3>
      </div>

      <div className="mb-4">
        <div className="text-xs text-gray-400 mb-2">中断屏蔽字 (点击切换)</div>
        <div className="flex gap-2">
          {priorities.map((p) => (
            <button
              key={p.level}
              onClick={() => toggleMask(p.level)}
              className={`px-3 py-2 rounded text-xs font-mono ${
                mask & (1 << p.level)
                  ? `bg-${p.color}-600/30 border border-${p.color}-500 text-${p.color}-300`
                  : "bg-gray-800 border border-gray-600 text-gray-500"
              }`}
            >
              <Shield className="w-3 h-3 inline mr-1" />
              L{p.level}: {mask & (1 << p.level) ? "开放" : "屏蔽"}
            </button>
          ))}
        </div>
        <div className="text-[10px] text-gray-500 mt-1 font-mono">屏蔽字: {mask.toString(2).padStart(4, "0")}</div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4">
        {priorities.map((p) => {
          const isActive = activeInterrupts.includes(p.level);
          const isBlocked = !(mask & (1 << p.level));
          return (
            <motion.button
              key={p.level}
              onClick={() => triggerInterrupt(p.level)}
              disabled={isBlocked}
              className={`p-3 rounded-lg border text-xs ${
                isBlocked ? "opacity-40 cursor-not-allowed border-gray-700 bg-gray-800/20" :
                isActive ? `border-${p.color}-500 bg-${p.color}-500/10` : "border-gray-600 bg-gray-800/30 hover:bg-gray-700/50"
              }`}
              whileHover={!isBlocked ? { scale: 1.03 } : undefined}
            >
              <div className={`text-${p.color}-400 font-medium`}>L{p.level}: {p.name}</div>
              <div className="text-gray-400 mt-1">{p.desc}</div>
              {isActive && <div className={`mt-2 w-2 h-2 rounded-full bg-${p.color}-400 mx-auto animate-pulse`} />}
            </motion.button>
          );
        })}
      </div>

      <div className="flex gap-2 mb-4">
        <button onClick={stepExecution} disabled={execStack.length === 0}
          className="px-4 py-1.5 bg-red-600 rounded text-sm text-white hover:bg-red-500 disabled:opacity-50">
          执行一步
        </button>
        <button onClick={() => { setActiveInterrupts([]); setExecStack([]); }}
          className="px-3 py-1.5 bg-gray-700 rounded text-sm text-gray-300 hover:bg-gray-600">
          清空
        </button>
      </div>

      {execStack.length > 0 && (
        <div className="p-3 bg-gray-800/30 rounded-lg">
          <div className="text-xs text-gray-400 mb-2">中断嵌套栈:</div>
          <div className="flex gap-1">
            {execStack.map((e, i) => {
              const p = priorities[e.level];
              return (
                <motion.div key={`${e.level}-${e.phase}`}
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className={`px-2 py-1 rounded text-xs bg-${p?.color || "gray"}-600/20 text-${p?.color || "gray"}-300`}
                >
                  L{e.level} [{e.phase === "entry" ? "保存现场" : e.phase === "service" ? "服务中" : "恢复返回"}]
                </motion.div>
              );
            })}
          </div>
          {currentExecuting && (
            <div className="text-xs text-gray-400 mt-2">
              当前执行: L{currentExecuting.level} - {priorities[currentExecuting.level]?.name}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
